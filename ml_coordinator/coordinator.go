// ml_coordinator/coordinator.go — Training coordinator.
//
// The coordinator is the "brain" of the distributed training cluster.
// It does three things:
//
// 1. HEARTBEAT TRACKING
//    Workers send a heartbeat every 5 seconds. If a worker misses 3 heartbeats
//    it is declared dead. This is the same timeout logic the server uses for
//    client leases — just applied at the training-workflow level.
//
// 2. CRASH RECOVERY ORCHESTRATION
//    When a worker dies, the coordinator:
//      a. Reads the dead worker's last committed step from the DFS.
//      b. Logs this for the operator / restart script.
//      c. If a replacement worker starts and calls RegisterWorker(), the
//         coordinator tells it which step to resume from.
//    Without this, the replacement worker would restart from step 0 and waste
//    all previous compute. With this, it restarts from the last checkpoint.
//
// 3. GLOBAL SNAPSHOT
//    Periodically calls InitiateSnapshot on the primary DFS server.
//    The snapshot captures: which step each worker is on, which tensor hashes
//    are stored, and the full replication state. This is the Chandy-Lamport
//    snapshot you already implemented — extended to also capture ML state.
//
// 4. HTTP DASHBOARD
//    Exposes /status on port 8080 showing:
//      - Each worker's current step and last heartbeat time
//      - Dedup ratio (new tensors / total tensors)
//      - Whether any worker is dead

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	ikkat "distributed-system-ikkat"
	pb "distributed-system-ikkat/filesystem"

	"google.golang.org/grpc"
)

const (
	HeartbeatTimeout   = 15 * time.Second // 3 missed heartbeats at 5s interval
	SnapshotInterval   = 2 * time.Minute
	StragglerThreshold = 200 // if a worker is this many steps behind the max, it's a straggler
)

// WorkerStatus tracks the live state of one training worker.
type WorkerStatus struct {
	WorkerID      string
	CurrentStep   int
	LastHeartbeat time.Time
	Alive         bool
	DeadSince     *time.Time
	ResumeStep    int // the step a replacement should start from
}

// Coordinator manages the training cluster lifecycle.
type Coordinator struct {
	mu      sync.Mutex
	workers map[string]*WorkerStatus

	// DFS client for reading checkpoint state and triggering snapshots.
	dfsConn      *grpc.ClientConn
	snapClient   pb.SnapshotServiceClient
	serverHandle *ikkat.ServerHandle

	// snapshotID counter
	snapCount int
}

func NewCoordinator(dfsAddresses []string) (*Coordinator, error) {
	c, conn, err := ikkat.DialClient(dfsAddresses)
	if err != nil {
		return nil, fmt.Errorf("connect to DFS: %w", err)
	}

	// conn is *grpc.ClientConn — pass it directly to NewSnapshotServiceClient.
	// handle.GRPCClient() returns pb.FileServiceClient (a stub), NOT the raw
	// connection — those are different types and not interchangeable.
	snapClient := pb.NewSnapshotServiceClient(conn)

	coord := &Coordinator{
		workers:      make(map[string]*WorkerStatus),
		dfsConn:      conn,
		snapClient:   snapClient,
		serverHandle: ikkat.NewServerHandle(c),
	}
	return coord, nil
}

// RegisterWorker is called by a worker on startup.
// If the worker has crashed before, the coordinator returns the resume step.
func (c *Coordinator) RegisterWorker(workerID string) (resumeStep int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if status, ok := c.workers[workerID]; ok && !status.Alive {
		// This is a restart of a dead worker.
		log.Printf("Coordinator: worker %s restarted, last committed step was %d",
			workerID, status.ResumeStep)
		status.Alive = true
		status.LastHeartbeat = time.Now()
		status.DeadSince = nil
		return status.ResumeStep
	}

	// Fresh worker.
	c.workers[workerID] = &WorkerStatus{
		WorkerID:      workerID,
		CurrentStep:   0,
		LastHeartbeat: time.Now(),
		Alive:         true,
	}
	log.Printf("Coordinator: new worker %s registered", workerID)
	return 0
}

// Heartbeat is called by workers every 5 seconds with their current step.
func (c *Coordinator) Heartbeat(workerID string, step int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	status, ok := c.workers[workerID]
	if !ok {
		c.workers[workerID] = &WorkerStatus{
			WorkerID:      workerID,
			CurrentStep:   step,
			LastHeartbeat: time.Now(),
			Alive:         true,
		}
		return
	}
	status.LastHeartbeat = time.Now()
	status.CurrentStep = step
	status.Alive = true
}

// MonitorWorkers runs in a goroutine, checking for dead workers periodically.
func (c *Coordinator) MonitorWorkers(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.checkDeadWorkers(ctx)
			c.checkStragglers()
		}
	}
}

func (c *Coordinator) checkDeadWorkers(ctx context.Context) {
	c.mu.Lock()
	var dead []string
	for id, status := range c.workers {
		if status.Alive && time.Since(status.LastHeartbeat) > HeartbeatTimeout {
			status.Alive = false
			now := time.Now()
			status.DeadSince = &now
			dead = append(dead, id)
		}
	}
	c.mu.Unlock()

	for _, workerID := range dead {
		log.Printf("Coordinator: worker %s DIED (heartbeat timeout)", workerID)

		// Read the dead worker's last committed step from the DFS.
		meta, err := c.serverHandle.LoadCheckpointMeta(ctx, workerID, -1)
		if err != nil {
			log.Printf("Coordinator: could not load last checkpoint for %s: %v", workerID, err)
			continue
		}

		c.mu.Lock()
		if status, ok := c.workers[workerID]; ok {
			status.ResumeStep = meta.Step
		}
		c.mu.Unlock()

		log.Printf("Coordinator: worker %s can resume from step %d", workerID, meta.Step)
	}
}

func (c *Coordinator) checkStragglers() {
	c.mu.Lock()
	defer c.mu.Unlock()

	maxStep := 0
	for _, status := range c.workers {
		if status.Alive && status.CurrentStep > maxStep {
			maxStep = status.CurrentStep
		}
	}

	for id, status := range c.workers {
		if !status.Alive {
			continue
		}
		lag := maxStep - status.CurrentStep
		if lag > StragglerThreshold {
			log.Printf("Coordinator: STRAGGLER detected: worker %s is %d steps behind (at %d, max %d)",
				id, lag, status.CurrentStep, maxStep)
		}
	}
}

// TriggerSnapshot periodically initiates a Chandy-Lamport snapshot.
func (c *Coordinator) TriggerSnapshot(ctx context.Context) {
	ticker := time.NewTicker(SnapshotInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.mu.Lock()
			c.snapCount++
			snapID := fmt.Sprintf("training-snapshot-%d", c.snapCount)
			c.mu.Unlock()

			log.Printf("Coordinator: initiating Chandy-Lamport snapshot %s", snapID)
			snapCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
			resp, err := c.snapClient.InitiateSnapshot(snapCtx, &pb.SnapshotRequest{
				SnapshotId: snapID,
			})
			cancel()
			if err != nil {
				log.Printf("Coordinator: snapshot %s failed: %v", snapID, err)
				continue
			}
			if resp.Success {
				log.Printf("Coordinator: snapshot %s completed successfully", snapID)
			}
		}
	}
}

// StatusResponse is served by the HTTP dashboard.
type StatusResponse struct {
	Workers      []*WorkerStatus `json:"workers"`
	MaxStep      int             `json:"max_step"`
	MinAliveStep int             `json:"min_alive_step"`
	DeadWorkers  []string        `json:"dead_workers"`
	Timestamp    time.Time       `json:"timestamp"`
}

func (c *Coordinator) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	c.mu.Lock()
	defer c.mu.Unlock()

	resp := StatusResponse{
		Timestamp:    time.Now(),
		MinAliveStep: 1<<31 - 1,
	}

	for _, status := range c.workers {
		resp.Workers = append(resp.Workers, status)
		if !status.Alive {
			resp.DeadWorkers = append(resp.DeadWorkers, status.WorkerID)
			continue
		}
		if status.CurrentStep > resp.MaxStep {
			resp.MaxStep = status.CurrentStep
		}
		if status.CurrentStep < resp.MinAliveStep {
			resp.MinAliveStep = status.CurrentStep
		}
	}
	if resp.MinAliveStep == 1<<31-1 {
		resp.MinAliveStep = 0
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func main() {
	servers := flag.String("servers", "localhost:5001,localhost:5002,localhost:5003",
		"comma-separated DFS server addresses")
	httpPort := flag.String("http", "8080", "HTTP dashboard port")
	flag.Parse()

	addrs := splitAddresses(*servers)
	coord, err := NewCoordinator(addrs)
	if err != nil {
		log.Fatalf("Failed to create coordinator: %v", err)
	}
	defer coord.dfsConn.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start background routines.
	go coord.MonitorWorkers(ctx)
	go coord.TriggerSnapshot(ctx)

	// Start HTTP dashboard.
	http.Handle("/status", coord)
	http.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		workerID := r.URL.Query().Get("id")
		if workerID == "" {
			http.Error(w, "missing id", http.StatusBadRequest)
			return
		}
		resumeStep := coord.RegisterWorker(workerID)
		json.NewEncoder(w).Encode(map[string]int{"resume_step": resumeStep})
	})
	http.HandleFunc("/heartbeat", func(w http.ResponseWriter, r *http.Request) {
		workerID := r.URL.Query().Get("id")
		step := 0
		fmt.Sscanf(r.URL.Query().Get("step"), "%d", &step)
		if workerID == "" {
			http.Error(w, "missing id", http.StatusBadRequest)
			return
		}
		coord.Heartbeat(workerID, step)
		w.WriteHeader(http.StatusOK)
	})

	log.Printf("Coordinator started — dashboard at http://localhost:%s/status", *httpPort)
	log.Fatal(http.ListenAndServe(":"+*httpPort, nil))
}

func splitAddresses(s string) []string {
	var result []string
	current := ""
	for _, c := range s {
		if c == ',' {
			if current != "" {
				result = append(result, current)
			}
			current = ""
		} else {
			current += string(c)
		}
	}
	if current != "" {
		result = append(result, current)
	}
	return result
}
