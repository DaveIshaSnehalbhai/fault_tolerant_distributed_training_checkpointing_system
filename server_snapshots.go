package ikkat

// Chandy-Lamport Snapshot for Client Failure Handling
//
// WHY: When a client crashes mid-processing (e.g. after appending to local
// cache but before Close/Commit reaches the server), we need a consistent
// global snapshot of:
//   1. Which files are open by which clients
//   2. The version each client last committed
//   3. The current prime set per file (dedup state)
//
// This lets an operator or automated recovery process answer:
//   "Did client X actually commit its writes before it died?"
//
// HOW (Chandy-Lamport adapted for this system):
//   - The primary server is the snapshot initiator (it is the single
//     coordinator in this primary-backup model).
//   - "Channels" in C-L terms are the gRPC connections between primary and
//     each backup, plus the virtual channels from each open client to the
//     server (represented by the openMap + files maps).
//   - Step 1: Primary records its own local state atomically.
//   - Step 2: Primary sends MARKER messages to all backups via SendMarker RPC.
//   - Step 3: Each backup, upon receiving the first MARKER, records its own
//     state and forwards MARKERs on all its outgoing channels (in this
//     primary-backup design backups have no outgoing write channels, so they
//     just ACK).
//   - Step 4: Primary collects ACKs. Once all backups have ACKed, the global
//     snapshot is complete and saved to disk as a JSON file.
//
// CLIENT FAILURE RECOVERY using the snapshot:
//   When a client is detected as dead (lease expired), the snapshot tells us:
//     - Was the client's FD captured in the snapshot with Dirty=true?
//     - If yes: the server has the last committed version; the client's
//       local-only append is lost. The server file is consistent.
//     - If no: the client had already closed cleanly before the snapshot.
//   The operator can re-run the failed client with FAIL_AFTER_PROCESSED
//   unset; the client reads its checkpoint file and resumes from where it left
//   off, skipping already-committed work.

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	pb "distributed-system-ikkat/filesystem"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// SnapshotClientInfo captures one open-file slot held by a client at snapshot
// time.
type SnapshotClientInfo struct {
	ClientID string      `json:"client_id"`
	Filename string      `json:"filename"`
	FD       int32       `json:"fd"`
	Mode     pb.FileMode `json:"mode"`
	Version  int32       `json:"version"` // server's version at snapshot time
}

// SnapshotFileInfo captures the dedup state of one output file.
type SnapshotFileInfo struct {
	Filename  string   `json:"filename"`
	Version   int32    `json:"version"`
	PrimeSet  []uint64 `json:"prime_set"`  // all primes already written
	CommitIdx int      `json:"commit_idx"` // log index at snapshot time
}

// GlobalSnapshot is the on-disk representation of a complete Chandy-Lamport
// snapshot.
type GlobalSnapshot struct {
	SnapshotID  string               `json:"snapshot_id"`
	TakenAt     time.Time            `json:"taken_at"`
	ServerID    string               `json:"server_id"`
	CommitIndex int                  `json:"commit_index"`
	OpenClients []SnapshotClientInfo `json:"open_clients"`
	Files       []SnapshotFileInfo   `json:"files"`
	// ML training state captured at snapshot time (derived from disk —
	// see InitiateSnapshot for how these are populated)
	WorkerSteps map[string]int `json:"worker_steps"`
	TensorCount int            `json:"tensor_count"`
}

// SnapshotState holds in-progress snapshot coordination state on the primary.
type SnapshotState struct {
	mu         sync.Mutex
	id         string
	localSnap  *GlobalSnapshot
	pendingAck map[string]bool // peerID -> acked?
	done       chan struct{}
}

// --- RPC handler: a backup calls this on the primary to send its local state
// back. In our primary-backup model the backup's "state" is just an ACK that
// it received the marker and recorded itself.
func (s *server) ReceiveMarkerAck(peerID string) {
	s.mu.Lock()
	snap := s.snapshot
	s.mu.Unlock()

	if snap == nil {
		return
	}
	snap.mu.Lock()
	defer snap.mu.Unlock()

	snap.pendingAck[peerID] = true

	// check if all peers have acked
	allDone := true
	for _, acked := range snap.pendingAck {
		if !acked {
			allDone = false
			break
		}
	}
	if allDone {
		select {
		case snap.done <- struct{}{}:
		default:
		}
	}
}

// SendMarker is called BY the primary ON each backup to deliver the marker
// message. The backup records its local state and ACKs.
func (s *server) SendMarker(ctx context.Context, req *pb.SnapshotRequest) (*pb.SnapshotResponse, error) {
	// Record backup local state
	s.mu.Lock()
	snap := GlobalSnapshot{
		SnapshotID:  req.SnapshotId,
		TakenAt:     time.Now(),
		ServerID:    s.id,
		CommitIndex: s.commitIndex,
	}

	// capture open client slots on this backup
	for fd, meta := range s.files {
		meta.mu.Lock()
		for clientID, cs := range meta.clients {
			e, ok := s.table[meta.filename]
			var ver int32
			if ok {
				e.mu.Lock()
				ver = e.version
				e.mu.Unlock()
			}
			snap.OpenClients = append(snap.OpenClients, SnapshotClientInfo{
				ClientID: clientID,
				Filename: meta.filename,
				FD:       fd,
				Mode:     cs.mode,
				Version:  ver,
			})
		}
		meta.mu.Unlock()
	}

	// capture file versions
	for name, entry := range s.table {
		entry.mu.Lock()
		ver := entry.version
		entry.mu.Unlock()
		snap.Files = append(snap.Files, SnapshotFileInfo{
			Filename:  name,
			Version:   ver,
			CommitIdx: s.commitIndex,
		})
	}
	s.mu.Unlock()

	// persist backup snapshot to disk
	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return nil, status.Errorf(codes.Internal, "marshal snapshot failed: %v", err)
	}
	path := fmt.Sprintf("snapshot_%s_%s.json", s.id, req.SnapshotId)
	if err := os.WriteFile(path, data, 0644); err != nil {
		return nil, status.Errorf(codes.Internal, "write snapshot failed: %v", err)
	}
	log.Printf("Backup %s: recorded snapshot %s", s.id, req.SnapshotId)
	return &pb.SnapshotResponse{Success: true}, nil
}

// InitiateSnapshot is called to kick off a full Chandy-Lamport snapshot.
// The caller can be an admin tool, a test, or the primary itself on a timer.
// It blocks until the snapshot is complete or times out.
func (s *server) InitiateSnapshot(ctx context.Context, req *pb.SnapshotRequest) (*pb.SnapshotResponse, error) {
	if s.role != Primary {
		return nil, status.Errorf(codes.FailedPrecondition, "only the primary can initiate snapshots")
	}

	snapID := req.SnapshotId
	log.Printf("Primary %s: initiating snapshot %s", s.id, snapID)

	// ── Step 1: Record primary local state atomically ──────────────────────
	s.mu.Lock()

	snap := &GlobalSnapshot{
		SnapshotID:  snapID,
		TakenAt:     time.Now(),
		ServerID:    s.id,
		CommitIndex: s.commitIndex,
	}

	// capture every open client slot
	for fd, meta := range s.files {
		meta.mu.Lock()
		for clientID, cs := range meta.clients {
			e, ok := s.table[meta.filename]
			var ver int32
			if ok {
				e.mu.Lock()
				ver = e.version
				e.mu.Unlock()
			}
			snap.OpenClients = append(snap.OpenClients, SnapshotClientInfo{
				ClientID: clientID,
				Filename: meta.filename,
				FD:       fd,
				Mode:     cs.mode,
				Version:  ver,
			})
		}
		meta.mu.Unlock()
	}

	// capture file version + prime set for each output file
	for name, entry := range s.table {
		entry.mu.Lock()
		ver := entry.version
		entry.mu.Unlock()

		fi := SnapshotFileInfo{
			Filename:  name,
			Version:   ver,
			CommitIdx: s.commitIndex,
		}
		if psPtr, ok := s.filesPrimes[name]; ok && psPtr != nil {
			for p := range *psPtr {
				fi.PrimeSet = append(fi.PrimeSet, p)
			}
		}
		snap.Files = append(snap.Files, fi)
	}
	// build pending-ack set for all peers
	snapState := &SnapshotState{
		id:         snapID,
		localSnap:  snap,
		pendingAck: make(map[string]bool),
		done:       make(chan struct{}, 1),
	}
	for peerID, srv := range s.servers {
		if peerID == s.id || !srv.Alive {
			continue
		}
		snapState.pendingAck[peerID] = false
	}
	s.snapshot = snapState
	peers := make([]ServerInfo, 0, len(s.servers))
	for peerID, srv := range s.servers {
		if peerID != s.id && srv.Alive {
			peers = append(peers, srv)
		}
	}
	s.mu.Unlock()

	// ── Capture ML training state from disk ────────────────────────────────
	// WorkerSteps and TensorCount are derived from files on disk rather than
	// in-memory server fields: every checkpoint commit (via ServerHandle in
	// server_handle.go) writes a "latest_<workerID>.ptr" file containing the
	// last committed step, and tensor blobs are stored as
	// "tensors_<2charprefix>_<restofhash>" files. Scanning these at snapshot
	// time gives an accurate point-in-time view of training progress without
	// requiring any additional server-side state to be kept in sync.
	snap.WorkerSteps = make(map[string]int)
	outputDir := filepath.Join(s.rootDir, "output")
	if entries, err := os.ReadDir(outputDir); err == nil {
		for _, e := range entries {
			name := e.Name()
			switch {
			case strings.HasPrefix(name, "latest_") && strings.HasSuffix(name, ".ptr"):
				workerID := strings.TrimSuffix(strings.TrimPrefix(name, "latest_"), ".ptr")
				data, err := os.ReadFile(filepath.Join(outputDir, name))
				if err != nil {
					continue
				}
				var step int
				if _, err := fmt.Sscanf(string(data), "%d", &step); err == nil {
					snap.WorkerSteps[workerID] = step
				}
			case strings.HasPrefix(name, "tensors_"):
				snap.TensorCount++
			}
		}
	}

	// persist primary snapshot to disk immediately (C-L: record before sending markers)
	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return nil, status.Errorf(codes.Internal, "marshal snapshot failed: %v", err)
	}
	snapPath := filepath.Join(".", fmt.Sprintf("snapshot_%s_%s.json", s.id, snapID))
	if err := os.WriteFile(snapPath, data, 0644); err != nil {
		return nil, status.Errorf(codes.Internal, "write snapshot failed: %v", err)
	}
	log.Printf("Primary %s: local snapshot written to %s", s.id, snapPath)

	// ── Step 2: Send MARKER to all alive backups in parallel ───────────────
	for _, peer := range peers {
		go func(p ServerInfo) {
			conn, err := grpc.NewClient(p.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				log.Printf("Snapshot: cannot reach backup %s: %v", p.ID, err)
				// treat unreachable peer as acked so we don't hang forever
				s.ReceiveMarkerAck(p.ID)
				return
			}
			defer conn.Close()
			client := pb.NewSnapshotServiceClient(conn)
			mCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			resp, err := client.SendMarker(mCtx, &pb.SnapshotRequest{SnapshotId: snapID})
			if err != nil || !resp.Success {
				log.Printf("Snapshot: backup %s marker failed: %v", p.ID, err)
			}
			// ACK regardless — if backup is down we still want to finish
			s.ReceiveMarkerAck(p.ID)
		}(peer)
	}

	// ── Step 3: Wait for all ACKs or timeout ───────────────────────────────
	if len(snapState.pendingAck) > 0 {
		select {
		case <-snapState.done:
			log.Printf("Primary %s: snapshot %s complete (all backups acked)", s.id, snapID)
		case <-time.After(10 * time.Second):
			log.Printf("Primary %s: snapshot %s timed out waiting for backups", s.id, snapID)
		}
	}

	// clear snapshot state
	s.mu.Lock()
	s.snapshot = nil
	s.mu.Unlock()

	return &pb.SnapshotResponse{Success: true}, nil
}

// LoadSnapshot reads a previously saved snapshot from disk.
// Use this for post-mortem analysis of client failures.
func LoadSnapshot(path string) (*GlobalSnapshot, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var snap GlobalSnapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return nil, err
	}
	return &snap, nil
}

// CheckClientInSnapshot returns true if the given clientID had an open dirty
// file descriptor captured in the snapshot.  This tells you: "did this client
// crash before committing its last write?"
func CheckClientInSnapshot(snap *GlobalSnapshot, clientID string) (dirty bool, details []SnapshotClientInfo) {
	for _, oc := range snap.OpenClients {
		if oc.ClientID == clientID {
			// Mode WRITE (1) means the client had the file open for writing.
			// If the snapshot was taken while the client was still processing,
			// Mode==WRITE means there may be uncommitted local data.
			if oc.Mode == pb.FileMode_WRITE {
				dirty = true
			}
			details = append(details, oc)
		}
	}
	return
}
