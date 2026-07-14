// ml_worker/worker.go — A simulated distributed training worker.
//
// What this simulates:
//   - A real PyTorch DDP worker calls optimizer.step() in a loop.
//   - Every checkpoint_interval steps, it serializes model.state_dict()
//     to bytes and saves via our DFS.
//   - On crash (SIGKILL / os.Exit), the next startup reads the last
//     committed checkpoint meta and resumes from there.
//
// This file is pure Go and uses math/rand to simulate tensor updates,
// so it runs without Python, PyTorch, or CUDA.
//
// Run multiple instances to simulate distributed training:
//   go run ml_worker/worker.go -id=worker_0 -total=3
//   go run ml_worker/worker.go -id=worker_1 -total=3
//   go run ml_worker/worker.go -id=worker_2 -total=3

package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	ikkat "distributed-system-ikkat"
	ml "distributed-system-ikkat/ml_common"
)

const (
	CheckpointInterval = 100  // save checkpoint every N steps
	TotalSteps         = 1000 // total training steps
	LearningRate       = 0.01
)

// SimpleModel is a tiny 3-layer MLP represented as named float32 tensors.
// In real training this would be model.parameters() from PyTorch.
type SimpleModel struct {
	tensors map[string]*ml.TensorBlob
}

// NewSimpleModel creates a randomly initialized model.
func NewSimpleModel() *SimpleModel {
	m := &SimpleModel{tensors: make(map[string]*ml.TensorBlob)}
	// Layer 0: 128×256 weight + 256 bias
	m.tensors["layer0.weight"] = randomTensor("layer0.weight", []int{128, 256})
	m.tensors["layer0.bias"] = randomTensor("layer0.bias", []int{256})
	// Layer 1: 256×128 weight + 128 bias
	m.tensors["layer1.weight"] = randomTensor("layer1.weight", []int{256, 128})
	m.tensors["layer1.bias"] = randomTensor("layer1.bias", []int{128})
	// Layer 2 (output): 128×10 weight + 10 bias
	m.tensors["layer2.weight"] = randomTensor("layer2.weight", []int{128, 10})
	m.tensors["layer2.bias"] = randomTensor("layer2.bias", []int{10})
	// Embedding table — large and tends to stabilize early (good for dedup demo)
	m.tensors["embedding.weight"] = randomTensor("embedding.weight", []int{1000, 64})
	return m
}

func randomTensor(name string, shape []int) *ml.TensorBlob {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]byte, n*4) // float32 = 4 bytes each
	for i := 0; i < n; i++ {
		val := float32(rand.NormFloat64() * 0.02) // small random init
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(val))
	}
	return &ml.TensorBlob{Name: name, DType: ml.Float32, Shape: shape, Data: data}
}

// Step simulates one optimizer step on the model.
// In real training this is: loss.backward(); optimizer.step()
// We update only a random subset of tensors to simulate realistic convergence
// where early layers stabilize (and thus deduplicate) before later ones.
func (m *SimpleModel) Step(step int) {
	// After step 200, the embedding table stabilizes (no more updates).
	// After step 400, layer0 stabilizes. This gives dedup rates that improve
	// over training — exactly what you see with real models.
	updateProb := map[string]float64{
		"embedding.weight": 1.0 - math.Min(1.0, float64(step)/200.0),
		"layer0.weight":    1.0 - math.Min(1.0, float64(step)/400.0),
		"layer0.bias":      1.0 - math.Min(1.0, float64(step)/400.0),
		"layer1.weight":    1.0 - math.Min(1.0, float64(step)/600.0),
		"layer1.bias":      1.0 - math.Min(1.0, float64(step)/600.0),
		"layer2.weight":    1.0, // output layer always trains
		"layer2.bias":      1.0,
	}

	for name, prob := range updateProb {
		if rand.Float64() > prob {
			continue // this tensor didn't change this step → will dedup at checkpoint
		}
		t := m.tensors[name]
		n := len(t.Data) / 4
		for i := 0; i < n; i++ {
			val := math.Float32frombits(binary.LittleEndian.Uint32(t.Data[i*4:]))
			// gradient descent with tiny random gradient
			grad := float32(rand.NormFloat64() * 0.001)
			val -= LearningRate * grad
			binary.LittleEndian.PutUint32(t.Data[i*4:], math.Float32bits(val))
		}
	}
}

// SaveCheckpoint saves the model state at the current step to the DFS.
// Returns dedup stats for logging.
func (m *SimpleModel) SaveCheckpoint(ctx context.Context, s *ikkat.ServerHandle, workerID string, step int) (*ml.CheckpointMeta, error) {
	// 1. Collect all tensors as a flat list.
	tensors := make([]*ml.TensorBlob, 0, len(m.tensors))
	for _, t := range m.tensors {
		tensors = append(tensors, t)
	}

	// 2. Send tensors to server for deduplication + storage.
	req := &ml.SaveTensorsRequest{
		WorkerID: workerID,
		Step:     step,
		Tensors:  tensors,
	}
	saveResp, err := s.SaveTensors(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("SaveTensors failed: %w", err)
	}

	// 3. Build checkpoint meta from the returned hashes.
	meta := &ml.CheckpointMeta{
		WorkerID:     workerID,
		Step:         step,
		SavedAt:      time.Now(),
		TensorMap:    saveResp.HashByName,
		ShapeMap:     make(map[string][]int),
		DTypeMap:     make(map[string]ml.DType),
		NewTensors:   saveResp.NewCount,
		TotalTensors: len(tensors),
	}
	for _, t := range tensors {
		meta.ShapeMap[t.Name] = t.Shape
		meta.DTypeMap[t.Name] = t.DType
	}

	// 4. Commit the meta file — this is the atomic "checkpoint complete" marker,
	//    analogous to Close() in the existing system.
	if err := s.SaveCheckpointMeta(ctx, meta); err != nil {
		return nil, fmt.Errorf("SaveCheckpointMeta failed: %w", err)
	}

	return meta, nil
}

// RestoreCheckpoint loads the latest committed checkpoint for this worker.
// Called on startup after a crash — the worker doesn't know which step it
// was at, so it reads the latest committed meta from the server.
func (m *SimpleModel) RestoreCheckpoint(ctx context.Context, s *ikkat.ServerHandle, workerID string) (int, error) {
	// Load the latest committed meta.
	meta, err := s.LoadCheckpointMeta(ctx, workerID, -1)
	if err != nil {
		return 0, err // no checkpoint found — start from scratch
	}

	log.Printf("RestoreCheckpoint: found checkpoint at step %d (%s)",
		meta.Step, meta.DeduplicationStats())

	// Load all tensor blobs by their content hashes.
	tensors, err := s.LoadTensors(ctx, meta.TensorMap)
	if err != nil {
		return 0, fmt.Errorf("LoadTensors failed: %w", err)
	}

	// Restore model state.
	for _, t := range tensors {
		if existing, ok := m.tensors[t.Name]; ok {
			// Reattach shape/dtype from meta (LoadTensors returns raw bytes only).
			t.Shape = meta.ShapeMap[t.Name]
			t.DType = meta.DTypeMap[t.Name]
			existing.Data = t.Data
			existing.Shape = t.Shape
		}
	}

	log.Printf("RestoreCheckpoint: model restored to step %d (%d tensors)",
		meta.Step, len(tensors))
	return meta.Step, nil
}

func main() {
	workerID := flag.String("id", "worker_0", "unique worker ID")
	totalWorkers := flag.Int("total", 3, "total number of workers")
	servers := flag.String("servers", "localhost:5001,localhost:5002,localhost:5003", "comma-separated server addresses")
	crashAtStep := flag.Int("crash", 0, "simulate crash at this step (0 = no crash)")
	flag.Parse()

	log.Printf("Worker %s starting (total_workers=%d)", *workerID, *totalWorkers)

	// Connect to the DFS cluster.
	addrs := splitAddresses(*servers)
	client, conn, err := ikkat.DialClient(addrs)
	if err != nil {
		log.Fatalf("Failed to connect to DFS: %v", err)
	}
	defer conn.Close()

	handle := ikkat.NewServerHandle(client)
	ctx := context.Background()
	model := NewSimpleModel()

	// Try to restore from a previous checkpoint.
	resumeStep := 0
	restoredStep, err := model.RestoreCheckpoint(ctx, handle, *workerID)
	if err != nil {
		log.Printf("No existing checkpoint found, starting from step 0: %v", err)
	} else {
		resumeStep = restoredStep
		log.Printf("Resuming from step %d", resumeStep)
	}

	// Training loop.
	for step := resumeStep + 1; step <= TotalSteps; step++ {
		// Simulate training step.
		model.Step(step)

		// Simulate crash for testing.
		if *crashAtStep > 0 && step == *crashAtStep {
			log.Printf("Worker %s: SIMULATING CRASH at step %d", *workerID, step)
			os.Exit(1)
		}

		// Save checkpoint every CheckpointInterval steps.
		if step%CheckpointInterval == 0 {
			meta, err := model.SaveCheckpoint(ctx, handle, *workerID, step)
			if err != nil {
				log.Printf("Checkpoint failed at step %d: %v", step, err)
				continue
			}
			log.Printf("Worker %s: %s", *workerID, meta.DeduplicationStats())
		}
	}

	log.Printf("Worker %s: training complete (step %d)", *workerID, TotalSteps)
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
