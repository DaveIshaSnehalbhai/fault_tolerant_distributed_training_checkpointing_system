package ml_common

import (
	"encoding/json"
	"fmt"
	"time"
)

// CheckpointMeta is the small metadata file stored per checkpoint.
// It is stored as a JSON file on the DFS at path:
//
//	checkpoints/<workerID>/step_<step>.meta
//
// The actual tensor bytes are stored separately at:
//
//	tensors/<sha256hash>  (content-addressed, shared across all workers)
//
// This split is the key design: the meta file is tiny (just names + hashes),
// so listing all checkpoints for a worker is cheap. The tensor blobs are
// shared: if worker_0 and worker_1 have the same frozen embedding table,
// it is stored only once.
//
// ANALOGY: this is like Git's commit object (small, contains tree pointers)
// vs blob objects (actual file content, content-addressed by SHA).
type CheckpointMeta struct {
	WorkerID string    `json:"worker_id"`
	Step     int       `json:"step"`
	SavedAt  time.Time `json:"saved_at"`
	// TensorMap maps tensor name → SHA-256 hash of that tensor's bytes.
	// To restore the model: for each (name, hash), call LoadTensor(hash)
	// and reconstruct the tensor using the stored shape/dtype.
	TensorMap map[string]string `json:"tensor_map"` // name → hash
	// ShapeMap stores the shape for each tensor so the client can
	// pre-allocate correctly when loading.
	ShapeMap map[string][]int `json:"shape_map"`
	DTypeMap map[string]DType `json:"dtype_map"`
	// NewTensors counts how many tensors were actually new (not deduped)
	// in this checkpoint. Useful for measuring dedup efficiency.
	NewTensors   int `json:"new_tensors"`
	TotalTensors int `json:"total_tensors"`
}

// MetaPath returns the DFS path for this checkpoint's metadata file.
func MetaPath(workerID string, step int) string {
	return fmt.Sprintf("output/checkpoints_%s_step_%06d.meta", workerID, step)
}

// TensorPath returns the DFS path for a tensor blob by its content hash.
// All workers share the same tensor namespace — this is where dedup happens.
func TensorPath(hash string) string {
	// Use a two-level directory structure to avoid too many files in one dir.
	// First 2 chars of hash as prefix (like Git's .git/objects/ab/cdef...)
	return fmt.Sprintf("output/tensors_%s_%s", hash[:2], hash[2:])
}

// LatestMetaPath returns the path to the "pointer" file that tracks which
// step is the latest committed checkpoint for a worker.
// This is what the coordinator reads to find where a dead worker left off.
func LatestMetaPath(workerID string) string {
	return fmt.Sprintf("output/latest_%s.ptr", workerID)
}

// Marshal serializes CheckpointMeta to JSON bytes for storage.
func (m *CheckpointMeta) Marshal() ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// UnmarshalMeta deserializes JSON bytes into a CheckpointMeta.
func UnmarshalMeta(data []byte) (*CheckpointMeta, error) {
	var m CheckpointMeta
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// DeduplicationStats prints a summary of storage savings for a checkpoint.
func (m *CheckpointMeta) DeduplicationStats() string {
	if m.TotalTensors == 0 {
		return "no tensors"
	}
	deduped := m.TotalTensors - m.NewTensors
	pct := float64(deduped) / float64(m.TotalTensors) * 100
	return fmt.Sprintf(
		"step=%d: %d/%d tensors deduped (%.1f%% storage saved)",
		m.Step, deduped, m.TotalTensors, pct,
	)
}
