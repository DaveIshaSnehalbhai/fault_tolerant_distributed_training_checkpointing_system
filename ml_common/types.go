package ml_common

// SaveTensorsRequest is sent by a training worker (via ServerHandle.SaveTensors)
// to persist a batch of tensors with deduplication.
//
// This type used to live in server_ml.go alongside a native gRPC-style
// SaveTensors implementation on *server. That implementation was removed
// because it was never wired up as an actual RPC (no proto service) and
// duplicated what ServerHandle already does using the existing
// Create/Open/Write/Close/TestAuth RPCs. The type itself is still needed —
// it's the request shape used by ServerHandle.SaveTensors and ml_worker —
// so it now lives here in ml_common where both server_handle.go and
// ml_worker/worker.go can import it without a circular dependency.
type SaveTensorsRequest struct {
	WorkerID string
	Step     int
	Tensors  []*TensorBlob
}

// SaveTensorsResponse tells the worker which content hash corresponds to
// each tensor name (for building CheckpointMeta.TensorMap), plus dedup stats.
type SaveTensorsResponse struct {
	// HashByName maps tensor name → SHA-256 content hash.
	HashByName map[string]string
	// NewCount = how many tensors were actually written (not deduped).
	NewCount int
	// DedupeCount = how many tensors already existed on the server.
	DedupeCount int
	// NewBytes = total bytes actually written to disk this checkpoint
	// (sum of len(tensor.Data) for tensors that were NOT deduped).
	NewBytes int64
	// DedupedBytes = total bytes that did NOT need to be written because
	// an identical tensor already existed on the server (the storage saved
	// by content-addressed deduplication for this checkpoint).
	DedupedBytes int64
}
