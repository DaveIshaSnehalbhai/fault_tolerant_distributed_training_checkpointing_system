// server_handle.go — a thin wrapper around the existing *client that exposes
// the ML-specific operations (SaveTensors, LoadTensors, SaveCheckpointMeta,
// LoadCheckpointMeta) without leaking the internal server struct.
//
// Think of this as the ML equivalent of what DialClient() returns —
// it's the client-side facade for the ML checkpoint layer.
//
// Why not just call server methods directly?
// Because ml_worker and ml_coordinator are separate binaries. They talk to
// the server over the network (gRPC). ServerHandle is the client-side stub.
// In this implementation, it uses the existing file Open/Write/Close/Read
// primitives of the DFS to store tensors — so no new gRPC proto is needed.
//
// How it maps:
//   SaveTensors(req)         → for each new tensor: Create+Open+WriteFile+Close
//   LoadTensors(hashByName)  → for each hash: Open+Read
//   SaveCheckpointMeta(meta) → Create+Open+WriteFile+Close for the .meta file
//   LoadCheckpointMeta(...)  → Open+Read for the .meta file

package ikkat

import (
	"context"
	"fmt"

	pb "distributed-system-ikkat/filesystem"
	ml "distributed-system-ikkat/ml_common"
)

// ServerHandle is the client-side handle for ML checkpoint operations.
// Obtained by calling NewServerHandle(client) after DialClient().
type ServerHandle struct {
	c *client
}

// NewServerHandle wraps an existing client connection with ML operations.
func NewServerHandle(c *client) *ServerHandle {
	return &ServerHandle{c: c}
}

// GRPCClient exposes the underlying gRPC client for coordinator use.
func (h *ServerHandle) GRPCClient() pb.FileServiceClient {
	return h.c.grpcClient
}

// SaveTensors stores a batch of tensors on the DFS with deduplication.
// For each tensor, it hashes the content and checks if the file already
// exists on the server. Only new (non-duplicate) tensors are written.
func (h *ServerHandle) SaveTensors(ctx context.Context, req *ml.SaveTensorsRequest) (*ml.SaveTensorsResponse, error) {
	resp := &ml.SaveTensorsResponse{
		HashByName: make(map[string]string),
	}

	for _, tensor := range req.Tensors {
		hash := ml.HashTensor(tensor)
		resp.HashByName[tensor.Name] = hash

		tensorPath := ml.TensorPath(hash)
		workerID := req.WorkerID

		// Check if this tensor already exists (dedup check).
		// We use TestAuth — if the file exists and has version > 0, it's stored.
		authResp, err := h.c.testAuth(ctx, tensorPath)
		if err == nil && authResp.Version > 0 {
			// Already stored — this is the dedup win.
			resp.DedupeCount++
			resp.DedupedBytes += int64(len(tensor.Data))
			continue
		}

		// New tensor — create and write.
		if _, err := h.c.Create(ctx, tensorPath, workerID); err != nil {
			// AlreadyExists is fine — race between workers, tensor is stored.
			// Any other error is real.
			if isAlreadyExists(err) {
				resp.DedupeCount++
				resp.DedupedBytes += int64(len(tensor.Data))
				continue
			}
			return nil, fmt.Errorf("Create tensor %s: %w", tensor.Name, err)
		}
		if _, err := h.c.Open(ctx, tensorPath, pb.FileMode_WRITE, workerID); err != nil {
			return nil, fmt.Errorf("Open tensor %s: %w", tensor.Name, err)
		}
		if err := h.c.WriteFile(tensorPath, tensor.Data); err != nil {
			return nil, fmt.Errorf("Write tensor %s: %w", tensor.Name, err)
		}
		if err := h.c.Close(ctx, tensorPath, workerID); err != nil {
			return nil, fmt.Errorf("Close tensor %s: %w", tensor.Name, err)
		}
		resp.NewCount++
		resp.NewBytes += int64(len(tensor.Data))
	}

	return resp, nil
}

// SaveCheckpointMeta persists the checkpoint metadata file to the DFS.
// This is the final "commit" operation for a checkpoint — analogous to
// calling Close() on the output file in the prime filter system.
func (h *ServerHandle) SaveCheckpointMeta(ctx context.Context, meta *ml.CheckpointMeta) error {
	data, err := meta.Marshal()
	if err != nil {
		return fmt.Errorf("marshal meta: %w", err)
	}

	metaPath := ml.MetaPath(meta.WorkerID, meta.Step)
	workerID := meta.WorkerID

	if _, err := h.c.Create(ctx, metaPath, workerID); err != nil && !isAlreadyExists(err) {
		return fmt.Errorf("Create meta %s: %w", metaPath, err)
	}
	if _, err := h.c.Open(ctx, metaPath, pb.FileMode_WRITE, workerID); err != nil {
		return fmt.Errorf("Open meta: %w", err)
	}
	if err := h.c.WriteFile(metaPath, data); err != nil {
		return fmt.Errorf("Write meta: %w", err)
	}
	if err := h.c.Close(ctx, metaPath, workerID); err != nil {
		return fmt.Errorf("Close meta: %w", err)
	}

	// Update the "latest" pointer.
	latestPath := ml.LatestMetaPath(workerID)
	if _, err := h.c.Create(ctx, latestPath, workerID); err != nil && !isAlreadyExists(err) {
		return fmt.Errorf("Create latest ptr: %w", err)
	}
	if _, err := h.c.Open(ctx, latestPath, pb.FileMode_WRITE, workerID); err != nil {
		return fmt.Errorf("Open latest ptr: %w", err)
	}
	if err := h.c.WriteFile(latestPath, []byte(fmt.Sprintf("%d", meta.Step))); err != nil {
		return fmt.Errorf("Write latest ptr: %w", err)
	}
	if err := h.c.Close(ctx, latestPath, workerID); err != nil {
		return fmt.Errorf("Close latest ptr: %w", err)
	}

	return nil
}

// LoadCheckpointMeta reads checkpoint metadata for a worker.
// Pass step = -1 to load the latest committed checkpoint.
func (h *ServerHandle) LoadCheckpointMeta(ctx context.Context, workerID string, step int) (*ml.CheckpointMeta, error) {
	resolvedStep := step
	if step == -1 {
		// Read the latest pointer file.
		latestPath := ml.LatestMetaPath(workerID)
		if _, err := h.c.Open(ctx, latestPath, pb.FileMode_READ, workerID); err != nil {
			return nil, fmt.Errorf("no checkpoint for worker %s: %w", workerID, err)
		}
		data, err := h.c.Read(ctx, latestPath, workerID)
		if err != nil {
			h.c.Close(ctx, latestPath, workerID)
			return nil, fmt.Errorf("read latest pointer: %w", err)
		}
		h.c.Close(ctx, latestPath, workerID) // release read lease promptly
		fmt.Sscanf(string(data), "%d", &resolvedStep)
	}

	metaPath := ml.MetaPath(workerID, resolvedStep)
	if _, err := h.c.Open(ctx, metaPath, pb.FileMode_READ, workerID); err != nil {
		return nil, fmt.Errorf("open meta for step %d: %w", resolvedStep, err)
	}
	data, err := h.c.Read(ctx, metaPath, workerID)
	if err != nil {
		h.c.Close(ctx, metaPath, workerID)
		return nil, fmt.Errorf("read meta for step %d: %w", resolvedStep, err)
	}
	h.c.Close(ctx, metaPath, workerID)
	return ml.UnmarshalMeta(data)
}

// LoadTensors retrieves tensor blobs by their content hashes.
func (h *ServerHandle) LoadTensors(ctx context.Context, hashByName map[string]string) ([]*ml.TensorBlob, error) {
	var tensors []*ml.TensorBlob
	for name, hash := range hashByName {
		tensorPath := ml.TensorPath(hash)
		if _, err := h.c.Open(ctx, tensorPath, pb.FileMode_READ, "recovery"); err != nil {
			return nil, fmt.Errorf("open tensor %s (hash %s): %w", name, hash[:8], err)
		}
		data, err := h.c.Read(ctx, tensorPath, "recovery")
		if err != nil {
			h.c.Close(ctx, tensorPath, "recovery")
			return nil, fmt.Errorf("read tensor %s: %w", name, err)
		}
		h.c.Close(ctx, tensorPath, "recovery") // release read lease promptly
		tensors = append(tensors, &ml.TensorBlob{
			Name: name,
			Data: data,
		})
	}
	return tensors, nil
}

func isAlreadyExists(err error) bool {
	if err == nil {
		return false
	}
	return containsStr(err.Error(), "AlreadyExists") || containsStr(err.Error(), "already exists")
}

func containsStr(s, sub string) bool {
	return len(s) >= len(sub) && func() bool {
		for i := 0; i <= len(s)-len(sub); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	}()
}
