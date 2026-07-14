// Package ml_common defines the shared types used by the ML checkpoint layer
//
// ANALOGY TO EXISTING SYSTEM:
//
//	existing: prime number (uint64)  ──► unit of deduplication
//	here:     TensorBlob (name+bytes) ──► unit of deduplication
//
//	existing: IsPrime(n) bool         ──► membership test
//	here:     HashTensor(t) string    ──► content fingerprint
//
//	existing: filesPrimes map         ──► set of seen values
//	here:     tensorStore map         ──► set of seen blobs
package ml_common

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// DType represents the data type of a tensor's elements.
type DType string

const (
	Float32 DType = "float32"
	Float64 DType = "float64"
	Int32   DType = "int32"
	Int64   DType = "int64"
)

// TensorBlob is a named tensor with its raw bytes.
// This is the unit that gets content-addressed and deduplicated.
//
// Why raw bytes and not a float slice?
// Because the DFS stores arbitrary []byte. The ML layer serializes tensors
// to bytes before handing them to the storage layer, exactly like the prime
// filter serializes numbers to "2\n3\n5\n" text.
type TensorBlob struct {
	Name  string // e.g. "layer0.weight", "optimizer.momentum"
	DType DType
	Shape []int  // e.g. [768, 3072] for a transformer FFN weight
	Data  []byte // raw little-endian bytes of the tensor elements
}

// HashTensor computes the SHA-256 content hash of a tensor blob.
// Two tensors with identical data but different names have DIFFERENT hashes
// (name is included) to prevent accidental cross-layer deduplication.
// Two tensors with identical name AND identical data have the SAME hash —
// this is the deduplication key, analogous to the prime number value itself.
func HashTensor(t *TensorBlob) string {
	h := sha256.New()
	h.Write([]byte(t.Name))
	h.Write([]byte(t.DType))
	for _, dim := range t.Shape {
		h.Write([]byte(fmt.Sprintf("%d,", dim)))
	}
	h.Write(t.Data)
	return hex.EncodeToString(h.Sum(nil))
}

// BytesForShape returns the number of bytes needed for a tensor of this shape
// and dtype. Used to pre-allocate receive buffers.
func BytesForShape(shape []int, dtype DType) int {
	total := 1
	for _, d := range shape {
		total *= d
	}
	switch dtype {
	case Float32, Int32:
		return total * 4
	case Float64, Int64:
		return total * 8
	default:
		return total * 4
	}
}

// SimulateTensor creates a fake tensor for testing/demo purposes.
// In real use, workers serialize their actual PyTorch/NumPy arrays here.
func SimulateTensor(name string, shape []int, seed float32) *TensorBlob {
	dtype := Float32
	n := BytesForShape(shape, dtype)
	data := make([]byte, n)
	// fill with deterministic pattern based on seed so same seed == same tensor
	// (simulates a "converged" layer that stops changing)
	for i := range data {
		val := byte(int(seed*100)%256) ^ byte(i%256)
		data[i] = val
	}
	return &TensorBlob{
		Name:  name,
		DType: dtype,
		Shape: shape,
		Data:  data,
	}
}
