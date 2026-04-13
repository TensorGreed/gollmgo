// Package backend defines the runner interface for GPU execution backends.
package backend

import "context"

// Capabilities describes what a backend runner supports.
type Capabilities struct {
	FP16         bool
	BF16         bool
	FP8          bool
	INT8         bool
	PagedAttention bool
	CUDAGraphs   bool
}

// WarmupProfile describes the shapes to warm up.
type WarmupProfile struct {
	MaxBatchSize int
	MaxSeqLen    int
	BlockSize    int
}

// Batch is the flattened input to one backend step.
type Batch struct {
	// SequenceIDs identifies each sequence in the batch.
	SequenceIDs []uint64
	// TokenIDs is the flattened token input for this step.
	TokenIDs []int32
	// Positions gives the position of each token in its sequence.
	Positions []int32
	// SlotMapping maps each token to its KV cache slot.
	SlotMapping []int32
	// IsPrefill indicates whether each sequence is in prefill mode.
	IsPrefill []bool
}

// StepOutput holds the result of one backend step.
type StepOutput struct {
	// Logits per sequence (outer slice = sequence, inner = vocab logits).
	// For decode sequences, only the last-token logits are returned.
	Logits [][]float32
}

// Runner is the interface for GPU execution backends.
// CUDA is the initial implementation; ROCm will follow using the same contract.
type Runner interface {
	// Warmup pre-allocates and compiles for expected workload shapes.
	Warmup(ctx context.Context, profile WarmupProfile) error
	// Step executes one forward pass for the given batch.
	Step(ctx context.Context, batch *Batch) (*StepOutput, error)
	// Capabilities reports what this backend supports.
	Capabilities() Capabilities
	// Close releases all backend resources.
	Close() error
}
