// Package backend defines the runner interface for GPU execution backends.
package backend

import "context"

// Capabilities describes what a backend runner supports.
type Capabilities struct {
	FP16           bool
	BF16           bool
	FP8            bool
	INT8           bool
	PagedAttention bool
	CUDAGraphs    bool
	// SpeculativeDecoding is true when the runner can evaluate draft tokens
	// in a single step (multi-position logits per sequence).
	SpeculativeDecoding bool
	// KVSwap is true when the runner exposes SnapshotKV/RestoreKV for
	// engine-level preemption with swap semantics.
	KVSwap bool
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
	// SeqTokenCounts[i] = number of tokens in this batch for sequence i.
	SeqTokenCounts []int32
	// SeqContextLens[i] = total cached KV length for sequence i (including this batch).
	SeqContextLens []int32
	// SeqSlotTables[i] = full slot table for sequence i (len = SeqContextLens[i]).
	SeqSlotTables [][]int32
	// DraftTokens[i] is an optional list of speculative draft tokens appended
	// AFTER the sampled current token for sequence i. When non-empty, the
	// backend evaluates K+1 positions for that sequence (current token +
	// drafts) and returns per-position logits in StepOutput.LogitsPerPosition.
	// Applies only to decode steps; ignored for prefill.
	DraftTokens [][]int32
}

// StepOutput holds the result of one backend step.
type StepOutput struct {
	// Logits per sequence (outer slice = sequence, inner = vocab logits).
	// For decode sequences, only the last-token logits are returned.
	// When DraftTokens is non-empty for a sequence, the last-token logits
	// here correspond to the CURRENT token; see LogitsPerPosition for the
	// draft positions.
	Logits [][]float32
	// LogitsPerPosition[i] contains per-position logits for sequence i when
	// speculative verification is requested. Length is 1 + len(DraftTokens[i]):
	// index 0 is the current token, indices 1..K are the draft positions.
	// nil or empty when no drafts were requested for that sequence.
	LogitsPerPosition [][][]float32
}

// KVSnapshot is an opaque handle to a preempted sequence's KV state held
// in host memory. The engine passes it back to RestoreKV to resume.
type KVSnapshot interface {
	// NumBlocks returns the number of logical blocks captured.
	NumBlocks() int
	// BytesOnHost returns the size of the host-side allocation (0 for mock).
	BytesOnHost() int64
	// Release frees the host-side allocation. Must be called exactly once.
	Release() error
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

// KVSwapper is an optional capability: runners that implement it can
// snapshot and restore the KV cache contents for a set of blocks so a
// preempted sequence can resume without recomputing its prefill.
type KVSwapper interface {
	// SnapshotKV copies the KV contents of the given block IDs off the
	// device into host memory. The returned snapshot handle owns the host
	// allocation until Release is called.
	SnapshotKV(ctx context.Context, blockIDs []int32) (KVSnapshot, error)
	// RestoreKV copies a snapshot's contents back into the given (freshly
	// allocated) block IDs. The snapshot's block count must match len(blockIDs).
	// The snapshot is NOT released; the caller must call snap.Release() when done.
	RestoreKV(ctx context.Context, snap KVSnapshot, blockIDs []int32) error
}
