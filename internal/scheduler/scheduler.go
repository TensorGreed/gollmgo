// Package scheduler defines the scheduling interface and sequence state model.
package scheduler

import "context"

// SchedulerOutput is the result of one scheduler tick.
type SchedulerOutput struct {
	// ScheduledSequences are the sequences selected for this step.
	ScheduledSequences []*Sequence
	// PreemptedSequenceIDs are active sequences evicted during this tick.
	// The engine uses this to release or preserve runtime state coherently.
	PreemptedSequenceIDs []uint64
	// PrefillBudgetUsed is the number of prefill tokens in this batch.
	PrefillBudgetUsed int
	// DecodeBudgetUsed is the number of decode tokens in this batch.
	DecodeBudgetUsed int
}

// Scheduler is the interface for request admission and batching.
type Scheduler interface {
	// Add submits a new sequence for scheduling.
	Add(seq *Sequence) error
	// Tick runs one scheduling iteration and returns the batch to execute.
	Tick(ctx context.Context) (*SchedulerOutput, error)
	// Complete marks a sequence as finished and releases resources.
	Complete(seqID uint64) error
	// Preempt forces a sequence back to waiting state. The scheduler's
	// configured PreemptMode determines whether progress is preserved.
	Preempt(seqID uint64) error
	// Len returns the total number of tracked sequences.
	Len() int
	// WaitingLen returns the number of sequences in the waiting queue.
	WaitingLen() int
	// ActiveLen returns the number of sequences in the active set.
	ActiveLen() int
	// Find returns a tracked sequence by ID, or nil if not tracked.
	Find(seqID uint64) *Sequence
	// PrefillChunkSize returns the configured prefill chunk size.
	// 0 means chunked prefill is disabled.
	PrefillChunkSize() int
	// PreemptMode returns the configured preemption mode.
	PreemptMode() PreemptMode
}
