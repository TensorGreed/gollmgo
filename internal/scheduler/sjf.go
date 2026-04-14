package scheduler

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

// SJFScheduler implements shortest-job-first continuous batching.
// The waiting queue is sorted by estimated job size (PromptLen + MaxTokens, ascending).
// Sequences that exceed MaxWaitTicks are promoted to the front for starvation prevention.
type SJFScheduler struct {
	cfg SchedulerConfig

	mu      sync.Mutex
	waiting []*Sequence
	active  map[uint64]*Sequence
	all     map[uint64]*Sequence
}

// NewSJFScheduler creates a shortest-job-first scheduler.
func NewSJFScheduler(cfg SchedulerConfig) *SJFScheduler {
	if cfg.MaxWaitTicks <= 0 {
		cfg.MaxWaitTicks = 100
	}
	return &SJFScheduler{
		cfg:    cfg,
		active: make(map[uint64]*Sequence),
		all:    make(map[uint64]*Sequence),
	}
}

func (s *SJFScheduler) Add(seq *Sequence) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cfg.MaxQueueDepth > 0 && len(s.waiting) >= s.cfg.MaxQueueDepth {
		return fmt.Errorf("%w: depth %d", ErrQueueFull, len(s.waiting))
	}

	s.waiting = append(s.waiting, seq)
	s.all[seq.ID] = seq
	return nil
}

// jobSize returns the estimated total work for a sequence.
func jobSize(seq *Sequence) int {
	return seq.PromptLen + seq.MaxTokens
}

func (s *SJFScheduler) Tick(_ context.Context) (*SchedulerOutput, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Increment age for all waiting sequences and sort.
	for _, seq := range s.waiting {
		seq.Age++
	}
	sort.SliceStable(s.waiting, func(i, j int) bool {
		// Promoted (aged-out) sequences go first.
		iProm := s.cfg.MaxWaitTicks > 0 && s.waiting[i].Age > s.cfg.MaxWaitTicks
		jProm := s.cfg.MaxWaitTicks > 0 && s.waiting[j].Age > s.cfg.MaxWaitTicks
		if iProm != jProm {
			return iProm
		}
		return jobSize(s.waiting[i]) < jobSize(s.waiting[j])
	})

	out := &SchedulerOutput{}
	batchSize := 0
	tokenBudget := s.cfg.MaxTokenBudget
	chunkSize := s.cfg.PrefillChunkSize
	if chunkSize <= 0 {
		chunkSize = 1 << 30
	}

	// 1. Active decode sequences.
	decodeIDs := make([]uint64, 0, len(s.active))
	prefillIDs := make([]uint64, 0)
	for id, seq := range s.active {
		if seq.State == SeqDecoding {
			decodeIDs = append(decodeIDs, id)
		} else if seq.State == SeqPrefilling {
			prefillIDs = append(prefillIDs, id)
		}
	}
	sort.Slice(decodeIDs, func(i, j int) bool { return decodeIDs[i] < decodeIDs[j] })
	sort.Slice(prefillIDs, func(i, j int) bool { return prefillIDs[i] < prefillIDs[j] })

	for _, id := range decodeIDs {
		if batchSize >= s.cfg.MaxBatchSize {
			break
		}
		out.ScheduledSequences = append(out.ScheduledSequences, s.active[id])
		out.DecodeBudgetUsed++
		tokenBudget--
		batchSize++
	}

	// 2. Resume partially-prefilled sequences.
	for _, id := range prefillIDs {
		if batchSize >= s.cfg.MaxBatchSize || tokenBudget <= 0 {
			break
		}
		seq := s.active[id]
		remaining := seq.PromptLen - seq.PrefillConsumed
		chunk := remaining
		if chunk > chunkSize {
			chunk = chunkSize
		}
		if chunk > tokenBudget {
			chunk = tokenBudget
		}
		if chunk <= 0 {
			continue
		}
		out.ScheduledSequences = append(out.ScheduledSequences, seq)
		out.PrefillBudgetUsed += chunk
		tokenBudget -= chunk
		batchSize++
	}

	// 3. Admit from sorted waiting queue.
	remaining := s.waiting[:0]
	for _, seq := range s.waiting {
		if batchSize >= s.cfg.MaxBatchSize || tokenBudget <= 0 {
			remaining = append(remaining, seq)
			continue
		}

		firstChunk := seq.PromptLen
		if firstChunk > chunkSize {
			firstChunk = chunkSize
		}
		if firstChunk > tokenBudget {
			remaining = append(remaining, seq)
			continue
		}

		if err := seq.Transition(SeqPrefilling); err != nil {
			remaining = append(remaining, seq)
			continue
		}

		restoreSwapState(seq)
		seq.Age = 0
		s.active[seq.ID] = seq
		out.ScheduledSequences = append(out.ScheduledSequences, seq)
		out.PrefillBudgetUsed += firstChunk
		tokenBudget -= firstChunk
		batchSize++
	}
	s.waiting = remaining

	return out, nil
}

func (s *SJFScheduler) Complete(seqID uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	seq, ok := s.all[seqID]
	if !ok {
		return fmt.Errorf("%w: %d", ErrSeqNotFound, seqID)
	}
	seq.Transition(SeqFinished)
	delete(s.active, seqID)
	delete(s.all, seqID)
	return nil
}

func (s *SJFScheduler) Preempt(seqID uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	seq, ok := s.active[seqID]
	if !ok {
		return fmt.Errorf("%w: %d", ErrSeqNotFound, seqID)
	}

	if err := seq.Transition(SeqPreempted); err != nil {
		return err
	}
	applyPreemption(seq, s.cfg.PreemptMode)
	if err := seq.Transition(SeqWaiting); err != nil {
		return err
	}

	delete(s.active, seqID)
	// Re-enqueue; sort order will be restored on next Tick.
	s.waiting = append(s.waiting, seq)
	return nil
}

func (s *SJFScheduler) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.all)
}

// WaitingLen returns the number of sequences in the waiting queue.
func (s *SJFScheduler) WaitingLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.waiting)
}

// ActiveLen returns the number of sequences in the active set.
func (s *SJFScheduler) ActiveLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.active)
}

// Find returns a sequence by ID, or nil if not tracked.
func (s *SJFScheduler) Find(seqID uint64) *Sequence {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.all[seqID]
}

// PrefillChunkSize returns the configured chunk size for prefill budgeting.
func (s *SJFScheduler) PrefillChunkSize() int { return s.cfg.PrefillChunkSize }

// PreemptMode returns the configured preemption mode.
func (s *SJFScheduler) PreemptMode() PreemptMode { return s.cfg.PreemptMode }

// Compile-time check.
var _ Scheduler = (*SJFScheduler)(nil)
