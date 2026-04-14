package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
)

// FCFSConfig configures the FCFS scheduler.
type FCFSConfig struct {
	MaxBatchSize     int         // max sequences per batch
	MaxTokenBudget   int         // max total tokens (prefill + decode) per tick
	MaxQueueDepth    int         // max waiting queue length (0 = unbounded)
	PrefillChunkSize int         // max prefill tokens per sequence per tick (0 = no chunking)
	PreemptMode      PreemptMode // how preempted sequences are restored on re-admission
}

// DefaultFCFSConfig returns reasonable defaults.
func DefaultFCFSConfig() FCFSConfig {
	return FCFSConfig{
		MaxBatchSize:     64,
		MaxTokenBudget:   4096,
		MaxQueueDepth:    256,
		PrefillChunkSize: 512,
	}
}

var (
	ErrQueueFull     = errors.New("scheduler: queue is full")
	ErrSeqNotFound   = errors.New("scheduler: sequence not found")
)

// FCFSScheduler implements first-come-first-served continuous batching.
// It maintains a waiting queue and an active decode set.
// Each Tick:
//  1. Collects active decode sequences.
//  2. Admits new sequences from the queue if budget allows.
//  3. Returns a mixed batch of prefill + decode sequences.
type FCFSScheduler struct {
	cfg FCFSConfig

	mu       sync.Mutex
	waiting  []*Sequence          // FCFS queue
	active   map[uint64]*Sequence // sequences in PREFILLING or DECODING
	all      map[uint64]*Sequence // all tracked sequences
}

// NewFCFSScheduler creates a scheduler.
func NewFCFSScheduler(cfg FCFSConfig) *FCFSScheduler {
	return &FCFSScheduler{
		cfg:    cfg,
		active: make(map[uint64]*Sequence),
		all:    make(map[uint64]*Sequence),
	}
}

func (s *FCFSScheduler) Add(seq *Sequence) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cfg.MaxQueueDepth > 0 && len(s.waiting) >= s.cfg.MaxQueueDepth {
		return fmt.Errorf("%w: depth %d", ErrQueueFull, len(s.waiting))
	}

	s.waiting = append(s.waiting, seq)
	s.all[seq.ID] = seq
	return nil
}

func (s *FCFSScheduler) Tick(_ context.Context) (*SchedulerOutput, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := &SchedulerOutput{}
	batchSize := 0
	tokenBudget := s.cfg.MaxTokenBudget
	chunkSize := s.cfg.PrefillChunkSize
	if chunkSize <= 0 {
		chunkSize = 1<<30 // effectively unlimited
	}

	// 1. Collect active decode sequences in deterministic order (by seq ID).
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

	// 2. Resume partially-prefilled sequences (chunked prefill continuation).
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

	// 3. Admit new sequences from waiting queue (FCFS order).
	remaining := s.waiting[:0]
	for _, seq := range s.waiting {
		if batchSize >= s.cfg.MaxBatchSize || tokenBudget <= 0 {
			remaining = append(remaining, seq)
			continue
		}

		// With chunked prefill, the first chunk cost is min(promptLen, chunkSize).
		firstChunk := seq.PromptLen
		if firstChunk > chunkSize {
			firstChunk = chunkSize
		}
		if firstChunk > tokenBudget {
			remaining = append(remaining, seq)
			continue
		}

		// Admit: transition to PREFILLING.
		if err := seq.Transition(SeqPrefilling); err != nil {
			remaining = append(remaining, seq)
			continue
		}

		restoreSwapState(seq)
		s.active[seq.ID] = seq
		out.ScheduledSequences = append(out.ScheduledSequences, seq)
		out.PrefillBudgetUsed += firstChunk
		tokenBudget -= firstChunk
		batchSize++
	}
	s.waiting = remaining

	return out, nil
}

func (s *FCFSScheduler) Complete(seqID uint64) error {
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

func (s *FCFSScheduler) Preempt(seqID uint64) error {
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
	// Re-enqueue at the front for fairness.
	s.waiting = append([]*Sequence{seq}, s.waiting...)
	return nil
}

func (s *FCFSScheduler) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.all)
}

// WaitingLen returns the number of sequences in the waiting queue.
func (s *FCFSScheduler) WaitingLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.waiting)
}

// ActiveLen returns the number of sequences in the active set.
func (s *FCFSScheduler) ActiveLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.active)
}

// Find returns a sequence by ID, or nil if not tracked.
func (s *FCFSScheduler) Find(seqID uint64) *Sequence {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.all[seqID]
}

// PrefillChunkSize returns the configured chunk size for prefill budgeting.
func (s *FCFSScheduler) PrefillChunkSize() int {
	return s.cfg.PrefillChunkSize
}

// PreemptMode returns the configured preemption mode.
func (s *FCFSScheduler) PreemptMode() PreemptMode {
	return s.cfg.PreemptMode
}

// Compile-time check.
var _ Scheduler = (*FCFSScheduler)(nil)
