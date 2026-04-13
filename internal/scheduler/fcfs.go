package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// FCFSConfig configures the FCFS scheduler.
type FCFSConfig struct {
	MaxBatchSize   int // max sequences per batch
	MaxTokenBudget int // max total tokens (prefill + decode) per tick
	MaxQueueDepth  int // max waiting queue length (0 = unbounded)
}

// DefaultFCFSConfig returns reasonable defaults.
func DefaultFCFSConfig() FCFSConfig {
	return FCFSConfig{
		MaxBatchSize:   64,
		MaxTokenBudget: 4096,
		MaxQueueDepth:  256,
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

	// 1. Collect active decode sequences first (they each cost 1 token).
	for _, seq := range s.active {
		if seq.State == SeqDecoding {
			if batchSize >= s.cfg.MaxBatchSize {
				break
			}
			out.ScheduledSequences = append(out.ScheduledSequences, seq)
			out.DecodeBudgetUsed++
			tokenBudget--
			batchSize++
		}
	}

	// 2. Admit new sequences from waiting queue (FCFS order).
	remaining := s.waiting[:0]
	for _, seq := range s.waiting {
		if batchSize >= s.cfg.MaxBatchSize || tokenBudget <= 0 {
			remaining = append(remaining, seq)
			continue
		}

		prefillCost := seq.PromptLen
		if prefillCost > tokenBudget {
			// This sequence's prefill doesn't fit in remaining budget.
			// Keep it in the queue for next tick.
			remaining = append(remaining, seq)
			continue
		}

		// Admit: transition to PREFILLING.
		if err := seq.Transition(SeqPrefilling); err != nil {
			remaining = append(remaining, seq)
			continue
		}

		s.active[seq.ID] = seq
		out.ScheduledSequences = append(out.ScheduledSequences, seq)
		out.PrefillBudgetUsed += prefillCost
		tokenBudget -= prefillCost
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

// Compile-time check.
var _ Scheduler = (*FCFSScheduler)(nil)
