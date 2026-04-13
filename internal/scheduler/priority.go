package scheduler

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

// PriorityScheduler implements priority-based continuous batching.
// The waiting queue is sorted by Priority (descending), with FCFS tiebreak (by ID ascending).
// When AutoPreempt is enabled, a high-priority arrival can preempt the lowest-priority
// active sequence to make room in the batch.
type PriorityScheduler struct {
	cfg SchedulerConfig

	mu      sync.Mutex
	waiting []*Sequence
	active  map[uint64]*Sequence
	all     map[uint64]*Sequence
}

// NewPriorityScheduler creates a priority-based scheduler.
func NewPriorityScheduler(cfg SchedulerConfig) *PriorityScheduler {
	return &PriorityScheduler{
		cfg:    cfg,
		active: make(map[uint64]*Sequence),
		all:    make(map[uint64]*Sequence),
	}
}

func (s *PriorityScheduler) Add(seq *Sequence) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cfg.MaxQueueDepth > 0 && len(s.waiting) >= s.cfg.MaxQueueDepth {
		return fmt.Errorf("%w: depth %d", ErrQueueFull, len(s.waiting))
	}

	// Auto-preempt: if batch is full and new sequence has higher priority than
	// the lowest-priority active sequence, preempt that active sequence.
	if s.cfg.AutoPreempt && len(s.active) >= s.cfg.MaxBatchSize {
		var lowestSeq *Sequence
		for _, act := range s.active {
			if lowestSeq == nil || act.Priority < lowestSeq.Priority ||
				(act.Priority == lowestSeq.Priority && act.ID > lowestSeq.ID) {
				lowestSeq = act
			}
		}
		if lowestSeq != nil && seq.Priority > lowestSeq.Priority {
			// Preempt the lowest-priority active sequence.
			lowestSeq.Transition(SeqPreempted)
			applyPreemption(lowestSeq, s.cfg.PreemptMode)
			lowestSeq.Transition(SeqWaiting)
			delete(s.active, lowestSeq.ID)
			s.waiting = append(s.waiting, lowestSeq)
		}
	}

	s.waiting = append(s.waiting, seq)
	s.all[seq.ID] = seq
	return nil
}

func (s *PriorityScheduler) sortWaiting() {
	sort.SliceStable(s.waiting, func(i, j int) bool {
		if s.waiting[i].Priority != s.waiting[j].Priority {
			return s.waiting[i].Priority > s.waiting[j].Priority
		}
		return s.waiting[i].ID < s.waiting[j].ID
	})
}

func (s *PriorityScheduler) Tick(_ context.Context) (*SchedulerOutput, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Increment age for starvation prevention.
	for _, seq := range s.waiting {
		seq.Age++
	}

	s.sortWaiting()

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

	// 2. Resume partially-prefilled.
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

	// 3. Admit from priority-sorted waiting queue.
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

func (s *PriorityScheduler) Complete(seqID uint64) error {
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

func (s *PriorityScheduler) Preempt(seqID uint64) error {
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
	s.waiting = append(s.waiting, seq)
	return nil
}

func (s *PriorityScheduler) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.all)
}

// WaitingLen returns the number of sequences in the waiting queue.
func (s *PriorityScheduler) WaitingLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.waiting)
}

// ActiveLen returns the number of sequences in the active set.
func (s *PriorityScheduler) ActiveLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.active)
}

// Find returns a sequence by ID, or nil if not tracked.
func (s *PriorityScheduler) Find(seqID uint64) *Sequence {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.all[seqID]
}

// Compile-time check.
var _ Scheduler = (*PriorityScheduler)(nil)
