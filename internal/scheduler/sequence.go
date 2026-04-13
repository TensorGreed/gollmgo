package scheduler

import (
	"errors"
	"fmt"
	"sync/atomic"
	"time"
)

// SeqState represents the lifecycle state of a sequence.
type SeqState int

const (
	SeqWaiting    SeqState = iota // queued, not yet scheduled
	SeqPrefilling                 // prompt tokens being processed
	SeqDecoding                   // generating tokens
	SeqFinished                   // generation complete (EOS or max tokens)
	SeqPreempted                  // evicted, eligible for re-admission
)

func (s SeqState) String() string {
	switch s {
	case SeqWaiting:
		return "WAITING"
	case SeqPrefilling:
		return "PREFILLING"
	case SeqDecoding:
		return "DECODING"
	case SeqFinished:
		return "FINISHED"
	case SeqPreempted:
		return "PREEMPTED"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", int(s))
	}
}

// Valid state transitions.
var validTransitions = map[SeqState][]SeqState{
	SeqWaiting:    {SeqPrefilling},
	SeqPrefilling: {SeqPrefilling, SeqDecoding}, // self-transition for chunked prefill
	SeqDecoding:   {SeqFinished, SeqPreempted},
	SeqPreempted:  {SeqWaiting},
	// SeqFinished is terminal.
}

var (
	ErrInvalidTransition = errors.New("scheduler: invalid state transition")
	ErrSequenceFinished  = errors.New("scheduler: sequence already finished")
)

// seqIDCounter is a global monotonic counter for sequence IDs.
var seqIDCounter atomic.Uint64

// NextSeqID returns the next unique sequence ID.
func NextSeqID() uint64 {
	return seqIDCounter.Add(1)
}

// SwapState holds saved state for PreemptSwap mode so a sequence can resume
// without full recomputation.
type SwapState struct {
	SavedPrefillConsumed int
	SavedGeneratedLen    int
}

// Sequence tracks the state of one inference request through the scheduler.
type Sequence struct {
	ID           uint64
	RequestID    string
	State        SeqState
	PromptLen    int
	GeneratedLen int
	MaxTokens    int
	TokenIDs     []int32 // prompt + generated so far

	// Chunked prefill tracking.
	PrefillConsumed int // how many prompt tokens have been processed so far

	// Priority for priority-based scheduling (higher = more important, default 0).
	Priority int

	// Age tracks how many ticks this sequence has spent in the waiting queue.
	// Used for starvation prevention in SJF and other policies.
	Age int

	// SwapState holds saved prefill/generation progress for PreemptSwap mode.
	SwapState *SwapState

	// Timestamps for latency metrics.
	CreatedAt     time.Time // when the sequence was created (for TTFT)
	LastTokenAt   time.Time // when the last token was emitted (for ITL)
}

// NewSequence creates a sequence in the Waiting state.
func NewSequence(requestID string, promptTokens []int32, maxTokens int) *Sequence {
	tokens := make([]int32, len(promptTokens))
	copy(tokens, promptTokens)
	return &Sequence{
		ID:        NextSeqID(),
		RequestID: requestID,
		State:     SeqWaiting,
		PromptLen: len(promptTokens),
		MaxTokens: maxTokens,
		TokenIDs:  tokens,
		CreatedAt: time.Now(),
	}
}

// Transition moves the sequence to a new state if the transition is valid.
func (s *Sequence) Transition(to SeqState) error {
	if s.State == SeqFinished {
		return ErrSequenceFinished
	}
	for _, valid := range validTransitions[s.State] {
		if valid == to {
			s.State = to
			return nil
		}
	}
	return fmt.Errorf("%w: %s -> %s", ErrInvalidTransition, s.State, to)
}

// AppendToken adds a generated token and increments the counter.
func (s *Sequence) AppendToken(tokenID int32) {
	s.TokenIDs = append(s.TokenIDs, tokenID)
	s.GeneratedLen++
}

// IsFinished returns true if the sequence is in a terminal state.
func (s *Sequence) IsFinished() bool {
	return s.State == SeqFinished
}

// TotalLen returns prompt + generated length.
func (s *Sequence) TotalLen() int {
	return s.PromptLen + s.GeneratedLen
}
