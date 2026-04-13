package scheduler

// PreemptMode controls what happens to a sequence when it is preempted.
type PreemptMode int

const (
	// PreemptRecompute resets PrefillConsumed to 0, requiring full recomputation
	// when the sequence is re-admitted. This is the default/existing behavior.
	PreemptRecompute PreemptMode = iota

	// PreemptSwap saves the sequence's prefill and generation progress into
	// SwapState so it can resume without full recomputation when re-admitted.
	PreemptSwap
)

func (m PreemptMode) String() string {
	switch m {
	case PreemptRecompute:
		return "RECOMPUTE"
	case PreemptSwap:
		return "SWAP"
	default:
		return "UNKNOWN"
	}
}

// applyPreemption handles the state save/reset logic for a preempted sequence.
func applyPreemption(seq *Sequence, mode PreemptMode) {
	switch mode {
	case PreemptSwap:
		seq.SwapState = &SwapState{
			SavedPrefillConsumed: seq.PrefillConsumed,
			SavedGeneratedLen:    seq.GeneratedLen,
		}
	default: // PreemptRecompute
		seq.PrefillConsumed = 0
		seq.SwapState = nil
	}
}

// restoreSwapState restores a sequence's progress from SwapState if available.
// Called when a swapped sequence is re-admitted.
func restoreSwapState(seq *Sequence) {
	if seq.SwapState != nil {
		seq.PrefillConsumed = seq.SwapState.SavedPrefillConsumed
		seq.SwapState = nil
	}
}
