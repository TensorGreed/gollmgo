package engine

// NGramDrafter generates draft tokens using an n-gram model built from
// the sequence's token history. Used for speculative decoding to predict
// multiple tokens per forward pass.
type NGramDrafter struct {
	N         int // n-gram size (e.g. 3 = trigram)
	MaxDrafts int // max tokens to draft per step
}

// NewNGramDrafter creates a drafter with the given n-gram size.
func NewNGramDrafter(n, maxDrafts int) *NGramDrafter {
	if n < 2 {
		n = 2
	}
	if maxDrafts < 1 {
		maxDrafts = 4
	}
	return &NGramDrafter{N: n, MaxDrafts: maxDrafts}
}

// Draft generates up to MaxDrafts token predictions from the sequence history.
// Returns the draft token IDs. If no n-gram match is found, returns nil.
func (d *NGramDrafter) Draft(history []int32) []int32 {
	if len(history) < d.N {
		return nil
	}

	// Build n-gram table from history.
	// Key: last (N-1) tokens → Value: next token seen after that pattern.
	// Use the most recent occurrence for each pattern.
	type ngramKey [8]int32 // max N=8

	table := make(map[ngramKey]int32)
	keyLen := d.N - 1

	for i := 0; i <= len(history)-d.N; i++ {
		var key ngramKey
		copy(key[:keyLen], history[i:i+keyLen])
		table[key] = history[i+keyLen]
	}

	// Generate drafts by repeatedly looking up the suffix.
	drafts := make([]int32, 0, d.MaxDrafts)
	suffix := make([]int32, keyLen)
	copy(suffix, history[len(history)-keyLen:])

	for len(drafts) < d.MaxDrafts {
		var key ngramKey
		copy(key[:keyLen], suffix)
		next, ok := table[key]
		if !ok {
			break
		}
		drafts = append(drafts, next)
		// Shift suffix.
		copy(suffix, suffix[1:])
		suffix[keyLen-1] = next
	}

	return drafts
}

// Verify checks draft tokens against the model's logits and returns
// the number of accepted tokens. The model produced logits for each
// draft position; we accept as long as the model's argmax matches the draft.
// Returns (accepted count, token to use at the first rejected position).
func Verify(drafts []int32, logitsPerPosition [][]float32) (int, int32) {
	for i, draft := range drafts {
		if i >= len(logitsPerPosition) {
			break
		}
		modelChoice := argmax(logitsPerPosition[i])
		if modelChoice != draft {
			// Reject from here. Use the model's choice at position i.
			return i, modelChoice
		}
	}
	// All drafts accepted. The "next" token comes from the last logits.
	if len(logitsPerPosition) > len(drafts) {
		return len(drafts), argmax(logitsPerPosition[len(drafts)])
	}
	return len(drafts), -1
}
