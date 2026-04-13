package engine

import "testing"

func TestNGramDrafterBasic(t *testing.T) {
	drafter := NewNGramDrafter(3, 4)

	// History: "the cat sat on the mat"
	history := []int32{10, 20, 30, 40, 10, 50}

	drafts := drafter.Draft(history)
	// Last bigram is [10, 50]. In the trigram table:
	//   [10, 20] → 30
	//   [20, 30] → 40
	//   [30, 40] → 10
	//   [40, 10] → 50
	// Suffix is [10, 50] — no match in table, so no drafts.
	if len(drafts) != 0 {
		t.Fatalf("expected 0 drafts for unseen suffix, got %d: %v", len(drafts), drafts)
	}
}

func TestNGramDrafterRepeatingPattern(t *testing.T) {
	drafter := NewNGramDrafter(3, 4)

	// Repeating pattern: A B C A B C A B C
	history := []int32{1, 2, 3, 1, 2, 3, 1, 2, 3}

	// Last bigram: [2, 3]. In trigram table:
	//   [1, 2] → 3
	//   [2, 3] → 1
	//   [3, 1] → 2
	// Draft from [2, 3]: → 1, then [3, 1] → 2, then [1, 2] → 3, then [2, 3] → 1
	drafts := drafter.Draft(history)
	if len(drafts) != 4 {
		t.Fatalf("expected 4 drafts, got %d: %v", len(drafts), drafts)
	}
	expected := []int32{1, 2, 3, 1}
	for i, d := range drafts {
		if d != expected[i] {
			t.Fatalf("draft[%d]: expected %d, got %d", i, expected[i], d)
		}
	}
}

func TestNGramDrafterShortHistory(t *testing.T) {
	drafter := NewNGramDrafter(3, 4)
	// History too short for trigrams.
	drafts := drafter.Draft([]int32{1})
	if len(drafts) != 0 {
		t.Fatalf("expected 0 drafts for short history, got %d", len(drafts))
	}
}

func TestVerifyAllAccepted(t *testing.T) {
	drafts := []int32{10, 20, 30}
	logits := [][]float32{
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5}, // argmax = 10
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5}, // argmax = 20
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5}, // argmax = 30
	}
	accepted, _ := Verify(drafts, logits)
	if accepted != 3 {
		t.Fatalf("expected 3 accepted, got %d", accepted)
	}
}

func TestVerifyPartialAccept(t *testing.T) {
	drafts := []int32{10, 20, 30}
	logits := [][]float32{
		make([]float32, 32), // all zeros → argmax = 0 (not 10)
	}
	logits[0][10] = 5 // argmax = 10 ✓

	logits = append(logits, make([]float32, 32))
	logits[1][25] = 5 // argmax = 25, not 20 ✗

	accepted, corrected := Verify(drafts, logits)
	if accepted != 1 {
		t.Fatalf("expected 1 accepted, got %d", accepted)
	}
	if corrected != 25 {
		t.Fatalf("expected corrected=25, got %d", corrected)
	}
}

func TestVerifyNoDrafts(t *testing.T) {
	accepted, _ := Verify(nil, nil)
	if accepted != 0 {
		t.Fatalf("expected 0 accepted, got %d", accepted)
	}
}
