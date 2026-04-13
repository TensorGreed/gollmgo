package engine

import (
	"context"
	"testing"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/model"
)

func TestInferenceEngineGreedyGeneration(t *testing.T) {
	// Mock runner returns logits where token ID 42 is always the argmax,
	// except after 3 tokens it returns EOS (token ID 2).
	stepCount := 0
	eosID := int32(2)
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			stepCount++
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				logits := make([]float32, 100)
				if stepCount >= 4 {
					logits[eosID] = 10.0 // EOS wins
				} else {
					logits[42] = 10.0 // token 42 wins
				}
				out.Logits[i] = logits
			}
			return out, nil
		},
	}

	tok := &model.MockTokenizer{Vocab: 100, EOS: eosID}
	eng := NewInferenceEngine(InferenceEngineConfig{
		Runner:    runner,
		Tokenizer: tok,
		Sampling:  SamplingParams{Temperature: 0}, // greedy
		MaxTokens: 10,
	})

	// Enqueue a request.
	err := eng.Enqueue(context.Background(), &Request{
		ID:        "test-1",
		TokenIDs:  []int32{1, 2, 3},
		MaxTokens: 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Run steps until finished.
	var allTokens []TokenResult
	for i := 0; i < 20; i++ {
		if err := eng.RunStep(context.Background()); err != nil {
			t.Fatal(err)
		}
		tokens, err := eng.NextTokens(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		allTokens = append(allTokens, tokens...)

		if !eng.HasPending() {
			break
		}
	}

	if len(allTokens) == 0 {
		t.Fatal("expected at least one token")
	}

	// First 3 tokens should be 42, then EOS.
	for i, tok := range allTokens {
		if i < 3 {
			if tok.TokenID != 42 {
				t.Errorf("step %d: expected token 42, got %d", i, tok.TokenID)
			}
			if tok.Finished {
				t.Errorf("step %d: should not be finished", i)
			}
		}
	}

	// Last token should be finished.
	last := allTokens[len(allTokens)-1]
	if !last.Finished {
		t.Error("last token should be finished")
	}
}

func TestInferenceEngineMaxTokensStop(t *testing.T) {
	// Runner always returns token 42 (never EOS).
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				logits := make([]float32, 50)
				logits[42] = 10.0
				out.Logits[i] = logits
			}
			return out, nil
		},
	}

	tok := &model.MockTokenizer{Vocab: 50, EOS: 0}
	eng := NewInferenceEngine(InferenceEngineConfig{
		Runner:    runner,
		Tokenizer: tok,
		Sampling:  SamplingParams{Temperature: 0},
	})

	eng.Enqueue(context.Background(), &Request{
		ID:        "test-max",
		TokenIDs:  []int32{1},
		MaxTokens: 3,
	})

	var allTokens []TokenResult
	for i := 0; i < 10; i++ {
		eng.RunStep(context.Background())
		tokens, _ := eng.NextTokens(context.Background())
		allTokens = append(allTokens, tokens...)
		if !eng.HasPending() {
			break
		}
	}

	if len(allTokens) != 3 {
		t.Fatalf("expected exactly 3 tokens, got %d", len(allTokens))
	}
	if !allTokens[2].Finished {
		t.Error("third token should be finished (max_tokens reached)")
	}
}

func TestInferenceEngineStop(t *testing.T) {
	runner := &backend.MockRunner{}
	tok := &model.MockTokenizer{Vocab: 10, EOS: 0}
	eng := NewInferenceEngine(InferenceEngineConfig{
		Runner:    runner,
		Tokenizer: tok,
		Sampling:  DefaultSamplingParams(),
	})

	eng.Stop()

	err := eng.Enqueue(context.Background(), &Request{ID: "x", TokenIDs: []int32{1}})
	if err == nil {
		t.Fatal("expected error after stop")
	}
}

func TestInferenceEngineMultipleSequences(t *testing.T) {
	// Runner returns different tokens for different batch positions.
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				logits := make([]float32, 20)
				// EOS immediately for all sequences.
				logits[2] = 10.0
				out.Logits[i] = logits
			}
			return out, nil
		},
	}

	tok := &model.MockTokenizer{Vocab: 20, EOS: 2}
	eng := NewInferenceEngine(InferenceEngineConfig{
		Runner:    runner,
		Tokenizer: tok,
		Sampling:  SamplingParams{Temperature: 0},
	})

	// Enqueue two requests.
	eng.Enqueue(context.Background(), &Request{ID: "a", TokenIDs: []int32{1}, MaxTokens: 5})
	eng.Enqueue(context.Background(), &Request{ID: "b", TokenIDs: []int32{3, 4}, MaxTokens: 5})

	eng.RunStep(context.Background())
	tokens, _ := eng.NextTokens(context.Background())

	// Both should finish with EOS on first step.
	if len(tokens) != 2 {
		t.Fatalf("expected 2 tokens, got %d", len(tokens))
	}
	for _, tok := range tokens {
		if !tok.Finished {
			t.Error("expected both sequences to finish on EOS")
		}
	}
}
