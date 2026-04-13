package engine

import (
	"context"
	"testing"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/model"
)

func TestInferenceEngineGreedyGeneration(t *testing.T) {
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
					logits[eosID] = 10.0
				} else {
					logits[42] = 10.0
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
		Sampling:  SamplingParams{Temperature: 0},
	})

	handle, err := eng.Enqueue(context.Background(), &Request{
		ID:        "test-1",
		TokenIDs:  []int32{1, 2, 3},
		MaxTokens: 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	var allTokens []TokenResult
	for i := 0; i < 20; i++ {
		if err := eng.RunStep(context.Background()); err != nil {
			t.Fatal(err)
		}
		// Drain available tokens.
		for {
			select {
			case tok := <-handle.Tokens:
				allTokens = append(allTokens, tok)
				if tok.Finished {
					goto done
				}
			default:
				goto nextStep
			}
		}
	nextStep:
		if !eng.HasPending() {
			break
		}
	}
done:

	if len(allTokens) == 0 {
		t.Fatal("expected at least one token")
	}

	for i, tok := range allTokens {
		if i < 3 {
			if tok.TokenID != 42 {
				t.Errorf("step %d: expected token 42, got %d", i, tok.TokenID)
			}
		}
	}

	last := allTokens[len(allTokens)-1]
	if !last.Finished {
		t.Error("last token should be finished")
	}
}

func TestInferenceEngineMaxTokensStop(t *testing.T) {
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

	handle, _ := eng.Enqueue(context.Background(), &Request{
		ID:        "test-max",
		TokenIDs:  []int32{1},
		MaxTokens: 3,
	})

	var allTokens []TokenResult
	for i := 0; i < 10; i++ {
		eng.RunStep(context.Background())
		for {
			select {
			case tok := <-handle.Tokens:
				allTokens = append(allTokens, tok)
				if tok.Finished {
					goto maxDone
				}
			default:
				goto maxNext
			}
		}
	maxNext:
		if !eng.HasPending() {
			break
		}
	}
maxDone:

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

	_, err := eng.Enqueue(context.Background(), &Request{ID: "x", TokenIDs: []int32{1}})
	if err == nil {
		t.Fatal("expected error after stop")
	}
}

func TestInferenceEngineMultipleSequences(t *testing.T) {
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				logits := make([]float32, 20)
				logits[2] = 10.0 // EOS
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

	hA, _ := eng.Enqueue(context.Background(), &Request{ID: "a", TokenIDs: []int32{1}, MaxTokens: 5})
	hB, _ := eng.Enqueue(context.Background(), &Request{ID: "b", TokenIDs: []int32{3, 4}, MaxTokens: 5})

	eng.RunStep(context.Background())

	tokA := <-hA.Tokens
	tokB := <-hB.Tokens

	if !tokA.Finished || !tokB.Finished {
		t.Error("expected both sequences to finish on EOS")
	}
}
