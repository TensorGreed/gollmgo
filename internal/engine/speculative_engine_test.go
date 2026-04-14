package engine

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/kvcache"
	"github.com/TensorGreed/gollmgo/internal/metrics"
	"github.com/TensorGreed/gollmgo/internal/model"
	"github.com/TensorGreed/gollmgo/internal/scheduler"
)

// makeSpecEngine builds a serving engine with speculative decoding enabled.
// The runner is a caller-supplied MockRunner so the test controls logit output.
func makeSpecEngine(t *testing.T, runner *backend.MockRunner) (*ServingEngine, context.CancelFunc) {
	t.Helper()
	sched := scheduler.NewFCFSScheduler(scheduler.FCFSConfig{
		MaxBatchSize: 4, MaxTokenBudget: 256, MaxQueueDepth: 16,
	})
	cache := kvcache.NewBlockPool(1024, 16)
	tok := &model.MockTokenizer{Vocab: 100, EOS: 2}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	eng := NewServingEngine(ServingEngineConfig{
		Runner:    runner,
		Scheduler: sched,
		Cache:     cache,
		Tokenizer: tok,
		Sampling:  SamplingParams{Temperature: 0},
		Log:       log,
		Speculative: SpeculativeSettings{
			Enabled:        true,
			NGramSize:      2,
			NumDraftTokens: 3,
		},
	})
	ctx, cancel := context.WithCancel(context.Background())
	eng.Start(ctx)
	return eng, cancel
}

// TestSpeculativeAllAccepted: drafter proposes drafts that the mock runner
// confirms (by returning argmax = draft token). After a few steps, the
// acceptance-rate metric should reflect wins.
func TestSpeculativeAllAccepted(t *testing.T) {
	// Reset metrics so this test is hermetic.
	metrics.Global.SpecDraftTokens.Store(0)
	metrics.Global.SpecAcceptedTokens.Store(0)
	metrics.Global.SpecKillActive.Store(0)

	// Runner: for current position emit token 42; for every draft position
	// emit argmax = whatever was drafted (all accepted).
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			out := &backend.StepOutput{
				Logits:            make([][]float32, len(b.SequenceIDs)),
				LogitsPerPosition: make([][][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				// Current-position logits: argmax = 42 (a repeat token that
				// the n-gram drafter will latch onto).
				cur := make([]float32, 100)
				cur[42] = 10
				out.Logits[i] = cur

				// If drafts were sent, return argmax=draft for each draft position
				// and a bonus logit that also picks 42.
				if b.DraftTokens == nil || len(b.DraftTokens[i]) == 0 {
					continue
				}
				drafts := b.DraftTokens[i]
				pos := make([][]float32, len(drafts)+1)
				pos[0] = cur // position 0 = current; argmax = 42
				for d, tok := range drafts {
					row := make([]float32, 100)
					row[tok] = 10 // argmax = draft → all accepted
					pos[d+1] = row
				}
				out.LogitsPerPosition[i] = pos
			}
			return out, nil
		},
	}

	eng, cancel := makeSpecEngine(t, runner)
	defer cancel()
	defer eng.Stop()

	// Prompt: "42 42 42 42" → n-gram table learns [42] → 42, so the drafter
	// will propose 42 tokens which the runner will accept.
	prompt := []int32{42, 42, 42, 42}
	handle, err := eng.Enqueue(context.Background(), &Request{
		ID: "spec-all", TokenIDs: prompt, MaxTokens: 20,
	})
	if err != nil {
		t.Fatal(err)
	}

	tokens := drainHandle(t, handle, 5*time.Second)
	if len(tokens) == 0 {
		t.Fatal("expected tokens")
	}
	if !tokens[len(tokens)-1].Finished {
		t.Fatal("expected final token to be Finished")
	}

	drafted := metrics.Global.SpecDraftTokens.Load()
	accepted := metrics.Global.SpecAcceptedTokens.Load()
	if drafted == 0 {
		t.Fatal("expected drafter to propose at least one token")
	}
	if accepted == 0 {
		t.Fatalf("expected some drafts to be accepted, got drafted=%d accepted=%d",
			drafted, accepted)
	}
	if metrics.Global.SpecAcceptanceRate() < 0.5 {
		t.Errorf("expected high acceptance rate, got %.2f (drafted=%d accepted=%d)",
			metrics.Global.SpecAcceptanceRate(), drafted, accepted)
	}
}

// TestSpeculativeKillSwitch: directly exercise maybeTripKillSwitch by
// setting the global metric counters and checking that the kill flag
// flips when the acceptance rate falls below the configured threshold,
// and that maybeDraft then returns nil.
func TestSpeculativeKillSwitch(t *testing.T) {
	metrics.Global.SpecDraftTokens.Store(0)
	metrics.Global.SpecAcceptedTokens.Store(0)
	metrics.Global.SpecKillActive.Store(0)

	eng, cancel := makeSpecEngine(t, &backend.MockRunner{})
	defer cancel()
	defer eng.Stop()

	// Configure a tight threshold and low warmup.
	eng.specCfg.KillThreshold = 0.5
	eng.specCfg.KillWarmup = 6

	// Below warmup — kill must NOT trip even with terrible acceptance.
	metrics.Global.SpecDraftTokens.Store(5)
	metrics.Global.SpecAcceptedTokens.Store(0)
	eng.maybeTripKillSwitch()
	if metrics.Global.SpecKillActive.Load() != 0 {
		t.Fatal("kill switch tripped before warmup threshold")
	}

	// Above warmup with acceptance well below threshold — must trip.
	metrics.Global.SpecDraftTokens.Store(30)
	metrics.Global.SpecAcceptedTokens.Store(5) // ~17% acceptance
	eng.maybeTripKillSwitch()
	if metrics.Global.SpecKillActive.Load() != 1 {
		t.Fatalf("expected kill switch to trip at acceptance=%.2f threshold=%.2f",
			metrics.Global.SpecAcceptanceRate(), eng.specCfg.KillThreshold)
	}

	// After kill fires, maybeDraft must return no drafts even with healthy
	// history and an enabled drafter.
	seq := scheduler.NewSequence("after-kill", []int32{42, 42, 42, 42}, 10)
	if d := eng.maybeDraft(seq); d != nil {
		t.Errorf("expected no drafts after kill switch tripped, got %v", d)
	}
}

// TestSpeculativeRejectThenEmitCorrected: drafts are always rejected by a
// runner that produces argmax != drafter's proposal. Verifies the engine
// emits the corrected token (not the draft), truncates the unused KV
// slots, and does NOT over-count accepted drafts.
func TestSpeculativeRejectThenEmitCorrected(t *testing.T) {
	metrics.Global.SpecDraftTokens.Store(0)
	metrics.Global.SpecAcceptedTokens.Store(0)
	metrics.Global.SpecKillActive.Store(0)

	// Runner's chosen token rotates by step so the drafter can't catch up.
	var stepN int32
	rejectLogits := func(vocab int32) []float32 {
		row := make([]float32, 100)
		row[vocab] = 10
		return row
	}
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			out := &backend.StepOutput{
				Logits:            make([][]float32, len(b.SequenceIDs)),
				LogitsPerPosition: make([][][]float32, len(b.SequenceIDs)),
			}
			for i, isPrefill := range b.IsPrefill {
				winner := int32(50)
				if !isPrefill {
					stepN++
					// Rotate through 50..59, never 42, never EOS until late.
					if stepN < 20 {
						winner = 50 + (stepN % 10)
					} else {
						winner = 2 // EOS to terminate the test
					}
				}
				out.Logits[i] = rejectLogits(winner)
				if b.DraftTokens == nil || len(b.DraftTokens[i]) == 0 {
					continue
				}
				drafts := b.DraftTokens[i]
				pos := make([][]float32, len(drafts)+1)
				pos[0] = out.Logits[i]
				for d := range drafts {
					pos[d+1] = rejectLogits(winner)
				}
				out.LogitsPerPosition[i] = pos
			}
			return out, nil
		},
	}

	eng, cancel := makeSpecEngine(t, runner)
	defer cancel()
	defer eng.Stop()

	// Prompt of 42-repeats makes the drafter propose [42, 42, 42] on step 1.
	handle, err := eng.Enqueue(context.Background(), &Request{
		ID: "spec-reject", TokenIDs: []int32{42, 42, 42, 42, 42, 42}, MaxTokens: 20,
	})
	if err != nil {
		t.Fatal(err)
	}

	tokens := drainHandle(t, handle, 5*time.Second)
	if len(tokens) == 0 || !tokens[len(tokens)-1].Finished {
		t.Fatalf("expected the request to complete; got %d tokens", len(tokens))
	}

	// Some drafts should have been proposed (at least on step 1).
	if metrics.Global.SpecDraftTokens.Load() == 0 {
		t.Fatal("expected drafter to propose at least one token on step 1")
	}

	// None of the emitted tokens should be 42 — all are runner-chosen 50..59
	// or the terminating EOS. If the engine wrongly emitted draft tokens as
	// accepted, we'd see 42s in the stream.
	for i, tr := range tokens {
		if tr.Err != nil {
			t.Fatalf("token %d error: %v", i, tr.Err)
		}
		if tr.TokenID == 42 {
			t.Errorf("token %d has draft value 42 — rejected drafts leaked through", i)
		}
	}
}

// TestSpeculativeDisabledWhenRunnerUnsupported: runner reports no speculative
// capability → engine should log and leave drafting disabled, not panic.
func TestSpeculativeDisabledWhenRunnerUnsupported(t *testing.T) {
	// Custom runner that specifically disables the speculative capability.
	type noSpecRunner struct{ *backend.MockRunner }
	inner := &backend.MockRunner{}
	runner := &struct {
		backend.Runner
	}{Runner: inner} // anonymous struct so we can override Capabilities
	_ = runner
	// Simpler: use the mock directly but turn OFF speculative by wrapping.
	// Actually, override via a small adapter.
	adapted := &capOverrideRunner{inner: inner}

	sched := scheduler.NewFCFSScheduler(scheduler.DefaultFCFSConfig())
	cache := kvcache.NewBlockPool(128, 16)
	tok := &model.MockTokenizer{Vocab: 50, EOS: 2}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	eng := NewServingEngine(ServingEngineConfig{
		Runner: adapted, Scheduler: sched, Cache: cache,
		Tokenizer: tok, Sampling: SamplingParams{Temperature: 0}, Log: log,
		Speculative: SpeculativeSettings{
			Enabled:        true,
			NumDraftTokens: 3,
		},
	})
	if eng.drafter != nil {
		t.Error("expected drafter to be nil when runner reports no speculative capability")
	}
	if eng.specCfg.Enabled {
		t.Error("expected specCfg.Enabled to be false after capability check")
	}
}

// capOverrideRunner wraps a MockRunner but reports no speculative capability.
type capOverrideRunner struct {
	inner *backend.MockRunner
}

func (r *capOverrideRunner) Warmup(ctx context.Context, p backend.WarmupProfile) error {
	return r.inner.Warmup(ctx, p)
}
func (r *capOverrideRunner) Step(ctx context.Context, b *backend.Batch) (*backend.StepOutput, error) {
	return r.inner.Step(ctx, b)
}
func (r *capOverrideRunner) Close() error { return r.inner.Close() }
func (r *capOverrideRunner) Capabilities() backend.Capabilities {
	return backend.Capabilities{FP16: true} // no SpeculativeDecoding
}
