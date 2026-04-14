package config

import (
	"errors"
	"testing"
)

func TestDefaultConfigIsValid(t *testing.T) {
	c := DefaultConfig()
	if err := c.Validate(); err != nil {
		t.Fatalf("default config should be valid: %v", err)
	}
}

func TestValidatePort(t *testing.T) {
	c := DefaultConfig()
	c.Port = 0
	if !errors.Is(c.Validate(), ErrInvalidPort) {
		t.Fatal("expected ErrInvalidPort for port 0")
	}
	c.Port = 70000
	if !errors.Is(c.Validate(), ErrInvalidPort) {
		t.Fatal("expected ErrInvalidPort for port 70000")
	}
}

func TestValidateBatchSize(t *testing.T) {
	c := DefaultConfig()
	c.MaxBatchSize = -1
	if !errors.Is(c.Validate(), ErrInvalidBatchSize) {
		t.Fatal("expected ErrInvalidBatchSize")
	}
}

func TestValidateBlockSize(t *testing.T) {
	c := DefaultConfig()
	c.BlockSize = 0
	if !errors.Is(c.Validate(), ErrInvalidBlockSize) {
		t.Fatal("expected ErrInvalidBlockSize")
	}
}

func TestValidateMemoryFraction(t *testing.T) {
	c := DefaultConfig()
	c.MaxMemoryFraction = 0
	if !errors.Is(c.Validate(), ErrInvalidMemoryFraction) {
		t.Fatal("expected ErrInvalidMemoryFraction for 0")
	}
	c.MaxMemoryFraction = 1.5
	if !errors.Is(c.Validate(), ErrInvalidMemoryFraction) {
		t.Fatal("expected ErrInvalidMemoryFraction for 1.5")
	}
}

func TestValidateSchedulerPolicy(t *testing.T) {
	c := DefaultConfig()
	c.SchedulerPolicy = "bogus"
	if !errors.Is(c.Validate(), ErrInvalidSchedulerPolicy) {
		t.Fatal("expected ErrInvalidSchedulerPolicy")
	}
}

func TestValidatePreemptMode(t *testing.T) {
	c := DefaultConfig()
	// Swap is now a valid mode (backend support is discovered at runtime).
	c.PreemptMode = "swap"
	if err := c.Validate(); err != nil {
		t.Fatalf("expected swap mode to validate, got %v", err)
	}
	c.PreemptMode = "bogus"
	if !errors.Is(c.Validate(), ErrInvalidPreemptMode) {
		t.Fatal("expected ErrInvalidPreemptMode for unknown mode")
	}
}

func TestValidateAutoPreemptRequiresPriority(t *testing.T) {
	c := DefaultConfig()
	c.AutoPreempt = true
	c.SchedulerPolicy = "fcfs"
	if !errors.Is(c.Validate(), ErrAutoPreemptRequiresPrio) {
		t.Fatal("expected ErrAutoPreemptRequiresPrio")
	}
}

func TestValidatePrefixCacheCap(t *testing.T) {
	c := DefaultConfig()
	c.PrefixCacheMaxBlocks = -1
	if !errors.Is(c.Validate(), ErrInvalidPrefixCacheCap) {
		t.Fatal("expected ErrInvalidPrefixCacheCap")
	}
}

func TestValidateSpeculativeEnabled(t *testing.T) {
	// Enabled speculative decoding is accepted; the engine activates it only
	// when the runner reports SpeculativeDecoding capability.
	c := DefaultConfig()
	c.Speculative.Enabled = true
	c.Speculative.Mode = "ngram"
	c.Speculative.NGramSize = 3
	c.Speculative.NumDraftTokens = 4
	if err := c.Validate(); err != nil {
		t.Fatalf("expected enabled ngram speculative to validate, got %v", err)
	}
}

func TestValidateSpeculativeModeRejected(t *testing.T) {
	c := DefaultConfig()
	c.Speculative.Enabled = true
	c.Speculative.Mode = "bogus"
	if !errors.Is(c.Validate(), ErrInvalidSpeculativeMode) {
		t.Fatal("expected ErrInvalidSpeculativeMode")
	}
}

func TestValidateSpeculativeDraftModeRejected(t *testing.T) {
	c := DefaultConfig()
	c.Speculative.Enabled = true
	c.Speculative.Mode = "draft"
	if !errors.Is(c.Validate(), ErrInvalidSpeculativeMode) {
		t.Fatal("expected ErrInvalidSpeculativeMode for draft mode")
	}
}
