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
