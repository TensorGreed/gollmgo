package engine

import (
	"context"
	"testing"
)

func TestMockEngineEnqueueAndQueueLen(t *testing.T) {
	e := &MockEngine{}
	if err := e.Enqueue(context.Background(), &Request{ID: "r1"}); err != nil {
		t.Fatalf("enqueue failed: %v", err)
	}
	if e.QueueLen() != 1 {
		t.Fatalf("expected queue len 1, got %d", e.QueueLen())
	}
}

func TestMockEngineNextTokens(t *testing.T) {
	e := &MockEngine{}

	// Empty returns nil.
	tokens, err := e.NextTokens(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tokens != nil {
		t.Fatal("expected nil from empty engine")
	}

	// Push and drain.
	e.PushResults([]TokenResult{{SequenceID: 1, TokenID: 42, Finished: false}})
	tokens, err = e.NextTokens(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 1 || tokens[0].TokenID != 42 {
		t.Fatalf("unexpected tokens: %+v", tokens)
	}

	// Should be empty again.
	tokens, _ = e.NextTokens(context.Background())
	if tokens != nil {
		t.Fatal("expected nil after drain")
	}
}

func TestMockEngineStop(t *testing.T) {
	e := &MockEngine{}
	if err := e.Stop(); err != nil {
		t.Fatal(err)
	}
}
