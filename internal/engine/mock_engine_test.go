package engine

import (
	"context"
	"testing"
)

func TestMockEngineEnqueue(t *testing.T) {
	e := &MockEngine{}
	handle, err := e.Enqueue(context.Background(), &Request{ID: "r1", MaxTokens: 5})
	if err != nil {
		t.Fatalf("enqueue failed: %v", err)
	}
	if handle == nil {
		t.Fatal("expected non-nil handle")
	}
	if handle.Tokens == nil {
		t.Fatal("expected non-nil Tokens channel")
	}
}

func TestMockEnginePushAndReceive(t *testing.T) {
	e := &MockEngine{}
	handle, _ := e.Enqueue(context.Background(), &Request{ID: "r1", MaxTokens: 5})

	e.PushResultsTo("r1", []TokenResult{
		{SequenceID: 1, TokenID: 42, Finished: false},
		{SequenceID: 1, TokenID: 43, Finished: true},
	})

	tok := <-handle.Tokens
	if tok.TokenID != 42 {
		t.Fatalf("expected token 42, got %d", tok.TokenID)
	}
	tok = <-handle.Tokens
	if tok.TokenID != 43 || !tok.Finished {
		t.Fatalf("expected finished token 43, got %+v", tok)
	}

	// Channel should be closed after finished result.
	_, ok := <-handle.Tokens
	if ok {
		t.Fatal("expected channel to be closed")
	}
}

func TestMockEngineStop(t *testing.T) {
	e := &MockEngine{}
	if err := e.Stop(); err != nil {
		t.Fatal(err)
	}
}
