package api

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/TensorGreed/gollmgo/internal/config"
	"github.com/TensorGreed/gollmgo/internal/engine"
	"github.com/TensorGreed/gollmgo/internal/model"
)

func newTestServer(t *testing.T) (*Server, *engine.MockEngine) {
	t.Helper()
	eng := &engine.MockEngine{}
	cfg := config.DefaultConfig()
	tok := model.NewByteLevelTokenizer(256, 2)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return NewServer(cfg, eng, tok, log), eng
}

// --- Health endpoints ---

func TestServerLive(t *testing.T) {
	s, _ := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/health/live", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

func TestServerReady(t *testing.T) {
	s, _ := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/health/ready", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

// --- Models endpoint ---

func TestServerModels(t *testing.T) {
	s, _ := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var resp ModelsResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Data) == 0 {
		t.Fatal("expected at least one model")
	}
}

// --- Metrics endpoint ---

func TestServerMetrics(t *testing.T) {
	s, _ := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "gollmgo_requests_total") {
		t.Fatal("expected gollmgo_requests_total in metrics output")
	}
}

// --- Chat completions: validation ---

func TestChatCompletionsEmptyBody(t *testing.T) {
	s, _ := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 400 {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestChatCompletionsInvalidJSON(t *testing.T) {
	s, _ := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("not json"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 400 {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestChatCompletionsEmptyRole(t *testing.T) {
	s, _ := newTestServer(t)
	body := `{"messages":[{"role":"","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 400 {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestChatCompletionsNegativeMaxTokens(t *testing.T) {
	s, _ := newTestServer(t)
	body := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 400 {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

// --- Chat completions: non-streaming with mock ---

func TestChatCompletionsNonStreaming(t *testing.T) {
	s, eng := newTestServer(t)

	body := `{"model":"test","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	// Push results asynchronously after the engine registers the request.
	go func() {
		for len(eng.PendingRequestIDs()) == 0 {
			// spin until enqueued
		}
		ids := eng.PendingRequestIDs()
		eng.PushResultsTo(ids[0], []engine.TokenResult{
			{SequenceID: 1, TokenID: 72, Finished: false}, // 'H'
			{SequenceID: 1, TokenID: 105, Finished: true},  // 'i'
		})
	}()

	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp ChatCompletionResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Choices[0].Message.Content != "Hi" {
		t.Fatalf("expected 'Hi', got %q", resp.Choices[0].Message.Content)
	}
	if resp.Usage.CompletionTokens != 2 {
		t.Fatalf("expected 2 completion tokens, got %d", resp.Usage.CompletionTokens)
	}
}

// --- Chat completions: streaming with mock ---

func TestChatCompletionsStreaming(t *testing.T) {
	s, eng := newTestServer(t)

	body := `{"model":"test","messages":[{"role":"user","content":"go"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	go func() {
		for len(eng.PendingRequestIDs()) == 0 {
		}
		ids := eng.PendingRequestIDs()
		eng.PushResultsTo(ids[0], []engine.TokenResult{
			{SequenceID: 1, TokenID: 65, Finished: false}, // 'A'
			{SequenceID: 1, TokenID: 66, Finished: true},  // 'B'
		})
	}()

	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	respBody := rec.Body.String()
	if !strings.Contains(respBody, "data: ") {
		t.Fatal("expected SSE data lines")
	}
	if !strings.Contains(respBody, "data: [DONE]") {
		t.Fatal("expected [DONE] sentinel")
	}
}

// --- API behavior on channel close without Finished token ---

func TestChatCompletionsChannelCloseNonStreaming(t *testing.T) {
	s, eng := newTestServer(t)

	body := `{"model":"test","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	// Close the channel without sending a Finished token (simulates engine shutdown).
	go func() {
		for len(eng.PendingRequestIDs()) == 0 {
		}
		ids := eng.PendingRequestIDs()
		// Send one token, then close without Finished.
		eng.PushResultsTo(ids[0], []engine.TokenResult{
			{SequenceID: 1, TokenID: 72, Finished: false},
		})
		// Force close the channel by stopping the engine.
		eng.Stop()
	}()

	s.Handler().ServeHTTP(rec, req)

	// Should NOT be 200. The channel was closed without a Finished token.
	if rec.Code == 200 {
		t.Fatal("expected non-200 status on engine shutdown, got 200")
	}
}

func TestChatCompletionsChannelCloseStreaming(t *testing.T) {
	s, eng := newTestServer(t)

	body := `{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	go func() {
		for len(eng.PendingRequestIDs()) == 0 {
		}
		ids := eng.PendingRequestIDs()
		eng.PushResultsTo(ids[0], []engine.TokenResult{
			{SequenceID: 1, TokenID: 65, Finished: false},
		})
		eng.Stop()
	}()

	s.Handler().ServeHTTP(rec, req)

	respBody := rec.Body.String()
	// Must have finish_reason "error" — not error text as content.
	if !strings.Contains(respBody, `"finish_reason":"error"`) {
		t.Fatalf("expected finish_reason error in streaming response, got:\n%s", respBody)
	}
	// Must NOT have error text embedded as assistant content.
	if strings.Contains(respBody, "[error:") {
		t.Fatal("error text should not appear as assistant content")
	}
	if !strings.Contains(respBody, "[DONE]") {
		t.Fatal("expected [DONE] sentinel")
	}
}

func TestChatCompletionsStreamingTokenError(t *testing.T) {
	s, eng := newTestServer(t)

	body := `{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	go func() {
		for len(eng.PendingRequestIDs()) == 0 {
		}
		ids := eng.PendingRequestIDs()
		eng.PushResultsTo(ids[0], []engine.TokenResult{
			{SequenceID: 1, TokenID: 65, Finished: false},
			{SequenceID: 1, Finished: true, Err: fmt.Errorf("GPU OOM")},
		})
	}()

	s.Handler().ServeHTTP(rec, req)

	respBody := rec.Body.String()
	// Must terminate with finish_reason "error", not embed OOM text as content.
	if !strings.Contains(respBody, `"finish_reason":"error"`) {
		t.Fatalf("expected finish_reason error, got:\n%s", respBody)
	}
	if strings.Contains(respBody, "GPU OOM") {
		t.Fatal("backend error details should not leak to client as content")
	}
	if !strings.Contains(respBody, "[DONE]") {
		t.Fatal("expected [DONE]")
	}
}

func TestChatCompletionsTokenError(t *testing.T) {
	s, eng := newTestServer(t)

	body := `{"model":"test","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	go func() {
		for len(eng.PendingRequestIDs()) == 0 {
		}
		ids := eng.PendingRequestIDs()
		eng.PushResultsTo(ids[0], []engine.TokenResult{
			{SequenceID: 1, Finished: true, Err: fmt.Errorf("GPU OOM")},
		})
	}()

	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 500 {
		t.Fatalf("expected 500 on token error, got %d", rec.Code)
	}
}

// --- Config loading ---

func TestConfigLoadFile(t *testing.T) {
	tmpFile := t.TempDir() + "/config.json"
	cfg := config.DefaultConfig()
	cfg.Port = 9999
	data, _ := json.Marshal(cfg)
	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		t.Fatal(err)
	}

	loaded, err := config.LoadFile(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Port != 9999 {
		t.Fatalf("expected port 9999, got %d", loaded.Port)
	}
}
