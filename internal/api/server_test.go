package api

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/TensorGreed/gollmgo/internal/config"
	"github.com/TensorGreed/gollmgo/internal/engine"
)

func newTestServer(t *testing.T) (*Server, *engine.MockEngine) {
	t.Helper()
	eng := &engine.MockEngine{}
	cfg := config.DefaultConfig()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return NewServer(cfg, eng, log), eng
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

	eng.PushResults([]engine.TokenResult{
		{SequenceID: 1, TokenID: 72, Finished: false},
		{SequenceID: 1, TokenID: 105, Finished: true},
	})

	body := `{"model":"test","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
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

	eng.PushResults([]engine.TokenResult{
		{SequenceID: 1, TokenID: 65, Finished: false},
		{SequenceID: 1, TokenID: 66, Finished: true},
	})

	body := `{"model":"test","messages":[{"role":"user","content":"go"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	respBody := rec.Body.String()
	if !strings.Contains(respBody, "data: ") {
		t.Fatal("expected SSE data lines in streaming response")
	}
	if !strings.Contains(respBody, "data: [DONE]") {
		t.Fatal("expected [DONE] sentinel in streaming response")
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
