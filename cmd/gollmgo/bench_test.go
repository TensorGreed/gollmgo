package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/TensorGreed/gollmgo/internal/model"
)

func TestBuildPromptTextExactByteTokenizer(t *testing.T) {
	tok := model.NewByteLevelTokenizer(32000, 2)

	prompt, mode, err := buildPromptText(tok, 64, 7)
	if err != nil {
		t.Fatalf("buildPromptText error: %v", err)
	}
	if mode != "tokenizer_exact" {
		t.Fatalf("expected tokenizer_exact mode, got %q", mode)
	}

	count, err := promptTokenCount(tok, prompt)
	if err != nil {
		t.Fatalf("promptTokenCount error: %v", err)
	}
	if count != 64 {
		t.Fatalf("expected 64 tokens, got %d", count)
	}
}

func TestDoServingBenchRequestRejectsNonOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	_, err := doServingBenchRequest(srv.Client(), srv.URL, "hello", 8, model.NewByteLevelTokenizer(32000, 2))
	if err == nil || !strings.Contains(err.Error(), "unexpected status 503") {
		t.Fatalf("expected status error, got %v", err)
	}
}

func TestCollectStreamMetricsTokenizerAware(t *testing.T) {
	body := strings.NewReader(strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"ab"}}]}`,
		"",
		`data: {"choices":[{"delta":{"content":"c"}}]}`,
		"",
		`data: [DONE]`,
		"",
	}, "\n"))

	ttft, itls, tokens, err := collectStreamMetrics(body, time.Now(), model.NewByteLevelTokenizer(32000, 2))
	if err != nil {
		t.Fatalf("collectStreamMetrics error: %v", err)
	}
	if ttft <= 0 {
		t.Fatalf("expected positive TTFT, got %v", ttft)
	}
	if tokens != 3 {
		t.Fatalf("expected 3 tokens, got %d", tokens)
	}
	if len(itls) != 1 {
		t.Fatalf("expected 1 ITL entry, got %d", len(itls))
	}
}

func TestAggregateServingSamples(t *testing.T) {
	opts := benchOptions{
		NumPrompts:     100,
		PromptLen:      128,
		OutputLen:      64,
		Concurrency:    4,
		QPS:            5,
		WarmupRequests: 10,
		Repetitions:    3,
	}
	samples := []benchSample{
		{ElapsedSeconds: 30, TokensPerSecond: 100, RequestsPerSec: 2, TTFTP50Ms: 20, TTFTP95Ms: 30, TTFTP99Ms: 40, E2EP50Ms: 100, E2EP95Ms: 150, E2EP99Ms: 200, ITLP50Ms: 10, ITLP95Ms: 15, ITLP99Ms: 20},
		{ElapsedSeconds: 20, TokensPerSecond: 120, RequestsPerSec: 3, TTFTP50Ms: 15, TTFTP95Ms: 25, TTFTP99Ms: 35, E2EP50Ms: 90, E2EP95Ms: 140, E2EP99Ms: 180, ITLP50Ms: 9, ITLP95Ms: 14, ITLP99Ms: 18},
		{ElapsedSeconds: 25, TokensPerSecond: 110, RequestsPerSec: 2.5, TTFTP50Ms: 18, TTFTP95Ms: 28, TTFTP99Ms: 38, E2EP50Ms: 95, E2EP95Ms: 145, E2EP99Ms: 190, ITLP50Ms: 9.5, ITLP95Ms: 14.5, ITLP99Ms: 19},
	}

	result := aggregateServingSamples(opts, samples, "tokenizer_exact", 300, 4)
	if result.Mode != "serving" {
		t.Fatalf("expected serving mode, got %q", result.Mode)
	}
	if result.MeasuredRequests != 300 {
		t.Fatalf("expected measured_requests=300, got %d", result.MeasuredRequests)
	}
	if result.ErrorCount != 4 {
		t.Fatalf("expected error_count=4, got %d", result.ErrorCount)
	}
	if result.Repetitions != 3 {
		t.Fatalf("expected repetitions=3, got %d", result.Repetitions)
	}
	if result.ArrivalMode != "poisson_qps" {
		t.Fatalf("expected poisson_qps arrival mode, got %q", result.ArrivalMode)
	}
	if result.TokensPerSecond != 110 {
		t.Fatalf("expected median tokens/sec 110, got %v", result.TokensPerSecond)
	}
	if result.TTFTP50Ms != 18 {
		t.Fatalf("expected median TTFT P50 18, got %v", result.TTFTP50Ms)
	}
	if result.TokenCountMode != "tokenizer_completion" {
		t.Fatalf("expected tokenizer_completion token count mode, got %q", result.TokenCountMode)
	}
}
