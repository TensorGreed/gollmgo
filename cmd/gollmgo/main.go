// Package main is the CLI entrypoint for gollmgo.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/TensorGreed/gollmgo/internal/api"
	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/config"
	"github.com/TensorGreed/gollmgo/internal/engine"
	"github.com/TensorGreed/gollmgo/internal/kvcache"
	"github.com/TensorGreed/gollmgo/internal/model"
	"github.com/TensorGreed/gollmgo/internal/scheduler"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: gollmgo <command> [args]")
		fmt.Fprintln(os.Stderr, "commands: serve, bench, doctor")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		cmdServe(os.Args[2:])
	case "bench":
		cmdBench(os.Args[2:])
	case "doctor":
		cmdDoctor(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}

func cmdServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	configPath := fs.String("config", "", "path to config file (JSON)")
	host := fs.String("host", "", "bind host (overrides config)")
	port := fs.Int("port", 0, "bind port (overrides config)")
	logLevel := fs.String("log-level", "", "log level: debug, info, warn, error")
	modelPath := fs.String("model", "", "path to model file (.safetensors)")
	tokenizerPath := fs.String("tokenizer", "", "path to tokenizer.json (default: auto-detect)")
	deviceID := fs.Int("device", 0, "CUDA device ID")
	fs.Parse(args)

	// Load config.
	cfg := config.DefaultConfig()
	if *configPath != "" {
		var err error
		cfg, err = config.LoadFile(*configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	}

	// Apply CLI overrides.
	if *host != "" {
		cfg.Host = *host
	}
	if *port > 0 {
		cfg.Port = *port
	}
	if *logLevel != "" {
		cfg.LogLevel = *logLevel
	}
	if *modelPath != "" {
		cfg.ModelPath = *modelPath
	}
	if *tokenizerPath != "" {
		cfg.TokenizerPath = *tokenizerPath
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	log := newLogger(cfg.LogLevel)

	// Initialize runner and tokenizer.
	var runner backend.Runner
	var tokenizer model.Tokenizer

	numBlocks := 1024 // default for mock mode
	if cfg.ModelPath != "" {
		var err error
		runner, tokenizer, numBlocks, err = initGPURunner(log, cfg, *deviceID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "GPU init failed: %v\n", err)
			os.Exit(1)
		}
		log.Info("GPU model loaded", "model", cfg.ModelPath)
	} else {
		log.Info("no --model provided, using mock runner (development mode)")
		runner = &backend.MockRunner{}
		tokenizer = model.NewByteLevelTokenizer(32000, 2)
	}

	log.Info("starting gollmgo",
		"host", cfg.Host,
		"port", cfg.Port,
		"max_batch_size", cfg.MaxBatchSize,
		"block_size", cfg.BlockSize,
		"kv_blocks", numBlocks)

	// Create Go-side KV cache pool matching the GPU-side allocation.
	cache := kvcache.NewBlockPool(numBlocks, cfg.BlockSize)

	// Create scheduler from config-selected policy.
	policy := schedulerPolicyFromString(cfg.SchedulerPolicy)
	preemptMode := preemptModeFromString(cfg.PreemptMode)
	sched := scheduler.NewScheduler(policy, scheduler.SchedulerConfig{
		MaxBatchSize:     cfg.MaxBatchSize,
		MaxTokenBudget:   cfg.MaxTokenBudget,
		MaxQueueDepth:    cfg.MaxQueueDepth,
		PrefillChunkSize: cfg.PrefillChunkSize,
		AutoPreempt:      cfg.AutoPreempt,
		PreemptMode:      preemptMode,
	})
	log.Info("scheduler ready", "policy", policy.String(), "preempt_mode", preemptMode.String(),
		"prefill_chunk_size", cfg.PrefillChunkSize)

	// Create serving engine.
	eng := engine.NewServingEngine(engine.ServingEngineConfig{
		Runner:            runner,
		Scheduler:         sched,
		Cache:             cache,
		Tokenizer:         tokenizer,
		Sampling:          engine.DefaultSamplingParams(),
		Log:               log,
		EnablePrefixCache: cfg.PrefixCaching,
		MaxPrefixBlocks:   cfg.PrefixCacheMaxBlocks,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eng.Start(ctx)

	// Create and start API server.
	srv := api.NewServer(cfg, eng, tokenizer, log)

	// Graceful shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	log.Info("server ready", "addr", fmt.Sprintf("%s:%d", cfg.Host, cfg.Port))

	<-sigCh
	log.Info("shutting down...")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	eng.Stop()
	srv.Shutdown(shutdownCtx)

	// Release GPU resources (model, KV cache, backend handle).
	if runner != nil {
		if err := runner.Close(); err != nil {
			log.Error("runner close error", "error", err)
		}
	}
	log.Info("shutdown complete")
}

// benchConfig holds frozen benchmark parameters loaded from --config.
type benchConfig struct {
	Benchmark struct {
		Mode        string `json:"mode"`
		NumPrompts  int    `json:"num_prompts"`
		PromptLen   int    `json:"prompt_len"`
		OutputLen   int    `json:"output_len"`
		Concurrency int    `json:"concurrency"`
	} `json:"benchmark"`
}

func cmdBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	mode := fs.String("mode", "offline", "benchmark mode: offline, serving")
	numPrompts := fs.Int("num-prompts", 100, "number of prompts to run")
	promptLen := fs.Int("prompt-len", 128, "prompt length in tokens")
	outputLen := fs.Int("output-len", 128, "max output tokens per request")
	concurrency := fs.Int("concurrency", 1, "concurrent requests (serving mode)")
	targetURL := fs.String("url", "", "target server URL (serving mode)")
	configFile := fs.String("config", "", "path to frozen benchmark config JSON (overrides flags)")
	fs.Parse(args)

	log := newLogger("info")

	// If --config is provided, load frozen parameters from it.
	if *configFile != "" {
		data, err := os.ReadFile(*configFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading config: %v\n", err)
			os.Exit(1)
		}
		var bc benchConfig
		if err := json.Unmarshal(data, &bc); err != nil {
			fmt.Fprintf(os.Stderr, "error parsing config: %v\n", err)
			os.Exit(1)
		}
		if bc.Benchmark.Mode != "" {
			*mode = bc.Benchmark.Mode
		}
		if bc.Benchmark.NumPrompts > 0 {
			*numPrompts = bc.Benchmark.NumPrompts
		}
		if bc.Benchmark.PromptLen > 0 {
			*promptLen = bc.Benchmark.PromptLen
		}
		if bc.Benchmark.OutputLen > 0 {
			*outputLen = bc.Benchmark.OutputLen
		}
		if bc.Benchmark.Concurrency > 0 {
			*concurrency = bc.Benchmark.Concurrency
		}
		log.Info("loaded benchmark config", "path", *configFile)
	}

	switch *mode {
	case "offline":
		runOfflineBench(log, *numPrompts, *promptLen, *outputLen)
	case "serving":
		if *targetURL == "" {
			fmt.Fprintln(os.Stderr, "error: --url required for serving mode")
			os.Exit(1)
		}
		runServingBench(log, *targetURL, *numPrompts, *promptLen, *outputLen, *concurrency)
	default:
		fmt.Fprintf(os.Stderr, "unknown bench mode: %s\n", *mode)
		os.Exit(1)
	}
}

func runOfflineBench(log *slog.Logger, numPrompts, promptLen, outputLen int) {
	log.Info("offline benchmark",
		"num_prompts", numPrompts,
		"prompt_len", promptLen,
		"output_len", outputLen)

	// Create mock components for offline bench.
	runner := &backend.MockRunner{}
	tokenizer := model.NewByteLevelTokenizer(32000, 2)
	cache := kvcache.NewBlockPool(10000, 16)
	sched := scheduler.NewFCFSScheduler(scheduler.DefaultFCFSConfig())

	eng := engine.NewServingEngine(engine.ServingEngineConfig{
		Runner:    runner,
		Scheduler: sched,
		Cache:     cache,
		Tokenizer: tokenizer,
		Sampling:  engine.SamplingParams{Temperature: 0},
		Log:       log,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eng.Start(ctx)

	start := time.Now()

	// Enqueue all prompts, collecting handles. Track failures.
	type pendingReq struct {
		handle *engine.RequestHandle
	}
	var pending []pendingReq
	enqueueErrors := 0

	for i := 0; i < numPrompts; i++ {
		prompt := make([]int32, promptLen)
		for j := range prompt {
			prompt[j] = int32((j + i) % 255)
		}
		handle, err := eng.Enqueue(ctx, &engine.Request{
			ID:        fmt.Sprintf("bench-%d", i),
			TokenIDs:  prompt,
			MaxTokens: outputLen,
		})
		if err != nil {
			enqueueErrors++
			log.Warn("enqueue failed", "prompt", i, "error", err)
			continue
		}
		pending = append(pending, pendingReq{handle: handle})
	}

	if enqueueErrors > 0 {
		log.Warn("enqueue failures", "failed", enqueueErrors, "succeeded", len(pending))
	}

	// Wait for completion by draining each handle's channel.
	completed := 0
	for _, p := range pending {
		for tok := range p.handle.Tokens {
			if tok.Finished || tok.Err != nil {
				completed++
				break
			}
		}
	}

	elapsed := time.Since(start)
	totalTokens := completed * outputLen // approximate
	eng.Stop()

	// Print results.
	result := BenchResult{
		Timestamp:       time.Now().UTC().Format(time.RFC3339),
		GitSHA:          getGitSHA(),
		Hardware:        "DGX Spark GB10",
		Mode:            "offline",
		NumPrompts:      completed,
		PromptLen:       promptLen,
		OutputLen:       outputLen,
		ElapsedSeconds:  elapsed.Seconds(),
		TokensPerSecond: float64(totalTokens) / elapsed.Seconds(),
		RequestsPerSec:  float64(completed) / elapsed.Seconds(),
	}
	data, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(data))
	saveBenchResult(result, data)
}

func runServingBench(log *slog.Logger, targetURL string, numPrompts, promptLen, outputLen, concurrency int) {
	log.Info("serving benchmark",
		"url", targetURL,
		"num_prompts", numPrompts,
		"prompt_len", promptLen,
		"output_len", outputLen,
		"concurrency", concurrency)

	type requestResult struct {
		ttft   time.Duration   // time to first content token (from streaming SSE)
		e2e    time.Duration   // total request duration
		itls   []time.Duration // inter-token latencies
		tokens int
		err    bool
	}

	results := make(chan requestResult, numPrompts)
	sem := make(chan struct{}, concurrency)
	client := &http.Client{Timeout: 120 * time.Second}

	start := time.Now()

	for i := 0; i < numPrompts; i++ {
		sem <- struct{}{}
		go func(idx int) {
			defer func() { <-sem }()

			prompt := fmt.Sprintf("Prompt number %d. Please continue writing about topic %d.", idx, idx%10)

			payload := map[string]any{
				"model":      "gollmgo-default",
				"max_tokens": outputLen,
				"stream":     true,
				"messages": []map[string]string{
					{"role": "user", "content": prompt},
				},
			}
			body, _ := json.Marshal(payload)

			reqStart := time.Now()
			resp, err := client.Post(targetURL+"/v1/chat/completions", "application/json",
				io.NopCloser(io.NewSectionReader(newBytesReader(body), 0, int64(len(body)))))
			if err != nil {
				log.Error("request failed", "error", err)
				results <- requestResult{err: true}
				return
			}
			defer resp.Body.Close()

			// Parse SSE stream to measure real TTFT and per-token ITL.
			var ttft time.Duration
			var itls []time.Duration
			tokens := 0
			lastTokenTime := reqStart
			firstContent := true

			scanner := bufio.NewScanner(resp.Body)
			for scanner.Scan() {
				line := scanner.Text()
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					break
				}

				// Check if this chunk has content (skip role-only chunks).
				if !strings.Contains(data, `"content"`) {
					continue
				}

				now := time.Now()
				tokens++
				if firstContent {
					ttft = now.Sub(reqStart)
					firstContent = false
				} else {
					itls = append(itls, now.Sub(lastTokenTime))
				}
				lastTokenTime = now
			}
			e2e := time.Since(reqStart)
			results <- requestResult{ttft: ttft, e2e: e2e, itls: itls, tokens: tokens}
		}(i)
	}

	// Wait for all goroutines.
	for i := 0; i < concurrency; i++ {
		sem <- struct{}{}
	}
	close(results)

	elapsed := time.Since(start)

	// Collect and compute stats.
	var ttfts, e2es, allITLs []float64
	totalTokens := 0
	errorCount := 0
	for r := range results {
		if r.err {
			errorCount++
			continue
		}
		ttfts = append(ttfts, float64(r.ttft.Microseconds())/1000.0)
		e2es = append(e2es, float64(r.e2e.Microseconds())/1000.0)
		totalTokens += r.tokens
		for _, itl := range r.itls {
			allITLs = append(allITLs, float64(itl.Microseconds())/1000.0)
		}
	}

	count := len(ttfts)
	if count == 0 {
		fmt.Fprintln(os.Stderr, "no successful requests")
		return
	}

	sort.Float64s(ttfts)
	sort.Float64s(e2es)
	sort.Float64s(allITLs)

	gitSHA := getGitSHA()

	result := BenchResult{
		Timestamp:       time.Now().UTC().Format(time.RFC3339),
		GitSHA:          gitSHA,
		Hardware:        "DGX Spark GB10",
		Mode:            "serving",
		NumPrompts:      numPrompts,
		PromptLen:       promptLen,
		OutputLen:       outputLen,
		Concurrency:     concurrency,
		ElapsedSeconds:  elapsed.Seconds(),
		TokensPerSecond: float64(totalTokens) / elapsed.Seconds(),
		RequestsPerSec:  float64(count) / elapsed.Seconds(),
		ErrorCount:      errorCount,
		TTFTP50Ms:       percentile(ttfts, 0.50),
		TTFTP95Ms:       percentile(ttfts, 0.95),
		TTFTP99Ms:       percentile(ttfts, 0.99),
		E2EP50Ms:        percentile(e2es, 0.50),
		E2EP95Ms:        percentile(e2es, 0.95),
		E2EP99Ms:        percentile(e2es, 0.99),
		ITLP50Ms:        percentile(allITLs, 0.50),
		ITLP95Ms:        percentile(allITLs, 0.95),
		ITLP99Ms:        percentile(allITLs, 0.99),
	}
	data, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(data))

	// Save to bench/results/.
	saveBenchResult(result, data)
}

// BenchResult holds benchmark output per docs/benchmarks.md.
type BenchResult struct {
	Timestamp       string  `json:"timestamp"`
	GitSHA          string  `json:"git_sha"`
	Hardware        string  `json:"hardware"`
	Mode            string  `json:"mode"`
	NumPrompts      int     `json:"num_prompts"`
	PromptLen       int     `json:"prompt_len"`
	OutputLen       int     `json:"output_len"`
	Concurrency     int     `json:"concurrency,omitempty"`
	ElapsedSeconds  float64 `json:"elapsed_seconds"`
	TokensPerSecond float64 `json:"tokens_per_second"`
	RequestsPerSec  float64 `json:"requests_per_second"`
	ErrorCount      int     `json:"error_count"`
	TTFTP50Ms       float64 `json:"ttft_p50_ms"`
	TTFTP95Ms       float64 `json:"ttft_p95_ms"`
	TTFTP99Ms       float64 `json:"ttft_p99_ms"`
	E2EP50Ms        float64 `json:"e2e_p50_ms"`
	E2EP95Ms        float64 `json:"e2e_p95_ms"`
	E2EP99Ms        float64 `json:"e2e_p99_ms"`
	ITLP50Ms        float64 `json:"itl_p50_ms"`
	ITLP95Ms        float64 `json:"itl_p95_ms"`
	ITLP99Ms        float64 `json:"itl_p99_ms"`
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)-1) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func getGitSHA() string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func saveBenchResult(result BenchResult, data []byte) {
	dir := "bench/results"
	os.MkdirAll(dir, 0755)
	filename := fmt.Sprintf("%s/%s_%s_%s.json",
		dir, result.Mode, result.Timestamp[:10],
		strings.ReplaceAll(result.Timestamp[11:19], ":", ""))
	os.WriteFile(filename, data, 0644)
}

func schedulerPolicyFromString(s string) scheduler.Policy {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "sjf":
		return scheduler.PolicySJF
	case "priority":
		return scheduler.PolicyPriority
	default:
		return scheduler.PolicyFCFS
	}
}

func preemptModeFromString(s string) scheduler.PreemptMode {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "swap":
		return scheduler.PreemptSwap
	default:
		return scheduler.PreemptRecompute
	}
}

func cmdDoctor(_ []string) {
	fmt.Println("gollmgo doctor")
	fmt.Println("  Go version: runtime check")
	fmt.Println("  Config: valid defaults")
	fmt.Println("  GPU: check via build tag")
	fmt.Println("doctor: all checks passed (stub)")
}

func newLogger(level string) *slog.Logger {
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl}))
}

// newBytesReader is a helper to avoid importing bytes in the cgo-tagged build.
type bytesReader struct {
	data []byte
	pos  int
}

func newBytesReader(data []byte) *bytesReader { return &bytesReader{data: data} }

func (r *bytesReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

func (r *bytesReader) ReadAt(p []byte, off int64) (int, error) {
	if int(off) >= len(r.data) {
		return 0, io.EOF
	}
	n := copy(p, r.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}
