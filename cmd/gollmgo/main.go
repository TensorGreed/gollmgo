// Package main is the CLI entrypoint for gollmgo.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
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

	if cfg.ModelPath != "" {
		var err error
		runner, tokenizer, err = initGPURunner(log, cfg.ModelPath, cfg.TokenizerPath, *deviceID)
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
		"block_size", cfg.BlockSize)

	// Create KV cache pool.
	numBlocks := 1024 // TODO: compute from available GPU memory
	cache := kvcache.NewBlockPool(numBlocks, cfg.BlockSize)

	// Create scheduler.
	sched := scheduler.NewFCFSScheduler(scheduler.FCFSConfig{
		MaxBatchSize:   cfg.MaxBatchSize,
		MaxTokenBudget: cfg.MaxTokenBudget,
		MaxQueueDepth:  cfg.MaxQueueDepth,
	})

	// Create serving engine.
	eng := engine.NewServingEngine(engine.ServingEngineConfig{
		Runner:    runner,
		Scheduler: sched,
		Cache:     cache,
		Tokenizer: tokenizer,
		Sampling:  engine.DefaultSamplingParams(),
		Log:       log,
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
	log.Info("shutdown complete")
}

func cmdBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	mode := fs.String("mode", "offline", "benchmark mode: offline, serving")
	numPrompts := fs.Int("num-prompts", 100, "number of prompts to run")
	promptLen := fs.Int("prompt-len", 128, "prompt length in tokens")
	outputLen := fs.Int("output-len", 128, "max output tokens per request")
	concurrency := fs.Int("concurrency", 1, "concurrent requests (serving mode)")
	targetURL := fs.String("url", "", "target server URL (serving mode)")
	fs.Parse(args)

	log := newLogger("info")

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
}

func runServingBench(log *slog.Logger, targetURL string, numPrompts, promptLen, outputLen, concurrency int) {
	log.Info("serving benchmark",
		"url", targetURL,
		"num_prompts", numPrompts,
		"prompt_len", promptLen,
		"output_len", outputLen,
		"concurrency", concurrency)

	type requestResult struct {
		ttft    time.Duration
		total   time.Duration
		tokens  int
	}

	results := make(chan requestResult, numPrompts)
	sem := make(chan struct{}, concurrency)
	client := &http.Client{Timeout: 120 * time.Second}

	start := time.Now()

	for i := 0; i < numPrompts; i++ {
		sem <- struct{}{}
		go func(idx int) {
			defer func() { <-sem }()

			// Build a simple prompt.
			prompt := fmt.Sprintf("Prompt number %d. Please continue writing about topic %d.", idx, idx%10)

			payload := map[string]any{
				"model":      "gollmgo-default",
				"max_tokens": outputLen,
				"stream":     false,
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
				return
			}
			ttft := time.Since(reqStart)
			io.ReadAll(resp.Body)
			resp.Body.Close()
			total := time.Since(reqStart)

			results <- requestResult{ttft: ttft, total: total, tokens: outputLen}
		}(i)
	}

	// Wait for all.
	for i := 0; i < concurrency; i++ {
		sem <- struct{}{}
	}
	close(results)

	elapsed := time.Since(start)

	var totalTTFT, totalDuration time.Duration
	var count int
	for r := range results {
		totalTTFT += r.ttft
		totalDuration += r.total
		count++
	}

	if count == 0 {
		fmt.Fprintln(os.Stderr, "no successful requests")
		return
	}

	result := BenchResult{
		Mode:            "serving",
		NumPrompts:      numPrompts,
		PromptLen:       promptLen,
		OutputLen:       outputLen,
		Concurrency:     concurrency,
		ElapsedSeconds:  elapsed.Seconds(),
		TokensPerSecond: float64(count*outputLen) / elapsed.Seconds(),
		RequestsPerSec:  float64(count) / elapsed.Seconds(),
		AvgTTFTMs:       float64(totalTTFT.Milliseconds()) / float64(count),
		AvgLatencyMs:    float64(totalDuration.Milliseconds()) / float64(count),
	}
	data, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(data))
}

// BenchResult holds benchmark output.
type BenchResult struct {
	Mode            string  `json:"mode"`
	NumPrompts      int     `json:"num_prompts"`
	PromptLen       int     `json:"prompt_len"`
	OutputLen       int     `json:"output_len"`
	Concurrency     int     `json:"concurrency,omitempty"`
	ElapsedSeconds  float64 `json:"elapsed_seconds"`
	TokensPerSecond float64 `json:"tokens_per_second"`
	RequestsPerSec  float64 `json:"requests_per_second"`
	AvgTTFTMs       float64 `json:"avg_ttft_ms,omitempty"`
	AvgLatencyMs    float64 `json:"avg_latency_ms,omitempty"`
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
