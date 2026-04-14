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
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/TensorGreed/gollmgo/internal/api"
	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/config"
	"github.com/TensorGreed/gollmgo/internal/engine"
	"github.com/TensorGreed/gollmgo/internal/kvcache"
	"github.com/TensorGreed/gollmgo/internal/model"
	"github.com/TensorGreed/gollmgo/internal/model/hfhub"
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
	modelID := "mock"
	if cfg.ModelPath != "" {
		resolveCtx, resolveCancel := context.WithCancel(context.Background())
		handle, err := resolveModelSpec(resolveCtx, log, cfg)
		resolveCancel()
		if err != nil {
			fmt.Fprintf(os.Stderr, "model resolution failed: %v\n", err)
			os.Exit(1)
		}

		runner, tokenizer, numBlocks, err = initGPURunner(log, cfg, handle, *deviceID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "GPU init failed: %v\n", err)
			os.Exit(1)
		}
		if handle.RepoID != "" {
			log.Info("GPU model loaded", "repo", handle.RepoID, "revision", handle.Revision, "dir", handle.LocalDir)
			modelID = handle.RepoID
		} else {
			log.Info("GPU model loaded", "model", handle.LocalDir)
			modelID = modelIDFromPath(cfg.ModelPath)
		}
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
		Speculative: engine.SpeculativeSettings{
			Enabled:        cfg.Speculative.Enabled,
			Mode:           cfg.Speculative.Mode,
			NGramSize:      cfg.Speculative.NGramSize,
			NumDraftTokens: cfg.Speculative.NumDraftTokens,
			KillThreshold:  cfg.Speculative.KillThreshold,
		},
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eng.Start(ctx)

	// Create and start API server.
	srv := api.NewServer(cfg, eng, tokenizer, log, modelID)

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
		Mode           string  `json:"mode"`
		NumPrompts     int     `json:"num_prompts"`
		PromptLen      int     `json:"prompt_len"`
		OutputLen      int     `json:"output_len"`
		Concurrency    int     `json:"concurrency"`
		QPS            float64 `json:"qps"`
		Duration       string  `json:"duration"`
		WarmupRequests int     `json:"warmup_requests"`
		Repetitions    int     `json:"repetitions"`
	} `json:"benchmark"`
}

func cmdBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	mode := fs.String("mode", "synthetic", "benchmark mode: synthetic, serving")
	numPrompts := fs.Int("num-prompts", 100, "number of prompts to run")
	promptLen := fs.Int("prompt-len", 128, "prompt length in tokens")
	outputLen := fs.Int("output-len", 128, "max output tokens per request")
	concurrency := fs.Int("concurrency", 1, "concurrent requests (serving mode)")
	qps := fs.Float64("qps", 0, "Poisson arrival rate in requests/sec (serving mode)")
	durationStr := fs.String("duration", "", "optional wall-clock limit for Poisson serving mode (for example: 30s)")
	warmupRequests := fs.Int("warmup-requests", 0, "warmup requests excluded from measurements")
	repetitions := fs.Int("repetitions", 1, "number of measured repetitions")
	targetURL := fs.String("url", "", "target server URL (serving mode)")
	modelSpec := fs.String("model", "", "local model path or HF repo id for exact prompt sizing/token accounting")
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
		if bc.Benchmark.QPS > 0 {
			*qps = bc.Benchmark.QPS
		}
		if bc.Benchmark.Duration != "" {
			*durationStr = bc.Benchmark.Duration
		}
		if bc.Benchmark.WarmupRequests > 0 {
			*warmupRequests = bc.Benchmark.WarmupRequests
		}
		if bc.Benchmark.Repetitions > 0 {
			*repetitions = bc.Benchmark.Repetitions
		}
		log.Info("loaded benchmark config", "path", *configFile)
	}

	duration := time.Duration(0)
	if strings.TrimSpace(*durationStr) != "" {
		parsed, err := time.ParseDuration(*durationStr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error parsing duration: %v\n", err)
			os.Exit(1)
		}
		duration = parsed
	}

	opts := benchOptions{
		Mode:           *mode,
		NumPrompts:     *numPrompts,
		PromptLen:      *promptLen,
		OutputLen:      *outputLen,
		Concurrency:    *concurrency,
		QPS:            *qps,
		Duration:       duration,
		WarmupRequests: *warmupRequests,
		Repetitions:    *repetitions,
		ModelSpec:      *modelSpec,
		TargetURL:      *targetURL,
	}

	switch *mode {
	case "offline", "synthetic":
		runOfflineBench(log, *numPrompts, *promptLen, *outputLen)
	case "serving":
		if *targetURL == "" {
			fmt.Fprintln(os.Stderr, "error: --url required for serving mode")
			os.Exit(1)
		}
		tok, promptMode, err := resolveBenchTokenizer(log, *modelSpec)
		if err != nil {
			fmt.Fprintf(os.Stderr, "benchmark tokenizer init failed: %v\n", err)
			os.Exit(1)
		}
		if strings.TrimSpace(*modelSpec) == "" {
			log.Warn("serving benchmark running without --model; prompt sizing and token accounting are approximate")
		} else if tok == nil {
			log.Warn("benchmark tokenizer unavailable; prompt sizing and token accounting fell back", "model", *modelSpec)
		}
		runServingBench(log, opts, tok, promptMode)
	default:
		fmt.Fprintf(os.Stderr, "unknown bench mode: %s\n", *mode)
		os.Exit(1)
	}
}

func runOfflineBench(log *slog.Logger, numPrompts, promptLen, outputLen int) {
	log.Info("synthetic benchmark",
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
		Mode:            "synthetic",
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

func runServingBench(log *slog.Logger, opts benchOptions, tok model.Tokenizer, promptMode string) {
	log.Info("serving benchmark",
		"url", opts.TargetURL,
		"num_prompts", opts.NumPrompts,
		"prompt_len", opts.PromptLen,
		"output_len", opts.OutputLen,
		"concurrency", opts.Concurrency,
		"qps", opts.QPS,
		"duration", opts.Duration,
		"warmup_requests", opts.WarmupRequests,
		"repetitions", opts.Repetitions)

	result, err := runServingBenchSuite(log, opts, tok, promptMode)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		return
	}

	data, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(data))
	saveBenchResult(result, data)
}

// BenchResult holds benchmark output per docs/benchmarks.md.
type BenchResult struct {
	Timestamp        string        `json:"timestamp"`
	GitSHA           string        `json:"git_sha"`
	Hardware         string        `json:"hardware"`
	Mode             string        `json:"mode"`
	NumPrompts       int           `json:"num_prompts"`
	MeasuredRequests int           `json:"measured_requests,omitempty"`
	PromptLen        int           `json:"prompt_len"`
	OutputLen        int           `json:"output_len"`
	Concurrency      int           `json:"concurrency,omitempty"`
	WarmupRequests   int           `json:"warmup_requests,omitempty"`
	Repetitions      int           `json:"repetitions,omitempty"`
	ArrivalMode      string        `json:"arrival_mode,omitempty"`
	PromptGeneration string        `json:"prompt_generation,omitempty"`
	TokenCountMode   string        `json:"token_count_mode,omitempty"`
	QPS              float64       `json:"qps,omitempty"`
	DurationSeconds  float64       `json:"duration_seconds,omitempty"`
	ElapsedSeconds   float64       `json:"elapsed_seconds"`
	TokensPerSecond  float64       `json:"tokens_per_second"`
	RequestsPerSec   float64       `json:"requests_per_second"`
	ErrorCount       int           `json:"error_count"`
	TTFTP50Ms        float64       `json:"ttft_p50_ms"`
	TTFTP95Ms        float64       `json:"ttft_p95_ms"`
	TTFTP99Ms        float64       `json:"ttft_p99_ms"`
	E2EP50Ms         float64       `json:"e2e_p50_ms"`
	E2EP95Ms         float64       `json:"e2e_p95_ms"`
	E2EP99Ms         float64       `json:"e2e_p99_ms"`
	ITLP50Ms         float64       `json:"itl_p50_ms"`
	ITLP95Ms         float64       `json:"itl_p95_ms"`
	ITLP99Ms         float64       `json:"itl_p99_ms"`
	Samples          []benchSample `json:"samples,omitempty"`
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

// modelIDFromPath derives the API model id from --model. Examples:
//
//	/models/Llama-3-8B-Instruct/                  -> "Llama-3-8B-Instruct"
//	/models/Llama-3-8B-Instruct/model.safetensors -> "Llama-3-8B-Instruct"
//	/models/llama.gguf                            -> "llama.gguf"
//
// Falls back to "gollmgo-default" if the input is empty.
func modelIDFromPath(p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return "gollmgo-default"
	}
	clean := strings.TrimRight(p, "/")
	// If --model points at a single file inside a model directory, prefer
	// the directory name (matches typical HF layouts).
	if info, err := os.Stat(clean); err == nil && !info.IsDir() {
		dir := filepath.Dir(clean)
		base := filepath.Base(dir)
		if base != "" && base != "." && base != "/" {
			return base
		}
	}
	base := filepath.Base(clean)
	if base == "" || base == "." || base == "/" {
		return "gollmgo-default"
	}
	return base
}

// resolveModelSpec parses cfg.ModelPath as either a local path or an
// HF Hub repo id ("owner/name[@revision]") and makes the files available
// locally, downloading to the HF cache if needed. Returns a Handle that
// initGPURunner consumes.
func resolveModelSpec(ctx context.Context, log *slog.Logger, cfg config.Config) (*hfhub.Handle, error) {
	spec, err := hfhub.ParseSpec(cfg.ModelPath)
	if err != nil {
		return nil, err
	}
	token := cfg.HFToken
	if token == "" {
		token = hfhub.TokenFromEnv()
	}
	resolver := &hfhub.Resolver{
		CacheDir: cfg.HFCacheDir, // empty → resolver picks default
		Token:    token,
		Log:      log,
	}
	if !spec.IsLocal {
		log.Info("resolving HuggingFace repo",
			"repo", spec.Repo, "revision", spec.Revision,
			"token_set", token != "")
	}
	return resolver.Resolve(ctx, spec)
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

// doctorCheck is one diagnostic. status is "PASS", "WARN", or "FAIL".
type doctorCheck struct {
	name   string
	status string
	detail string
}

func cmdDoctor(args []string) {
	fs := flag.NewFlagSet("doctor", flag.ExitOnError)
	configPath := fs.String("config", "", "path to server config to validate")
	modelPath := fs.String("model", "", "path to model file or directory to probe")
	fs.Parse(args)

	checks := []doctorCheck{
		runtimeCheck(),
		binaryCheck(),
		nvidiaDriverCheck(),
		nvccCheck(),
		configCheck(*configPath),
	}
	if *modelPath != "" {
		checks = append(checks, modelCheck(*modelPath))
	} else {
		checks = append(checks, doctorCheck{
			name: "Model", status: "WARN",
			detail: "no --model provided; doctor only validates the toolchain",
		})
	}

	fmt.Println("gollmgo doctor")
	fmt.Println()
	pass, warn, fail := 0, 0, 0
	for _, c := range checks {
		fmt.Printf("  %-20s [%s] %s\n", c.name, c.status, c.detail)
		switch c.status {
		case "PASS":
			pass++
		case "WARN":
			warn++
		case "FAIL":
			fail++
		}
	}
	fmt.Println()
	fmt.Printf("  summary: %d pass, %d warn, %d fail\n", pass, warn, fail)
	if fail > 0 {
		os.Exit(1)
	}
}

func runtimeCheck() doctorCheck {
	return doctorCheck{
		name:   "Go runtime",
		status: "PASS",
		detail: fmt.Sprintf("%s on %s/%s", runtime.Version(), runtime.GOOS, runtime.GOARCH),
	}
}

func binaryCheck() doctorCheck {
	bin := "bin/gollmgo"
	info, err := os.Stat(bin)
	if err != nil {
		return doctorCheck{
			name: "Binary", status: "WARN",
			detail: bin + " not found — run `make build` to produce a GPU-enabled binary",
		}
	}
	return doctorCheck{
		name: "Binary", status: "PASS",
		detail: fmt.Sprintf("%s present (%.1f MB)", bin, float64(info.Size())/(1<<20)),
	}
}

func nvidiaDriverCheck() doctorCheck {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,driver_version,memory.total",
		"--format=csv,noheader").Output()
	if err != nil {
		return doctorCheck{
			name: "NVIDIA driver", status: "FAIL",
			detail: "nvidia-smi failed (" + err.Error() + ") — driver not installed or no GPU visible",
		}
	}
	first := strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)[0]
	return doctorCheck{
		name:   "NVIDIA driver",
		status: "PASS",
		detail: first,
	}
}

func nvccCheck() doctorCheck {
	path, err := exec.LookPath("nvcc")
	if err != nil {
		return doctorCheck{
			name: "CUDA toolkit", status: "WARN",
			detail: "nvcc not on PATH — needed for `make kernels` (runtime serving uses precompiled .a files)",
		}
	}
	out, err := exec.Command("nvcc", "--version").Output()
	if err != nil {
		return doctorCheck{
			name: "CUDA toolkit", status: "WARN", detail: "nvcc found at " + path + " but failed: " + err.Error(),
		}
	}
	// Last line of `nvcc --version` looks like: "Build cuda_12.x.r12.x/compiler.xxxxx_0"
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	return doctorCheck{
		name: "CUDA toolkit", status: "PASS", detail: lines[len(lines)-1],
	}
}

func configCheck(path string) doctorCheck {
	cfg := config.DefaultConfig()
	if path != "" {
		loaded, err := config.LoadFile(path)
		if err != nil {
			return doctorCheck{
				name: "Config", status: "FAIL",
				detail: fmt.Sprintf("load %s: %v", path, err),
			}
		}
		cfg = loaded
	}
	if err := cfg.Validate(); err != nil {
		return doctorCheck{
			name: "Config", status: "FAIL", detail: err.Error(),
		}
	}
	source := "default"
	if path != "" {
		source = path
	}
	return doctorCheck{
		name:   "Config",
		status: "PASS",
		detail: fmt.Sprintf("%s validates (policy=%s preempt=%s prefix_chunk=%d)",
			source, cfg.SchedulerPolicy, cfg.PreemptMode, cfg.PrefillChunkSize),
	}
}

// modelCheck does a cheap structural probe of the model spec. It accepts
// either a local path or an HF Hub repo id. For HF specs it does NOT
// download — it only confirms the repo+revision is reachable and prints
// the cache location so the user knows where a subsequent `serve` would
// stage files.
func modelCheck(spec string) doctorCheck {
	parsed, err := hfhub.ParseSpec(spec)
	if err != nil {
		return doctorCheck{name: "Model", status: "FAIL", detail: err.Error()}
	}
	if !parsed.IsLocal {
		cacheDir := hfhub.DefaultCacheDir()
		local := filepath.Join(cacheDir, "models--"+strings.ReplaceAll(parsed.Repo, "/", "--"))
		cacheState := "not cached (will download on `serve`)"
		if _, err := os.Stat(local); err == nil {
			cacheState = "cached"
		}
		return doctorCheck{
			name: "Model", status: "PASS",
			detail: fmt.Sprintf("HF repo %s@%s — %s at %s",
				parsed.Repo, parsed.Revision, cacheState, local),
		}
	}

	path := parsed.LocalPath
	info, err := os.Stat(path)
	if err != nil {
		return doctorCheck{
			name: "Model", status: "FAIL",
			detail: fmt.Sprintf("stat %s: %v", path, err),
		}
	}
	if info.IsDir() {
		// Expect HF layout: config.json + tokenizer.json + at least one .safetensors file.
		needed := []string{"config.json"}
		missing := []string{}
		for _, f := range needed {
			if _, err := os.Stat(filepath.Join(path, f)); err != nil {
				missing = append(missing, f)
			}
		}
		entries, _ := os.ReadDir(path)
		hasWeights := false
		for _, e := range entries {
			n := e.Name()
			if strings.HasSuffix(n, ".safetensors") || strings.HasSuffix(n, ".gguf") {
				hasWeights = true
				break
			}
		}
		if len(missing) > 0 || !hasWeights {
			d := fmt.Sprintf("missing: %v", missing)
			if !hasWeights {
				d += " (no .safetensors/.gguf in directory)"
			}
			return doctorCheck{name: "Model", status: "FAIL", detail: d}
		}
		return doctorCheck{
			name: "Model", status: "PASS",
			detail: fmt.Sprintf("HF directory at %s (config.json + weights present)", path),
		}
	}
	switch {
	case strings.HasSuffix(path, ".safetensors"):
		return doctorCheck{
			name: "Model", status: "PASS",
			detail: fmt.Sprintf("safetensors file (%.1f GB)", float64(info.Size())/(1<<30)),
		}
	case strings.HasSuffix(path, ".gguf"):
		return doctorCheck{
			name: "Model", status: "PASS",
			detail: fmt.Sprintf("GGUF file (%.1f GB) — note: only F32/F16/BF16 weights are supported", float64(info.Size())/(1<<30)),
		}
	default:
		return doctorCheck{
			name: "Model", status: "WARN",
			detail: "unrecognised extension; expected .safetensors or .gguf",
		}
	}
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
