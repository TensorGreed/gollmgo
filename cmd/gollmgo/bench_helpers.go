package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/TensorGreed/gollmgo/internal/config"
	"github.com/TensorGreed/gollmgo/internal/model"
)

type benchOptions struct {
	Mode           string
	NumPrompts     int
	PromptLen      int
	OutputLen      int
	Concurrency    int
	QPS            float64
	Duration       time.Duration
	WarmupRequests int
	Repetitions    int
	ModelSpec      string
	TargetURL      string
}

type benchSample struct {
	Index              int     `json:"index"`
	MeasuredRequests   int     `json:"measured_requests"`
	SuccessfulRequests int     `json:"successful_requests"`
	ErrorCount         int     `json:"error_count"`
	ElapsedSeconds     float64 `json:"elapsed_seconds"`
	TokensPerSecond    float64 `json:"tokens_per_second"`
	RequestsPerSec     float64 `json:"requests_per_second"`
	TTFTP50Ms          float64 `json:"ttft_p50_ms"`
	TTFTP95Ms          float64 `json:"ttft_p95_ms"`
	TTFTP99Ms          float64 `json:"ttft_p99_ms"`
	E2EP50Ms           float64 `json:"e2e_p50_ms"`
	E2EP95Ms           float64 `json:"e2e_p95_ms"`
	E2EP99Ms           float64 `json:"e2e_p99_ms"`
	ITLP50Ms           float64 `json:"itl_p50_ms"`
	ITLP95Ms           float64 `json:"itl_p95_ms"`
	ITLP99Ms           float64 `json:"itl_p99_ms"`
}

type benchRequestResult struct {
	ttft   time.Duration
	e2e    time.Duration
	itls   []time.Duration
	tokens int
	err    bool
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

var benchGrowthFragments = []string{
	" alpha beta gamma delta epsilon zeta eta theta",
	" request context about batching latency throughput and scheduling",
	" the system should continue the explanation with concrete details",
	" layer cache attention prefill decode queue fairness memory budget",
	" alpha beta gamma delta epsilon",
	" latency throughput fairness batching",
	" deterministic output for benchmarking",
	" alpha beta gamma",
	" token token token",
	" repeat repeat repeat",
}

var benchExactFragments = []string{
	" alpha", " beta", " gamma", " delta", " epsilon",
	" zeta", " eta", " theta", " iota", " kappa",
	" the", " and", " or", " with", " for",
	" x", " y", " z", " a", " b", " c",
	" 0", " 1", " 2", " 3", " 4", " 5",
	".", ",", "!", "?", ";", ":", "\n", " -",
}

func (o benchOptions) arrivalMode() string {
	if o.QPS > 0 {
		return "poisson_qps"
	}
	return "closed_loop"
}

func resolveBenchTokenizer(log *slog.Logger, modelSpec string) (model.Tokenizer, string, error) {
	modelSpec = strings.TrimSpace(modelSpec)
	if modelSpec == "" {
		return nil, "approximate", nil
	}

	cfg := config.DefaultConfig()
	cfg.ModelPath = modelSpec

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	handle, err := resolveModelSpec(ctx, log, cfg)
	if err != nil {
		return nil, "", err
	}
	if handle.TokenizerPath == "" {
		return nil, "approximate", nil
	}

	tok, err := model.LoadHFTokenizer(handle.TokenizerPath, "</s>", "<s>")
	if err != nil {
		return nil, "", err
	}
	return tok, "tokenizer_exact", nil
}

func buildBenchPrompts(tok model.Tokenizer, count, promptLen, offset int) ([]string, string, error) {
	if count <= 0 {
		return nil, "approximate", nil
	}
	prompts := make([]string, 0, count)
	mode := "approximate"
	for i := 0; i < count; i++ {
		prompt, exactMode, err := buildPromptText(tok, promptLen, offset+i)
		if err != nil {
			return nil, "", err
		}
		if exactMode == "tokenizer_exact" {
			mode = exactMode
		}
		prompts = append(prompts, prompt)
	}
	return prompts, mode, nil
}

func buildPromptText(tok model.Tokenizer, targetTokens, idx int) (string, string, error) {
	if targetTokens <= 0 {
		return "", "approximate", nil
	}
	if tok == nil {
		return buildApproxPrompt(targetTokens, idx), "approximate", nil
	}

	words := buildPromptCorpusWords(idx)
	lo, hi := 1, len(words)
	bestText := ""
	bestCount := 0
	for lo <= hi {
		mid := (lo + hi) / 2
		text := strings.Join(words[:mid], " ")
		count, err := promptTokenCount(tok, text)
		if err != nil {
			return "", "", err
		}
		if count <= targetTokens {
			bestText = text
			bestCount = count
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	if bestText == "" {
		return "", "", fmt.Errorf("benchmark: could not fit corpus into %d tokens", targetTokens)
	}
	if bestCount == targetTokens {
		return bestText, "tokenizer_exact", nil
	}

	suffix, ok, err := exactPromptSuffix(tok, bestText, bestCount, targetTokens, 0, map[string]bool{})
	if err != nil {
		return "", "", err
	}
	if ok {
		return bestText + suffix, "tokenizer_exact", nil
	}

	return "", "", fmt.Errorf("benchmark: could not build exact %d-token prompt", targetTokens)
}

func buildPromptCorpusWords(idx int) []string {
	base := []string{
		"benchmark", "request", fmt.Sprintf("%04d", idx),
		"studies", "latency", "throughput", "batching", "queue",
		"fairness", "cache", "behavior", "prompt", "reuse", "prefill",
		"decode", "memory", "pressure", "scheduling", "decisions",
		"deterministic", "serving", "gpu", "kernels", "tokenizer",
		"metrics", "tail", "latency", "throughput", "prefill", "decode",
		"queue", "budget", "block", "reuse", "stability", "correctness",
	}
	words := make([]string, 0, len(base)*8)
	for i := 0; i < 8; i++ {
		words = append(words, base...)
	}
	return words
}

func buildApproxPrompt(targetTokens, idx int) string {
	base := fmt.Sprintf("Benchmark request %d", idx)
	var b strings.Builder
	b.WriteString(base)
	for b.Len() < targetTokens*8 {
		b.WriteString(" alpha beta gamma delta")
	}
	return b.String()
}

func bestPromptGrowth(tok model.Tokenizer, text string, count, target int) (string, int, error) {
	bestText := text
	bestCount := count
	for _, frag := range benchGrowthFragments {
		nextText := text + frag
		nextCount, err := promptTokenCount(tok, nextText)
		if err != nil {
			return "", 0, err
		}
		if nextCount > target || nextCount <= bestCount {
			continue
		}
		bestText = nextText
		bestCount = nextCount
	}
	return bestText, bestCount, nil
}

func exactPromptSuffix(tok model.Tokenizer, prefix string, prefixCount, targetCount, depth int, seen map[string]bool) (string, bool, error) {
	if prefixCount == targetCount {
		return "", true, nil
	}
	if prefixCount > targetCount || depth >= 6 {
		return "", false, nil
	}

	key := fmt.Sprintf("%d|%d|%s", depth, prefixCount, tailKey(prefix))
	if seen[key] {
		return "", false, nil
	}
	seen[key] = true

	for _, frag := range benchExactFragments {
		nextPrefix := prefix + frag
		nextCount, err := promptTokenCount(tok, nextPrefix)
		if err != nil {
			return "", false, err
		}
		if nextCount > targetCount || nextCount <= prefixCount {
			continue
		}
		suffix, ok, err := exactPromptSuffix(tok, nextPrefix, nextCount, targetCount, depth+1, seen)
		if err != nil {
			return "", false, err
		}
		if ok {
			return frag + suffix, true, nil
		}
	}

	return "", false, nil
}

func tailKey(s string) string {
	if len(s) <= 24 {
		return s
	}
	return s[len(s)-24:]
}

func promptTokenCount(tok model.Tokenizer, text string) (int, error) {
	ids, err := tok.Encode(text)
	if err != nil {
		return 0, err
	}
	return len(ids), nil
}

func completionTokenCount(tok model.Tokenizer, text string) (int, error) {
	ids, err := tok.Encode(text)
	if err != nil {
		return 0, err
	}
	if bos := tok.BOSTokenID(); bos >= 0 && len(ids) > 0 && ids[0] == bos {
		return len(ids) - 1, nil
	}
	return len(ids), nil
}

func runServingBenchSuite(log *slog.Logger, opts benchOptions, tok model.Tokenizer, promptMode string) (BenchResult, error) {
	repetitions := opts.Repetitions
	if repetitions <= 0 {
		repetitions = 1
	}

	warmupRequests := opts.WarmupRequests
	client := &http.Client{Timeout: 120 * time.Second}
	samples := make([]benchSample, 0, repetitions)
	measuredRequests := 0
	errorCount := 0

	for rep := 0; rep < repetitions; rep++ {
		if warmupRequests > 0 {
			warmupPrompts, _, err := buildBenchPrompts(tok, warmupRequests, opts.PromptLen, -(rep+1)*1000000)
			if err != nil {
				return BenchResult{}, err
			}
			if err := runWarmupRequests(client, opts.TargetURL, warmupPrompts, opts.OutputLen, tok); err != nil {
				return BenchResult{}, err
			}
		}

		prompts, _, err := buildBenchPrompts(tok, opts.NumPrompts, opts.PromptLen, rep*opts.NumPrompts)
		if err != nil {
			return BenchResult{}, err
		}
		sample, err := runServingBenchSample(client, opts, prompts, tok, rep)
		if err != nil {
			return BenchResult{}, err
		}
		log.Info("serving benchmark repetition complete",
			"index", rep+1,
			"repetitions", repetitions,
			"tokens_per_second", sample.TokensPerSecond,
			"requests_per_second", sample.RequestsPerSec,
			"ttft_p50_ms", sample.TTFTP50Ms,
			"itl_p50_ms", sample.ITLP50Ms)
		samples = append(samples, sample)
		measuredRequests += sample.MeasuredRequests
		errorCount += sample.ErrorCount
	}

	return aggregateServingSamples(opts, samples, promptMode, measuredRequests, errorCount), nil
}

func runWarmupRequests(client *http.Client, targetURL string, prompts []string, outputLen int, tok model.Tokenizer) error {
	for _, prompt := range prompts {
		if _, err := doServingBenchRequest(client, targetURL, prompt, outputLen, tok); err != nil {
			return err
		}
	}
	return nil
}

func runServingBenchSample(client *http.Client, opts benchOptions, prompts []string, tok model.Tokenizer, rep int) (benchSample, error) {
	if len(prompts) == 0 {
		return benchSample{}, fmt.Errorf("benchmark: no prompts to run")
	}

	results := make(chan benchRequestResult, len(prompts))
	var wg sync.WaitGroup

	semSize := opts.Concurrency
	if semSize <= 0 {
		semSize = 1
	}
	sem := make(chan struct{}, semSize)

	launch := func(prompt string) {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()

			result, err := doServingBenchRequest(client, opts.TargetURL, prompt, opts.OutputLen, tok)
			if err != nil {
				results <- benchRequestResult{err: true}
				return
			}
			results <- result
		}()
	}

	measured := 0
	start := time.Now()

	if opts.QPS > 0 {
		rng := rand.New(rand.NewSource(int64(rep + 1)))
		next := start
		deadline := time.Time{}
		if opts.Duration > 0 {
			deadline = start.Add(opts.Duration)
		}
		for _, prompt := range prompts {
			if !deadline.IsZero() && next.After(deadline) {
				break
			}
			if sleep := time.Until(next); sleep > 0 {
				time.Sleep(sleep)
			}
			launch(prompt)
			measured++
			next = next.Add(samplePoissonInterval(rng, opts.QPS))
		}
	} else {
		for _, prompt := range prompts {
			launch(prompt)
			measured++
		}
	}

	wg.Wait()
	close(results)

	elapsed := time.Since(start)
	var ttfts, e2es, allITLs []float64
	totalTokens := 0
	errorCount := 0
	successes := 0

	for r := range results {
		if r.err {
			errorCount++
			continue
		}
		successes++
		ttfts = append(ttfts, float64(r.ttft.Microseconds())/1000.0)
		e2es = append(e2es, float64(r.e2e.Microseconds())/1000.0)
		totalTokens += r.tokens
		for _, itl := range r.itls {
			allITLs = append(allITLs, float64(itl.Microseconds())/1000.0)
		}
	}

	if successes == 0 {
		return benchSample{}, fmt.Errorf("benchmark: no successful requests")
	}

	sort.Float64s(ttfts)
	sort.Float64s(e2es)
	sort.Float64s(allITLs)

	return benchSample{
		Index:              rep + 1,
		MeasuredRequests:   measured,
		SuccessfulRequests: successes,
		ErrorCount:         errorCount,
		ElapsedSeconds:     elapsed.Seconds(),
		TokensPerSecond:    float64(totalTokens) / elapsed.Seconds(),
		RequestsPerSec:     float64(successes) / elapsed.Seconds(),
		TTFTP50Ms:          percentile(ttfts, 0.50),
		TTFTP95Ms:          percentile(ttfts, 0.95),
		TTFTP99Ms:          percentile(ttfts, 0.99),
		E2EP50Ms:           percentile(e2es, 0.50),
		E2EP95Ms:           percentile(e2es, 0.95),
		E2EP99Ms:           percentile(e2es, 0.99),
		ITLP50Ms:           percentile(allITLs, 0.50),
		ITLP95Ms:           percentile(allITLs, 0.95),
		ITLP99Ms:           percentile(allITLs, 0.99),
	}, nil
}

func doServingBenchRequest(client *http.Client, targetURL, prompt string, outputLen int, tok model.Tokenizer) (benchRequestResult, error) {
	payload := map[string]any{
		"model":      "gollmgo-default",
		"max_tokens": outputLen,
		"stream":     true,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return benchRequestResult{}, err
	}

	reqStart := time.Now()
	resp, err := client.Post(targetURL+"/v1/chat/completions", "application/json",
		io.NopCloser(io.NewSectionReader(newBytesReader(body), 0, int64(len(body)))))
	if err != nil {
		return benchRequestResult{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		return benchRequestResult{}, fmt.Errorf("benchmark: unexpected status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}

	ttft, itls, tokens, err := collectStreamMetrics(resp.Body, reqStart, tok)
	if err != nil {
		return benchRequestResult{}, err
	}

	return benchRequestResult{
		ttft:   ttft,
		e2e:    time.Since(reqStart),
		itls:   itls,
		tokens: tokens,
	}, nil
}

func collectStreamMetrics(body io.Reader, reqStart time.Time, tok model.Tokenizer) (time.Duration, []time.Duration, int, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)

	var (
		ttft          time.Duration
		itls          []time.Duration
		lastChunkTime = reqStart
		firstToken    = true
		totalTokens   int
		outputText    strings.Builder
	)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return 0, nil, 0, fmt.Errorf("benchmark: parse stream chunk: %w", err)
		}

		content := streamChunkContent(chunk)
		if content == "" {
			continue
		}

		now := time.Now()
		if firstToken {
			ttft = now.Sub(reqStart)
			firstToken = false
		} else {
			itls = append(itls, now.Sub(lastChunkTime))
		}
		lastChunkTime = now

		deltaTokens := 1
		if tok != nil {
			outputText.WriteString(content)
			newCount, err := completionTokenCount(tok, outputText.String())
			if err != nil {
				return 0, nil, 0, err
			}
			deltaTokens = newCount - totalTokens
			if deltaTokens < 0 {
				return 0, nil, 0, fmt.Errorf("benchmark: token count went backwards")
			}
		}
		if deltaTokens > 0 {
			totalTokens += deltaTokens
		}
	}

	if err := scanner.Err(); err != nil {
		return 0, nil, 0, fmt.Errorf("benchmark: read stream: %w", err)
	}
	if totalTokens == 0 {
		return 0, nil, 0, fmt.Errorf("benchmark: stream produced no content tokens")
	}

	return ttft, itls, totalTokens, nil
}

func streamChunkContent(chunk streamChunk) string {
	var b strings.Builder
	for _, choice := range chunk.Choices {
		b.WriteString(choice.Delta.Content)
	}
	return b.String()
}

func samplePoissonInterval(rng *rand.Rand, qps float64) time.Duration {
	if qps <= 0 {
		return 0
	}
	seconds := rng.ExpFloat64() / qps
	return time.Duration(seconds * float64(time.Second))
}

func aggregateServingSamples(opts benchOptions, samples []benchSample, promptMode string, measuredRequests, errorCount int) BenchResult {
	result := BenchResult{
		Timestamp:        time.Now().UTC().Format(time.RFC3339),
		GitSHA:           getGitSHA(),
		Hardware:         "DGX Spark GB10",
		Mode:             "serving",
		NumPrompts:       opts.NumPrompts,
		PromptLen:        opts.PromptLen,
		OutputLen:        opts.OutputLen,
		Concurrency:      opts.Concurrency,
		MeasuredRequests: measuredRequests,
		WarmupRequests:   max(opts.WarmupRequests, 0),
		Repetitions:      max(opts.Repetitions, 1),
		ErrorCount:       errorCount,
		ArrivalMode:      opts.arrivalMode(),
		PromptGeneration: promptMode,
		TokenCountMode:   tokenCountMode(promptMode),
		Samples:          samples,
	}
	if opts.QPS > 0 {
		result.QPS = opts.QPS
	}
	if opts.Duration > 0 {
		result.DurationSeconds = opts.Duration.Seconds()
	}

	var (
		elapsedVals []float64
		tpsVals     []float64
		rpsVals     []float64
		ttft50Vals  []float64
		ttft95Vals  []float64
		ttft99Vals  []float64
		e2e50Vals   []float64
		e2e95Vals   []float64
		e2e99Vals   []float64
		itl50Vals   []float64
		itl95Vals   []float64
		itl99Vals   []float64
	)
	for _, sample := range samples {
		elapsedVals = append(elapsedVals, sample.ElapsedSeconds)
		tpsVals = append(tpsVals, sample.TokensPerSecond)
		rpsVals = append(rpsVals, sample.RequestsPerSec)
		ttft50Vals = append(ttft50Vals, sample.TTFTP50Ms)
		ttft95Vals = append(ttft95Vals, sample.TTFTP95Ms)
		ttft99Vals = append(ttft99Vals, sample.TTFTP99Ms)
		e2e50Vals = append(e2e50Vals, sample.E2EP50Ms)
		e2e95Vals = append(e2e95Vals, sample.E2EP95Ms)
		e2e99Vals = append(e2e99Vals, sample.E2EP99Ms)
		itl50Vals = append(itl50Vals, sample.ITLP50Ms)
		itl95Vals = append(itl95Vals, sample.ITLP95Ms)
		itl99Vals = append(itl99Vals, sample.ITLP99Ms)
	}
	sort.Float64s(elapsedVals)
	sort.Float64s(tpsVals)
	sort.Float64s(rpsVals)
	sort.Float64s(ttft50Vals)
	sort.Float64s(ttft95Vals)
	sort.Float64s(ttft99Vals)
	sort.Float64s(e2e50Vals)
	sort.Float64s(e2e95Vals)
	sort.Float64s(e2e99Vals)
	sort.Float64s(itl50Vals)
	sort.Float64s(itl95Vals)
	sort.Float64s(itl99Vals)

	result.ElapsedSeconds = percentile(elapsedVals, 0.50)
	result.TokensPerSecond = percentile(tpsVals, 0.50)
	result.RequestsPerSec = percentile(rpsVals, 0.50)
	result.TTFTP50Ms = percentile(ttft50Vals, 0.50)
	result.TTFTP95Ms = percentile(ttft95Vals, 0.50)
	result.TTFTP99Ms = percentile(ttft99Vals, 0.50)
	result.E2EP50Ms = percentile(e2e50Vals, 0.50)
	result.E2EP95Ms = percentile(e2e95Vals, 0.50)
	result.E2EP99Ms = percentile(e2e99Vals, 0.50)
	result.ITLP50Ms = percentile(itl50Vals, 0.50)
	result.ITLP95Ms = percentile(itl95Vals, 0.50)
	result.ITLP99Ms = percentile(itl99Vals, 0.50)

	return result
}

func tokenCountMode(promptMode string) string {
	if promptMode == "tokenizer_exact" {
		return "tokenizer_completion"
	}
	return "sse_content_chunks"
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func median(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	cp := append([]float64(nil), vals...)
	sort.Float64s(cp)
	mid := len(cp) / 2
	if len(cp)%2 == 1 {
		return cp[mid]
	}
	return (cp[mid-1] + cp[mid]) / 2
}

func spread(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	minVal, maxVal := vals[0], vals[0]
	for _, v := range vals[1:] {
		minVal = math.Min(minVal, v)
		maxVal = math.Max(maxVal, v)
	}
	return maxVal - minVal
}
