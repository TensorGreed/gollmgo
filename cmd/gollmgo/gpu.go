//go:build gpu

package main

import (
	"context"
	"fmt"
	"log/slog"
	"path/filepath"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/backend/cuda"
	"github.com/TensorGreed/gollmgo/internal/model"
)

// initGPURunner creates a CUDA runner with a loaded model.
func initGPURunner(log *slog.Logger, modelPath, tokenizerPath string, deviceID int) (backend.Runner, model.Tokenizer, error) {
	log.Info("initializing GPU runner", "model", modelPath, "device", deviceID)

	dir := filepath.Dir(modelPath)

	// --- Load config.json for authoritative model metadata ---
	configPath := filepath.Join(dir, "config.json")
	hfCfg, err := model.LoadHFConfig(configPath)
	if err != nil {
		return nil, nil, fmt.Errorf("load config.json: %w", err)
	}
	meta := hfCfg.ToModelMeta()

	log.Info("model config",
		"family", meta.Family,
		"layers", meta.NumLayers,
		"hidden", meta.HiddenSize,
		"heads", meta.NumHeads,
		"kv_heads", meta.NumKVHeads,
		"vocab", meta.VocabSize,
		"max_seq_len", meta.MaxSeqLen,
		"dtype", meta.Dtype)

	// --- Create CUDA runner ---
	runner, err := cuda.New(deviceID)
	if err != nil {
		return nil, nil, fmt.Errorf("create CUDA runner: %w", err)
	}

	info := runner.DeviceInfo()
	log.Info("GPU device",
		"name", info.Name,
		"compute", fmt.Sprintf("%d.%d", info.ComputeMajor, info.ComputeMinor),
		"total_gb", fmt.Sprintf("%.1f", float64(info.TotalMemoryBytes)/(1<<30)),
		"free_gb", fmt.Sprintf("%.1f", float64(info.FreeMemoryBytes)/(1<<30)))

	// --- Load weights ---
	log.Info("loading model weights", "path", modelPath)
	tensors, _, err := model.LoadSafetensorsWeights(modelPath)
	if err != nil {
		runner.Close()
		return nil, nil, fmt.Errorf("load weights: %w", err)
	}
	log.Info("weight tensors loaded", "count", len(tensors))

	// --- Create CUDA model ---
	cudaModel, err := cuda.LoadModel(runner, meta, hfCfg.IntermediateSize)
	if err != nil {
		runner.Close()
		return nil, nil, fmt.Errorf("create CUDA model: %w", err)
	}

	// Upload weights — convert BF16 to FP16 since kernels use __half.
	for i, t := range tensors {
		data := t.Data
		dtype := t.Dtype
		if dtype == "BF16" {
			data = model.ConvertBF16ToFP16(data)
			dtype = "F16"
		}
		if err := cudaModel.LoadWeight(t.Name, data, dtype); err != nil {
			cudaModel.Close()
			runner.Close()
			return nil, nil, fmt.Errorf("upload weight %q: %w", t.Name, err)
		}
		if (i+1)%50 == 0 || i == len(tensors)-1 {
			log.Info("weight upload progress", "loaded", i+1, "total", len(tensors))
		}
	}

	if err := cudaModel.Ready(); err != nil {
		cudaModel.Close()
		runner.Close()
		return nil, nil, fmt.Errorf("model ready: %w", err)
	}
	log.Info("model ready on GPU")

	// Warmup.
	if err := runner.Warmup(context.Background(), backend.WarmupProfile{
		MaxBatchSize: 64,
		MaxSeqLen:    meta.MaxSeqLen,
		BlockSize:    16,
	}); err != nil {
		log.Warn("warmup note", "error", err)
	}

	// --- Create GPU KV cache ---
	blockSize := 16
	numBlocks := 1024 // matches the Go-side pool in cmdServe
	numSlots := numBlocks * blockSize
	headDim := meta.HiddenSize / meta.NumHeads

	kvCache, err := cuda.NewCUDAKVCache(runner, meta.NumLayers, numSlots, hfCfg.NumKeyValueHeads, headDim)
	if err != nil {
		cudaModel.Close()
		runner.Close()
		return nil, nil, fmt.Errorf("create KV cache: %w", err)
	}
	log.Info("GPU KV cache created", "slots", numSlots, "kv_heads", hfCfg.NumKeyValueHeads, "head_dim", headDim)

	// --- Build runner with model + KV cache ---
	fullRunner := &cuda.CUDARunnerWithModel{
		CUDARunner: runner,
		Model:      cudaModel,
		KVCache:    kvCache,
	}

	// --- Load tokenizer ---
	var tokenizer model.Tokenizer
	if tokenizerPath == "" {
		tokenizerPath = filepath.Join(dir, "tokenizer.json")
	}

	hfTok, err := model.LoadHFTokenizer(tokenizerPath, "</s>", "<s>")
	if err != nil {
		log.Warn("HF tokenizer failed, using byte-level fallback", "error", err)
		tokenizer = model.NewByteLevelTokenizer(meta.VocabSize, int32(hfCfg.EosTokenID))
	} else {
		tokenizer = hfTok
		log.Info("tokenizer loaded", "vocab", hfTok.VocabSize(), "eos", hfTok.EOSTokenID())
	}

	return fullRunner, tokenizer, nil
}
