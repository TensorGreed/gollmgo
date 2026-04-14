//go:build gpu

package main

import (
	"context"
	"fmt"
	"log/slog"
	"path/filepath"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/backend/cuda"
	"github.com/TensorGreed/gollmgo/internal/config"
	"github.com/TensorGreed/gollmgo/internal/model"
	"github.com/TensorGreed/gollmgo/internal/model/hfhub"
)

// initGPURunner creates a CUDA runner with a loaded model. The handle
// carries an already-resolved model directory (either a local path or a
// freshly downloaded HF Hub repo — cmdServe handles that upstream).
// Returns the runner, tokenizer, and the computed number of KV cache blocks
// (so cmdServe can create a matching Go-side block pool).
func initGPURunner(log *slog.Logger, cfg config.Config, handle *hfhub.Handle, deviceID int) (backend.Runner, model.Tokenizer, int, error) {
	tokenizerPath := cfg.TokenizerPath
	dir := handle.LocalDir
	log.Info("initializing GPU runner", "model_dir", dir, "device", deviceID,
		"repo", handle.RepoID, "sharded", handle.IsSharded, "shards", len(handle.WeightsFiles))

	// --- Load config.json for authoritative model metadata ---
	configPath := handle.ConfigPath
	if configPath == "" {
		configPath = filepath.Join(dir, "config.json")
	}
	hfCfg, err := model.LoadHFConfig(configPath)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("load config.json: %w", err)
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
		return nil, nil, 0, fmt.Errorf("create CUDA runner: %w", err)
	}

	info := runner.DeviceInfo()
	log.Info("GPU device",
		"name", info.Name,
		"compute", fmt.Sprintf("%d.%d", info.ComputeMajor, info.ComputeMinor),
		"total_gb", fmt.Sprintf("%.1f", float64(info.TotalMemoryBytes)/(1<<30)),
		"free_gb", fmt.Sprintf("%.1f", float64(info.FreeMemoryBytes)/(1<<30)))

	// --- Load weights. Handle both single-file and sharded layouts. ---
	log.Info("loading model weights", "dir", dir)
	var tensors []model.WeightTensor
	if handle.IsSharded || len(handle.WeightsFiles) > 1 {
		tensors, _, err = model.LoadSafetensorsDirectory(dir)
	} else if len(handle.WeightsFiles) == 1 {
		tensors, _, err = model.LoadSafetensorsWeights(handle.WeightsFiles[0])
	} else {
		// Backwards-compat: caller passed a single-file ModelPath directly.
		tensors, _, err = model.LoadSafetensorsWeights(cfg.ModelPath)
	}
	if err != nil {
		runner.Close()
		return nil, nil, 0, fmt.Errorf("load weights: %w", err)
	}
	log.Info("weight tensors loaded", "count", len(tensors))

	// --- Create CUDA model ---
	cudaModel, err := cuda.LoadModel(runner, meta, hfCfg.IntermediateSize, cfg.Quantization)
	if err != nil {
		runner.Close()
		return nil, nil, 0, fmt.Errorf("create CUDA model: %w", err)
	}

	// Upload weights directly — BF16 weights stay in BF16 (native kernel support).
	for i, t := range tensors {
		if err := cudaModel.LoadWeight(t.Name, t.Data, t.Dtype); err != nil {
			cudaModel.Close()
			runner.Close()
			return nil, nil, 0, fmt.Errorf("upload weight %q: %w", t.Name, err)
		}
		if (i+1)%50 == 0 || i == len(tensors)-1 {
			log.Info("weight upload progress", "loaded", i+1, "total", len(tensors))
		}
	}

	if err := cudaModel.Ready(); err != nil {
		cudaModel.Close()
		runner.Close()
		return nil, nil, 0, fmt.Errorf("model ready: %w", err)
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

	// --- Compute KV cache size from device memory ---
	blockSize := cfg.BlockSize
	headDim := meta.HiddenSize / meta.NumHeads
	// Each KV block stores block_size tokens × num_kv_heads × head_dim × 2 bytes (FP16/BF16)
	// for both K and V, across all layers.
	bytesPerBlock := int64(blockSize) * int64(hfCfg.NumKeyValueHeads) * int64(headDim) * 2 * 2 * int64(meta.NumLayers)

	// Compute model weight memory from uploaded tensors.
	var weightBytes int64
	for _, t := range tensors {
		weightBytes += int64(len(t.Data))
	}

	// Budget: (device_free - weight_memory - scratch_headroom) * max_memory_fraction
	// Subtract weights and a fixed headroom for model scratch buffers, CUDA context, etc.
	scratchHeadroom := int64(512 * 1024 * 1024) // 512 MB for activations and CUDA overhead
	memFraction := cfg.MaxMemoryFraction
	freeAfterWeights := int64(info.FreeMemoryBytes) - weightBytes - scratchHeadroom
	if freeAfterWeights < 0 {
		freeAfterWeights = 0
	}
	availableBytes := int64(float64(freeAfterWeights) * memFraction)

	numBlocks := int(availableBytes / bytesPerBlock)
	if numBlocks < 64 {
		numBlocks = 64 // minimum viable cache
	}
	numSlots := numBlocks * blockSize

	log.Info("KV cache sizing",
		"device_free_gb", fmt.Sprintf("%.1f", float64(info.FreeMemoryBytes)/(1<<30)),
		"weight_gb", fmt.Sprintf("%.1f", float64(weightBytes)/(1<<30)),
		"max_memory_fraction", memFraction,
		"bytes_per_block", bytesPerBlock,
		"num_blocks", numBlocks,
		"num_slots", numSlots)

	kvCache, err := cuda.NewCUDAKVCache(runner, meta.NumLayers, numSlots, hfCfg.NumKeyValueHeads, headDim)
	if err != nil {
		cudaModel.Close()
		runner.Close()
		return nil, nil, 0, fmt.Errorf("create KV cache: %w", err)
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
		if handle.TokenizerPath != "" {
			tokenizerPath = handle.TokenizerPath
		} else {
			tokenizerPath = filepath.Join(dir, "tokenizer.json")
		}
	}

	hfTok, err := model.LoadHFTokenizer(tokenizerPath, "</s>", "<s>")
	if err != nil {
		log.Warn("HF tokenizer failed, using byte-level fallback", "error", err)
		tokenizer = model.NewByteLevelTokenizer(meta.VocabSize, int32(hfCfg.EosTokenID))
	} else {
		tokenizer = hfTok
		log.Info("tokenizer loaded", "vocab", hfTok.VocabSize(), "eos", hfTok.EOSTokenID())
	}

	return fullRunner, tokenizer, numBlocks, nil
}
