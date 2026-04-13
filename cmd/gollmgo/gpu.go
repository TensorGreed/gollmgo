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
// Returns the runner and tokenizer. Caller must close both.
func initGPURunner(log *slog.Logger, modelPath, tokenizerPath string, deviceID int) (backend.Runner, model.Tokenizer, error) {
	log.Info("initializing GPU runner", "model", modelPath, "device", deviceID)

	// Create CUDA runner.
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

	// Load model weights.
	log.Info("loading model weights", "path", modelPath)
	tensors, meta, err := model.LoadSafetensorsWeights(modelPath)
	if err != nil {
		runner.Close()
		return nil, nil, fmt.Errorf("load weights: %w", err)
	}

	log.Info("model metadata",
		"family", meta.Family,
		"layers", meta.NumLayers,
		"hidden", meta.HiddenSize,
		"vocab", meta.VocabSize,
		"dtype", meta.Dtype)

	// Infer intermediate size from gate_proj shape.
	intermediateSize := 0
	for _, t := range tensors {
		if t.Name == "model.layers.0.mlp.gate_proj.weight" && len(t.Shape) == 2 {
			intermediateSize = t.Shape[0]
			break
		}
	}
	if intermediateSize == 0 {
		intermediateSize = meta.HiddenSize * 4 // common default
	}

	// Create CUDA model.
	cudaModel, err := cuda.LoadModel(runner, meta, intermediateSize)
	if err != nil {
		runner.Close()
		return nil, nil, fmt.Errorf("create CUDA model: %w", err)
	}

	// Upload weights.
	for i, t := range tensors {
		if err := cudaModel.LoadWeight(t.Name, t.Data, t.Dtype); err != nil {
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

	// Warmup.
	if err := runner.Warmup(context.Background(), backend.WarmupProfile{
		MaxBatchSize: 64,
		MaxSeqLen:    meta.MaxSeqLen,
		BlockSize:    16,
	}); err != nil {
		log.Warn("warmup failed (non-fatal)", "error", err)
	}

	// Create runner with model.
	fullRunner := &cuda.CUDARunnerWithModel{
		CUDARunner:     runner,
		Model:          cudaModel,
		SeqSlotTables:  make(map[uint64][]int32),
		SeqContextLens: make(map[uint64]int32),
	}

	// Load tokenizer.
	var tokenizer model.Tokenizer
	if tokenizerPath == "" {
		// Try to find tokenizer.json next to the model.
		dir := filepath.Dir(modelPath)
		tokenizerPath = filepath.Join(dir, "tokenizer.json")
	}

	hfTok, err := model.LoadHFTokenizer(tokenizerPath, "</s>", "<s>")
	if err != nil {
		log.Warn("HF tokenizer load failed, falling back to byte-level",
			"path", tokenizerPath, "error", err)
		tokenizer = model.NewByteLevelTokenizer(meta.VocabSize, 2)
	} else {
		tokenizer = hfTok
		log.Info("tokenizer loaded", "vocab_size", hfTok.VocabSize(), "eos_id", hfTok.EOSTokenID())
	}

	return fullRunner, tokenizer, nil
}
