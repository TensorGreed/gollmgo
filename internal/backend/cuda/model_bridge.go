//go:build gpu

package cuda

/*
#include "gollmgo_model.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/model"
)

// CUDAModel wraps a GPU-resident model for forward pass execution.
type CUDAModel struct {
	handle  C.gollmgo_model_t
	backend C.gollmgo_backend_t
	config  model.ModelMeta
}

// LoadModel creates a CUDA model and loads weights from safetensors data.
func LoadModel(runner *CUDARunner, meta *model.ModelMeta, intermediateSize int) (*CUDAModel, error) {
	cfg := C.gollmgo_model_config_t{
		num_layers:        C.int(meta.NumLayers),
		hidden_size:       C.int(meta.HiddenSize),
		intermediate_size: C.int(intermediateSize),
		num_heads:         C.int(meta.NumHeads),
		num_kv_heads:      C.int(meta.NumKVHeads),
		vocab_size:        C.int(meta.VocabSize),
		max_seq_len:       C.int(meta.MaxSeqLen),
		head_dim:          C.int(0), // auto-derived
		rms_norm_eps:      C.float(1e-5),
	}

	var handle C.gollmgo_model_t
	status := C.gollmgo_model_create(runner.handle, &cfg, &handle)
	if status != C.GOLLMGO_OK {
		return nil, fmt.Errorf("cuda: model_create failed (status %d)", int(status))
	}

	return &CUDAModel{
		handle:  handle,
		backend: runner.handle,
		config:  *meta,
	}, nil
}

// LoadWeight copies a named weight tensor to GPU memory.
func (m *CUDAModel) LoadWeight(name string, data []byte, dtype string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cDtype := C.CString(dtype)
	defer C.free(unsafe.Pointer(cDtype))

	status := C.gollmgo_model_load_weight(
		m.handle,
		cName,
		unsafe.Pointer(&data[0]),
		C.int64_t(len(data)),
		cDtype,
	)
	if status != C.GOLLMGO_OK {
		return fmt.Errorf("cuda: load_weight %q failed (status %d)", name, int(status))
	}
	return nil
}

// Ready marks the model as ready for inference and allocates scratch buffers.
func (m *CUDAModel) Ready() error {
	status := C.gollmgo_model_ready(m.handle)
	if status != C.GOLLMGO_OK {
		return fmt.Errorf("cuda: model_ready failed (status %d)", int(status))
	}
	return nil
}

// Forward runs one eager forward pass and returns logits.
func (m *CUDAModel) Forward(_ context.Context, tokenIDs []int32, positions []int32) ([]float32, error) {
	n := len(tokenIDs)
	if n == 0 {
		return nil, fmt.Errorf("cuda: empty token input")
	}
	if len(positions) != n {
		return nil, fmt.Errorf("cuda: tokenIDs and positions length mismatch")
	}

	vocabSize := m.config.VocabSize
	logits := make([]float32, n*vocabSize)

	status := C.gollmgo_model_forward(
		m.backend,
		m.handle,
		(*C.int32_t)(unsafe.Pointer(&tokenIDs[0])),
		(*C.int32_t)(unsafe.Pointer(&positions[0])),
		C.int(n),
		(*C.float)(unsafe.Pointer(&logits[0])),
	)
	if status != C.GOLLMGO_OK {
		return nil, fmt.Errorf("cuda: forward failed (status %d)", int(status))
	}

	return logits, nil
}

// ForwardPrefill runs eager attention and writes K/V to the paged cache.
// Used for prefill steps so that subsequent decode steps can read from the cache.
func (m *CUDAModel) ForwardPrefill(
	_ context.Context,
	tokenIDs, positions, slotMapping []int32,
	kvCache *CUDAKVCache,
) ([]float32, error) {
	n := len(tokenIDs)
	if n == 0 {
		return nil, fmt.Errorf("cuda: empty token input")
	}

	vocabSize := m.config.VocabSize
	logits := make([]float32, n*vocabSize)

	var cacheHandle C.gollmgo_kvcache_t
	var slotPtr *C.int32_t
	if kvCache != nil && len(slotMapping) > 0 {
		cacheHandle = kvCache.handle
		slotPtr = (*C.int32_t)(unsafe.Pointer(&slotMapping[0]))
	}

	status := C.gollmgo_model_forward_prefill(
		m.backend,
		m.handle,
		(*C.int32_t)(unsafe.Pointer(&tokenIDs[0])),
		(*C.int32_t)(unsafe.Pointer(&positions[0])),
		slotPtr,
		C.int(n),
		cacheHandle,
		(*C.float)(unsafe.Pointer(&logits[0])),
	)
	if status != C.GOLLMGO_OK {
		return nil, fmt.Errorf("cuda: forward_prefill failed (status %d)", int(status))
	}
	return logits, nil
}

// ForwardPaged runs one forward pass with paged KV cache.
func (m *CUDAModel) ForwardPaged(
	_ context.Context,
	tokenIDs, positions, slotMapping []int32,
	kvCache *CUDAKVCache,
	seqLens, slotTables []int32,
	nSeqs, maxContextLen int,
	seqTokenCounts []int32,
) ([]float32, error) {
	nTokens := len(tokenIDs)
	if nTokens == 0 {
		return nil, fmt.Errorf("cuda: empty token input")
	}

	vocabSize := m.config.VocabSize
	logits := make([]float32, nSeqs*vocabSize)

	status := C.gollmgo_model_forward_paged(
		m.backend,
		m.handle,
		(*C.int32_t)(unsafe.Pointer(&tokenIDs[0])),
		(*C.int32_t)(unsafe.Pointer(&positions[0])),
		(*C.int32_t)(unsafe.Pointer(&slotMapping[0])),
		C.int(nTokens),
		kvCache.handle,
		(*C.int32_t)(unsafe.Pointer(&seqLens[0])),
		(*C.int32_t)(unsafe.Pointer(&slotTables[0])),
		C.int(nSeqs),
		C.int(maxContextLen),
		(*C.int32_t)(unsafe.Pointer(&seqTokenCounts[0])),
		(*C.float)(unsafe.Pointer(&logits[0])),
	)
	if status != C.GOLLMGO_OK {
		return nil, fmt.Errorf("cuda: forward_paged failed (status %d)", int(status))
	}

	return logits, nil
}

// Close destroys the model and frees GPU memory.
func (m *CUDAModel) Close() error {
	if m.handle != nil {
		C.gollmgo_model_destroy(m.handle)
		m.handle = nil
	}
	return nil
}

// CUDARunnerWithModel wraps CUDARunner + CUDAModel to implement backend.Runner.Step().
// If KVCache is set, Step uses the paged forward pass. Otherwise, eager.
type CUDARunnerWithModel struct {
	*CUDARunner
	Model   *CUDAModel
	KVCache *CUDAKVCache
}

// Step executes a forward pass using the loaded model.
// If the batch has SlotMapping and a KVCache is configured, uses paged attention.
// Otherwise falls back to eager (naive attention).
func (r *CUDARunnerWithModel) Step(ctx context.Context, batch *backend.Batch) (*backend.StepOutput, error) {
	if r.Model == nil {
		return nil, fmt.Errorf("cuda: no model loaded")
	}

	nSeq := len(batch.SequenceIDs)
	vocabSize := r.Model.config.VocabSize

	// Check if any sequence is in prefill mode.
	hasPrefill := false
	for _, p := range batch.IsPrefill {
		if p {
			hasPrefill = true
			break
		}
	}

	if hasPrefill {
		// Prefill: eager attention + KV cache write.
		return r.stepPrefill(ctx, batch, nSeq, vocabSize)
	}

	// Decode-only: paged attention reading from KV cache.
	if r.KVCache != nil && len(batch.SlotMapping) > 0 {
		return r.stepPaged(ctx, batch, nSeq, vocabSize)
	}

	// Fallback: eager (no KV cache).
	return r.stepEager(ctx, batch, nSeq, vocabSize)
}

func (r *CUDARunnerWithModel) stepPrefill(ctx context.Context, batch *backend.Batch, nSeq, vocabSize int) (*backend.StepOutput, error) {
	logits, err := r.Model.ForwardPrefill(ctx, batch.TokenIDs, batch.Positions, batch.SlotMapping, r.KVCache)
	if err != nil {
		return nil, err
	}

	// Return logits for the last token of each sequence (prefill).
	out := &backend.StepOutput{Logits: make([][]float32, nSeq)}
	totalTokens := len(batch.TokenIDs)
	for i := 0; i < nSeq; i++ {
		tokenIdx := totalTokens - nSeq + i
		if tokenIdx < 0 {
			tokenIdx = i
		}
		start := tokenIdx * vocabSize
		end := start + vocabSize
		if end > len(logits) {
			return nil, fmt.Errorf("cuda: logits index out of range")
		}
		out.Logits[i] = logits[start:end]
	}
	return out, nil
}

func (r *CUDARunnerWithModel) stepEager(ctx context.Context, batch *backend.Batch, nSeq, vocabSize int) (*backend.StepOutput, error) {
	logits, err := r.Model.Forward(ctx, batch.TokenIDs, batch.Positions)
	if err != nil {
		return nil, err
	}

	out := &backend.StepOutput{Logits: make([][]float32, nSeq)}
	totalTokens := len(batch.TokenIDs)
	for i := 0; i < nSeq; i++ {
		tokenIdx := totalTokens - nSeq + i
		if tokenIdx < 0 {
			tokenIdx = i
		}
		start := tokenIdx * vocabSize
		end := start + vocabSize
		if end > len(logits) {
			return nil, fmt.Errorf("cuda: logits index out of range")
		}
		out.Logits[i] = logits[start:end]
	}
	return out, nil
}

func (r *CUDARunnerWithModel) stepPaged(ctx context.Context, batch *backend.Batch, nSeq, vocabSize int) (*backend.StepOutput, error) {
	// Use per-sequence metadata from the batch (populated by ServingEngine).
	seqLens := batch.SeqContextLens
	seqTokenCounts := batch.SeqTokenCounts

	// Flatten per-sequence slot tables into a padded 2D array.
	maxContextLen := int32(0)
	for _, l := range seqLens {
		if l > maxContextLen {
			maxContextLen = l
		}
	}
	if maxContextLen == 0 {
		maxContextLen = 1
	}

	slotTables := make([]int32, nSeq*int(maxContextLen))
	for i := 0; i < nSeq && i < len(batch.SeqSlotTables); i++ {
		table := batch.SeqSlotTables[i]
		for j := 0; j < len(table) && j < int(maxContextLen); j++ {
			slotTables[i*int(maxContextLen)+j] = table[j]
		}
	}

	logits, err := r.Model.ForwardPaged(
		ctx,
		batch.TokenIDs, batch.Positions, batch.SlotMapping,
		r.KVCache,
		seqLens, slotTables,
		nSeq, int(maxContextLen),
		seqTokenCounts,
	)
	if err != nil {
		return nil, err
	}

	out := &backend.StepOutput{Logits: make([][]float32, nSeq)}
	for i := 0; i < nSeq; i++ {
		start := i * vocabSize
		end := start + vocabSize
		if end > len(logits) {
			return nil, fmt.Errorf("cuda: logits index out of range")
		}
		out.Logits[i] = logits[start:end]
	}
	return out, nil
}
