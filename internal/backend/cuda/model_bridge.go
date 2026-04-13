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
	dtypeFlag := C.int(0) // GOLLMGO_DTYPE_FP16
	if meta.Dtype == "bf16" || meta.Dtype == "BF16" || meta.Dtype == "bfloat16" {
		dtypeFlag = C.int(1) // GOLLMGO_DTYPE_BF16
	}

	cfg := C.gollmgo_model_config_t{
		num_layers:        C.int(meta.NumLayers),
		hidden_size:       C.int(meta.HiddenSize),
		intermediate_size: C.int(intermediateSize),
		num_heads:         C.int(meta.NumHeads),
		num_kv_heads:      C.int(meta.NumKVHeads),
		vocab_size:        C.int(meta.VocabSize),
		max_seq_len:       C.int(meta.MaxSeqLen),
		head_dim:          C.int(0),
		rms_norm_eps:      C.float(1e-5),
		dtype:             dtypeFlag,
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
		m.handle, cName, unsafe.Pointer(&data[0]),
		C.int64_t(len(data)), cDtype,
	)
	if status != C.GOLLMGO_OK {
		return fmt.Errorf("cuda: load_weight %q failed (status %d)", name, int(status))
	}
	return nil
}

// Ready marks the model as ready for inference.
func (m *CUDAModel) Ready() error {
	status := C.gollmgo_model_ready(m.handle)
	if status != C.GOLLMGO_OK {
		return fmt.Errorf("cuda: model_ready failed (status %d)", int(status))
	}
	return nil
}

// Forward runs one eager forward pass and returns logits for all tokens.
func (m *CUDAModel) Forward(_ context.Context, tokenIDs []int32, positions []int32) ([]float32, error) {
	n := len(tokenIDs)
	if n == 0 {
		return nil, fmt.Errorf("cuda: empty token input")
	}
	vocabSize := m.config.VocabSize
	logits := make([]float32, n*vocabSize)

	status := C.gollmgo_model_forward(
		m.backend, m.handle,
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
		m.backend, m.handle,
		(*C.int32_t)(unsafe.Pointer(&tokenIDs[0])),
		(*C.int32_t)(unsafe.Pointer(&positions[0])),
		slotPtr, C.int(n), cacheHandle,
		(*C.float)(unsafe.Pointer(&logits[0])),
	)
	if status != C.GOLLMGO_OK {
		return nil, fmt.Errorf("cuda: forward_prefill failed (status %d)", int(status))
	}
	return logits, nil
}

// ForwardPaged runs one paged forward pass for decode-only tokens.
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
		m.backend, m.handle,
		(*C.int32_t)(unsafe.Pointer(&tokenIDs[0])),
		(*C.int32_t)(unsafe.Pointer(&positions[0])),
		(*C.int32_t)(unsafe.Pointer(&slotMapping[0])),
		C.int(nTokens), kvCache.handle,
		(*C.int32_t)(unsafe.Pointer(&seqLens[0])),
		(*C.int32_t)(unsafe.Pointer(&slotTables[0])),
		C.int(nSeqs), C.int(maxContextLen),
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
type CUDARunnerWithModel struct {
	*CUDARunner
	Model   *CUDAModel
	KVCache *CUDAKVCache
}

// Step executes a forward pass. Handles mixed prefill/decode batches by
// splitting them: prefill sequences use eager attention + KV write,
// decode sequences use paged attention reading from the KV cache.
func (r *CUDARunnerWithModel) Step(ctx context.Context, batch *backend.Batch) (*backend.StepOutput, error) {
	if r.Model == nil {
		return nil, fmt.Errorf("cuda: no model loaded")
	}

	nSeq := len(batch.SequenceIDs)
	vocabSize := r.Model.config.VocabSize

	// Classify sequences.
	hasPrefill := false
	hasDecode := false
	for _, p := range batch.IsPrefill {
		if p {
			hasPrefill = true
		} else {
			hasDecode = true
		}
	}

	// Pure decode batch.
	if !hasPrefill && r.KVCache != nil && len(batch.SlotMapping) > 0 {
		return r.stepPaged(ctx, batch, nSeq, vocabSize)
	}

	// Pure prefill batch.
	if !hasDecode {
		return r.stepPrefillBatch(ctx, batch, nSeq, vocabSize)
	}

	// Mixed batch: split into prefill sub-batch and decode sub-batch,
	// run each through the correct path, merge results.
	return r.stepMixed(ctx, batch, nSeq, vocabSize)
}

// stepPrefillBatch handles a batch where all sequences are in prefill.
// Uses eager attention with KV cache write. Extracts last-token logits
// per sequence using SeqTokenCounts for correct variable-length indexing.
func (r *CUDARunnerWithModel) stepPrefillBatch(ctx context.Context, batch *backend.Batch, nSeq, vocabSize int) (*backend.StepOutput, error) {
	logits, err := r.Model.ForwardPrefill(ctx, batch.TokenIDs, batch.Positions, batch.SlotMapping, r.KVCache)
	if err != nil {
		return nil, err
	}
	return r.extractLastTokenLogits(logits, batch.SeqTokenCounts, nSeq, vocabSize)
}

func (r *CUDARunnerWithModel) stepEager(ctx context.Context, batch *backend.Batch, nSeq, vocabSize int) (*backend.StepOutput, error) {
	logits, err := r.Model.Forward(ctx, batch.TokenIDs, batch.Positions)
	if err != nil {
		return nil, err
	}
	return r.extractLastTokenLogits(logits, batch.SeqTokenCounts, nSeq, vocabSize)
}

func (r *CUDARunnerWithModel) stepPaged(ctx context.Context, batch *backend.Batch, nSeq, vocabSize int) (*backend.StepOutput, error) {
	seqLens := batch.SeqContextLens
	seqTokenCounts := batch.SeqTokenCounts

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
		ctx, batch.TokenIDs, batch.Positions, batch.SlotMapping,
		r.KVCache, seqLens, slotTables,
		nSeq, int(maxContextLen), seqTokenCounts,
	)
	if err != nil {
		return nil, err
	}

	// Paged forward returns [nSeq * vocabSize] — one per sequence already.
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

// stepMixed splits a mixed prefill/decode batch into two sub-batches.
func (r *CUDARunnerWithModel) stepMixed(ctx context.Context, batch *backend.Batch, nSeq, vocabSize int) (*backend.StepOutput, error) {
	// Separate prefill and decode sequences.
	var prefillBatch, decodeBatch backend.Batch
	prefillIdx := make([]int, 0) // original index in batch
	decodeIdx := make([]int, 0)

	tokenOffset := 0
	for i := 0; i < nSeq; i++ {
		count := int(batch.SeqTokenCounts[i])
		if batch.IsPrefill[i] {
			prefillIdx = append(prefillIdx, i)
			prefillBatch.SequenceIDs = append(prefillBatch.SequenceIDs, batch.SequenceIDs[i])
			prefillBatch.IsPrefill = append(prefillBatch.IsPrefill, true)
			prefillBatch.TokenIDs = append(prefillBatch.TokenIDs, batch.TokenIDs[tokenOffset:tokenOffset+count]...)
			prefillBatch.Positions = append(prefillBatch.Positions, batch.Positions[tokenOffset:tokenOffset+count]...)
			if len(batch.SlotMapping) > 0 {
				prefillBatch.SlotMapping = append(prefillBatch.SlotMapping, batch.SlotMapping[tokenOffset:tokenOffset+count]...)
			}
			prefillBatch.SeqTokenCounts = append(prefillBatch.SeqTokenCounts, batch.SeqTokenCounts[i])
			prefillBatch.SeqContextLens = append(prefillBatch.SeqContextLens, batch.SeqContextLens[i])
			if i < len(batch.SeqSlotTables) {
				prefillBatch.SeqSlotTables = append(prefillBatch.SeqSlotTables, batch.SeqSlotTables[i])
			}
		} else {
			decodeIdx = append(decodeIdx, i)
			decodeBatch.SequenceIDs = append(decodeBatch.SequenceIDs, batch.SequenceIDs[i])
			decodeBatch.IsPrefill = append(decodeBatch.IsPrefill, false)
			decodeBatch.TokenIDs = append(decodeBatch.TokenIDs, batch.TokenIDs[tokenOffset:tokenOffset+count]...)
			decodeBatch.Positions = append(decodeBatch.Positions, batch.Positions[tokenOffset:tokenOffset+count]...)
			if len(batch.SlotMapping) > 0 {
				decodeBatch.SlotMapping = append(decodeBatch.SlotMapping, batch.SlotMapping[tokenOffset:tokenOffset+count]...)
			}
			decodeBatch.SeqTokenCounts = append(decodeBatch.SeqTokenCounts, batch.SeqTokenCounts[i])
			decodeBatch.SeqContextLens = append(decodeBatch.SeqContextLens, batch.SeqContextLens[i])
			if i < len(batch.SeqSlotTables) {
				decodeBatch.SeqSlotTables = append(decodeBatch.SeqSlotTables, batch.SeqSlotTables[i])
			}
		}
		tokenOffset += count
	}

	// Run prefill sub-batch.
	var prefillOut *backend.StepOutput
	if len(prefillBatch.SequenceIDs) > 0 {
		var err error
		prefillOut, err = r.stepPrefillBatch(ctx, &prefillBatch, len(prefillBatch.SequenceIDs), vocabSize)
		if err != nil {
			return nil, err
		}
	}

	// Run decode sub-batch.
	var decodeOut *backend.StepOutput
	if len(decodeBatch.SequenceIDs) > 0 && r.KVCache != nil {
		var err error
		decodeOut, err = r.stepPaged(ctx, &decodeBatch, len(decodeBatch.SequenceIDs), vocabSize)
		if err != nil {
			return nil, err
		}
	}

	// Merge results back into original order.
	out := &backend.StepOutput{Logits: make([][]float32, nSeq)}
	pi, di := 0, 0
	for i := 0; i < nSeq; i++ {
		if batch.IsPrefill[i] {
			if prefillOut != nil && pi < len(prefillOut.Logits) {
				out.Logits[i] = prefillOut.Logits[pi]
			}
			pi++
		} else {
			if decodeOut != nil && di < len(decodeOut.Logits) {
				out.Logits[i] = decodeOut.Logits[di]
			}
			di++
		}
	}
	return out, nil
}

// extractLastTokenLogits picks the last token's logits for each sequence
// using SeqTokenCounts for correct variable-length indexing.
func (r *CUDARunnerWithModel) extractLastTokenLogits(allLogits []float32, seqTokenCounts []int32, nSeq, vocabSize int) (*backend.StepOutput, error) {
	out := &backend.StepOutput{Logits: make([][]float32, nSeq)}
	tokenOffset := 0

	for i := 0; i < nSeq; i++ {
		count := 1 // default: 1 token per sequence (decode)
		if i < len(seqTokenCounts) {
			count = int(seqTokenCounts[i])
		}
		lastIdx := tokenOffset + count - 1
		start := lastIdx * vocabSize
		end := start + vocabSize
		if end > len(allLogits) {
			return nil, fmt.Errorf("cuda: logits index out of range (seq %d, token %d, logits len %d)", i, lastIdx, len(allLogits))
		}
		out.Logits[i] = allLogits[start:end]
		tokenOffset += count
	}
	return out, nil
}

// Close releases the KV cache, model, and backend runner in reverse order.
// Safe to call multiple times.
func (r *CUDARunnerWithModel) Close() error {
	var firstErr error
	if r.KVCache != nil {
		if err := r.KVCache.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("cuda: kvcache close: %w", err)
		}
		r.KVCache = nil
	}
	if r.Model != nil {
		if err := r.Model.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("cuda: model close: %w", err)
		}
		r.Model = nil
	}
	if r.CUDARunner != nil {
		if err := r.CUDARunner.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("cuda: runner close: %w", err)
		}
		r.CUDARunner = nil
	}
	return firstErr
}

// Ensure CUDARunnerWithModel satisfies the Runner interface.
var _ backend.Runner = (*CUDARunnerWithModel)(nil)

// Prevent unused import.
var _ = unsafe.Pointer(nil)
