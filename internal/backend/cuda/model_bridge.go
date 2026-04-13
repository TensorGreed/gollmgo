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
	Model *CUDAModel
}

// Step executes a forward pass using the loaded model.
func (r *CUDARunnerWithModel) Step(ctx context.Context, batch *backend.Batch) (*backend.StepOutput, error) {
	if r.Model == nil {
		return nil, fmt.Errorf("cuda: no model loaded")
	}

	logits, err := r.Model.Forward(ctx, batch.TokenIDs, batch.Positions)
	if err != nil {
		return nil, err
	}

	vocabSize := r.Model.config.VocabSize
	nSeq := len(batch.SequenceIDs)
	out := &backend.StepOutput{
		Logits: make([][]float32, nSeq),
	}

	// For each sequence, extract the last token's logits.
	// In eager mode (M5), all tokens are passed; we take the last token per sequence.
	// For now, assume tokens are ordered per-sequence and we return the logits
	// for the last n_seq tokens.
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
