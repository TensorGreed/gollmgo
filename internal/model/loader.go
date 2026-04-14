// Package model defines interfaces for model and tokenizer loading.
package model

import "context"

// ModelMeta holds normalized metadata about a loaded model.
type ModelMeta struct {
	Name             string
	Family           string // e.g. "llama"
	NumLayers        int
	HiddenSize       int
	IntermediateSize int
	NumHeads         int
	NumKVHeads       int
	VocabSize        int
	MaxSeqLen        int
	Dtype            string // "fp16", "bf16", etc.
	BOSTokenID       int32
	EOSTokenID       int32
}

// Loader loads model weights and metadata from disk.
type Loader interface {
	// Load reads the model at the given path and returns metadata.
	Load(ctx context.Context, path string) (*ModelMeta, error)
}

// Tokenizer encodes and decodes text to/from token IDs.
type Tokenizer interface {
	Encode(text string) ([]int32, error)
	Decode(ids []int32) (string, error)
	VocabSize() int
	EOSTokenID() int32
}
