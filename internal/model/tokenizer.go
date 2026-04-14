// Package model — tokenizer.go provides the byte-level fallback tokenizer.
//
// On the serve path, the primary tokenizer is HFTokenizer (hf_tokenizer.go),
// which loads tokenizer.json. ByteLevelTokenizer below is used in two
// situations:
//   - mock/dev mode (no --model)
//   - HF tokenizer load failure (cmd/gollmgo/gpu.go logs a warning)
//
// It encodes any input as raw UTF-8 bytes; decoding is exact for ASCII and
// works for arbitrary bytes but won't reflect a real model's vocabulary.
// If a real LLM is loaded with this fallback active, output will be
// gibberish — make sure tokenizer.json is present alongside the weights.
package model

import (
	"fmt"
	"strings"
)

// ByteLevelTokenizer maps each byte to a unique token ID. See package doc
// for when this is used vs. the real HFTokenizer path.
type ByteLevelTokenizer struct {
	vocabSize int
	eosID     int32
}

// NewByteLevelTokenizer creates a byte-level tokenizer.
// vocabSize should be >= 256. eosID is the end-of-sequence token.
func NewByteLevelTokenizer(vocabSize int, eosID int32) *ByteLevelTokenizer {
	if vocabSize < 256 {
		vocabSize = 256
	}
	return &ByteLevelTokenizer{vocabSize: vocabSize, eosID: eosID}
}

func (t *ByteLevelTokenizer) Encode(text string) ([]int32, error) {
	if text == "" {
		return nil, nil
	}
	bs := []byte(text)
	ids := make([]int32, len(bs))
	for i, b := range bs {
		ids[i] = int32(b)
	}
	return ids, nil
}

func (t *ByteLevelTokenizer) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	sb.Grow(len(ids))
	for _, id := range ids {
		if id < 0 || id >= 256 {
			return "", fmt.Errorf("tokenizer: id %d out of byte range", id)
		}
		sb.WriteByte(byte(id))
	}
	return sb.String(), nil
}

func (t *ByteLevelTokenizer) VocabSize() int   { return t.vocabSize }
func (t *ByteLevelTokenizer) EOSTokenID() int32 { return t.eosID }

// Compile-time check.
var _ Tokenizer = (*ByteLevelTokenizer)(nil)
