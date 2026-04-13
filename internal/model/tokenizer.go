// Package model — tokenizer.go provides the byte-level BPE tokenizer.
//
// This is a minimal implementation sufficient for correctness testing.
// A high-performance tokenizer (sentencepiece or tiktoken compatible)
// will be integrated in a later milestone.
package model

import (
	"fmt"
	"strings"
)

// ByteLevelTokenizer maps each byte to a unique token ID.
// This is a placeholder for development and testing.
// Real tokenizer integration (sentencepiece/tiktoken) deferred to M5.
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
