package model

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// HFTokenizer loads and uses a HuggingFace tokenizer.json file.
// Supports BPE tokenizers as used by LLaMA, Mistral, etc.
type HFTokenizer struct {
	vocab     map[string]int32 // token string -> id
	invVocab  []string         // id -> token string
	merges    []mergePair
	eosID     int32
	bosID     int32
	vocabSize int
}

type mergePair struct {
	a, b string
}

// hfTokenizerJSON is the top-level structure of tokenizer.json.
type hfTokenizerJSON struct {
	Model struct {
		Type   string            `json:"type"`
		Vocab  map[string]int32  `json:"vocab"`
		Merges []string          `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

// LoadHFTokenizer loads a HuggingFace tokenizer.json file.
func LoadHFTokenizer(path string, eosToken, bosToken string) (*HFTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %s: %w", path, err)
	}

	var raw hfTokenizerJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("tokenizer: parse %s: %w", path, err)
	}

	if raw.Model.Type != "BPE" && raw.Model.Type != "" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q (expected BPE)", raw.Model.Type)
	}

	vocab := raw.Model.Vocab
	if vocab == nil {
		vocab = make(map[string]int32)
	}

	// Add special tokens.
	for _, tok := range raw.AddedTokens {
		vocab[tok.Content] = tok.ID
	}

	// Build inverse vocab.
	maxID := int32(0)
	for _, id := range vocab {
		if id > maxID {
			maxID = id
		}
	}
	invVocab := make([]string, maxID+1)
	for tok, id := range vocab {
		invVocab[id] = tok
	}

	// Parse merges.
	var merges []mergePair
	for _, line := range raw.Model.Merges {
		parts := strings.SplitN(line, " ", 2)
		if len(parts) == 2 {
			merges = append(merges, mergePair{a: parts[0], b: parts[1]})
		}
	}

	// Resolve special token IDs.
	eosID := int32(-1)
	bosID := int32(-1)
	if id, ok := vocab[eosToken]; ok {
		eosID = id
	}
	if id, ok := vocab[bosToken]; ok {
		bosID = id
	}

	return &HFTokenizer{
		vocab:     vocab,
		invVocab:  invVocab,
		merges:    merges,
		eosID:     eosID,
		bosID:     bosID,
		vocabSize: int(maxID + 1),
	}, nil
}

// Encode tokenizes text using BPE.
func (t *HFTokenizer) Encode(text string) ([]int32, error) {
	if text == "" {
		return nil, nil
	}

	// Start with individual bytes/characters as tokens.
	// HF BPE typically uses byte-level encoding where each byte is a token.
	tokens := make([]string, 0, len(text))
	for _, b := range []byte(text) {
		// HF byte-level BPE uses a specific byte-to-unicode mapping.
		// For now, use the raw byte representation.
		tok := string([]byte{b})
		tokens = append(tokens, tok)
	}

	// Build merge priority index.
	mergeRank := make(map[string]int, len(t.merges))
	for i, m := range t.merges {
		mergeRank[m.a+" "+m.b] = i
	}

	// Iteratively apply BPE merges.
	for {
		if len(tokens) < 2 {
			break
		}

		// Find the best (lowest rank) merge.
		bestRank := len(t.merges)
		bestIdx := -1
		for i := 0; i < len(tokens)-1; i++ {
			key := tokens[i] + " " + tokens[i+1]
			if rank, ok := mergeRank[key]; ok && rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}

		if bestIdx < 0 {
			break // no more applicable merges
		}

		// Apply the merge.
		merged := tokens[bestIdx] + tokens[bestIdx+1]
		newTokens := make([]string, 0, len(tokens)-1)
		newTokens = append(newTokens, tokens[:bestIdx]...)
		newTokens = append(newTokens, merged)
		newTokens = append(newTokens, tokens[bestIdx+2:]...)
		tokens = newTokens
	}

	// Convert token strings to IDs.
	ids := make([]int32, 0, len(tokens))
	for _, tok := range tokens {
		if id, ok := t.vocab[tok]; ok {
			ids = append(ids, id)
		} else {
			// Unknown token — try byte fallback.
			for _, b := range []byte(tok) {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				if id, ok := t.vocab[byteToken]; ok {
					ids = append(ids, id)
				}
				// If still unknown, skip (lossy).
			}
		}
	}

	return ids, nil
}

// Decode converts token IDs back to text.
func (t *HFTokenizer) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.invVocab) {
			continue
		}
		sb.WriteString(t.invVocab[id])
	}
	return sb.String(), nil
}

func (t *HFTokenizer) VocabSize() int   { return t.vocabSize }
func (t *HFTokenizer) EOSTokenID() int32 { return t.eosID }

// SpecialTokenIDs returns a sorted list of special token IDs for filtering.
func (t *HFTokenizer) SpecialTokenIDs() []int32 {
	var ids []int32
	if t.eosID >= 0 {
		ids = append(ids, t.eosID)
	}
	if t.bosID >= 0 {
		ids = append(ids, t.bosID)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// Compile-time check.
var _ Tokenizer = (*HFTokenizer)(nil)
