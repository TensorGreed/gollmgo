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
	addBOS    bool // prepend BOS token to every encode
}

type mergePair struct {
	a, b string
}

// hfTokenizerJSON is the top-level structure of tokenizer.json.
type hfTokenizerJSON struct {
	Model struct {
		Type   string           `json:"type"`
		Vocab  map[string]int32 `json:"vocab"`
		Merges []string         `json:"merges"`
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
		addBOS:    bosID >= 0, // prepend BOS if available
	}, nil
}

// byteToUnicode builds the GPT-2 / LLaMA byte-level BPE mapping.
// Bytes that are printable ASCII or Latin-1 supplement map to themselves.
// Other bytes map to U+0100 and up.
func byteToUnicode() [256]rune {
	var table [256]rune
	n := rune(0)
	for b := 0; b < 256; b++ {
		if (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			table[b] = rune(b)
		} else {
			table[b] = 256 + n
			n++
		}
	}
	return table
}

var b2u = byteToUnicode()

// unicodeToByte is the reverse of byteToUnicode.
func unicodeToByte() map[rune]byte {
	m := make(map[rune]byte, 256)
	for b := 0; b < 256; b++ {
		m[b2u[b]] = byte(b)
	}
	return m
}

var u2b = unicodeToByte()

// Encode tokenizes text using BPE.
func (t *HFTokenizer) Encode(text string) ([]int32, error) {
	if text == "" {
		return nil, nil
	}

	// SentencePiece-style normalization:
	// 1. Prepend ▁ to the entire input (word-initial marker).
	// 2. Replace all spaces with ▁.
	normalized := "▁" + strings.ReplaceAll(text, " ", "▁")

	tokens := make([]string, 0, len(normalized))
	for _, r := range normalized {
		tokens = append(tokens, string(r))
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

	// Prepend BOS token if configured.
	if t.addBOS && t.bosID >= 0 {
		ids = append([]int32{t.bosID}, ids...)
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
		// Skip BOS/EOS special tokens in decode output.
		if id == t.bosID || id == t.eosID {
			continue
		}
		sb.WriteString(t.invVocab[id])
	}
	// Post-process: ▁ represents space, <0xNN> represents raw bytes.
	raw := sb.String()

	// Replace byte tokens <0xNN> with actual bytes.
	var out strings.Builder
	i := 0
	for i < len(raw) {
		if i+6 <= len(raw) && raw[i] == '<' && raw[i+1] == '0' && raw[i+2] == 'x' && raw[i+5] == '>' {
			hex := raw[i+3 : i+5]
			var b byte
			fmt.Sscanf(hex, "%02X", &b)
			out.WriteByte(b)
			i += 6
		} else {
			out.WriteByte(raw[i])
			i++
		}
	}
	result := out.String()

	// SentencePiece: ▁ represents space.
	result = strings.ReplaceAll(result, "▁", " ")
	// Trim leading space (SentencePiece artifact).
	if len(result) > 0 && result[0] == ' ' {
		result = result[1:]
	}
	return result, nil
}

func (t *HFTokenizer) VocabSize() int    { return t.vocabSize }
func (t *HFTokenizer) BOSTokenID() int32 { return t.bosID }
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
