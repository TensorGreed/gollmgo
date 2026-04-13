package model

import (
	"encoding/json"
	"os"
	"testing"
)

func buildTestTokenizerJSON(t *testing.T) string {
	t.Helper()
	// Minimal SentencePiece-style tokenizer: "▁hi" as a merged token
	data := map[string]any{
		"model": map[string]any{
			"type": "BPE",
			"vocab": map[string]int{
				"▁":   0,
				"h":   1,
				"i":   2,
				"▁h":  3,
				"▁hi": 4,
			},
			"merges": []string{
				"▁ h",  // rank 0: ▁+h -> ▁h
				"▁h i", // rank 1: ▁h+i -> ▁hi
			},
		},
		"added_tokens": []map[string]any{
			{"id": 5, "content": "</s>", "special": true},
		},
	}
	path := t.TempDir() + "/tokenizer.json"
	raw, _ := json.Marshal(data)
	os.WriteFile(path, raw, 0644)
	return path
}

func TestHFTokenizerLoad(t *testing.T) {
	path := buildTestTokenizerJSON(t)
	tok, err := LoadHFTokenizer(path, "</s>", "<s>")
	if err != nil {
		t.Fatal(err)
	}
	if tok.EOSTokenID() != 5 {
		t.Fatalf("expected EOS=5, got %d", tok.EOSTokenID())
	}
}

func TestHFTokenizerEncodeDecode(t *testing.T) {
	path := buildTestTokenizerJSON(t)
	tok, _ := LoadHFTokenizer(path, "</s>", "<s>")

	// "hi" -> prepend ▁ -> "▁hi" -> merge ▁+h -> ▁h, then ▁h+i -> ▁hi = token 4
	ids, err := tok.Encode("hi")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != 4 {
		t.Fatalf("expected [4] for 'hi', got %v", ids)
	}

	text, _ := tok.Decode(ids)
	if text != "hi" {
		t.Fatalf("expected 'hi', got %q", text)
	}
}

func TestHFTokenizerEmpty(t *testing.T) {
	path := buildTestTokenizerJSON(t)
	tok, _ := LoadHFTokenizer(path, "</s>", "<s>")
	ids, _ := tok.Encode("")
	if ids != nil {
		t.Fatalf("expected nil for empty, got %v", ids)
	}
}

func TestHFTokenizerMissingFile(t *testing.T) {
	_, err := LoadHFTokenizer("/nonexistent/tokenizer.json", "</s>", "<s>")
	if err == nil {
		t.Fatal("expected error")
	}
}

// Test tokenization matches HuggingFace reference.
func TestHFTokenizerReferenceMatch(t *testing.T) {
	path := "/home/anuragj/Desktop/GitHub/gollmgo/models/TinyLlama-1.1B-Chat-v1.0/tokenizer.json"
	if _, err := os.Stat(path); err != nil {
		t.Skip("TinyLlama tokenizer not available")
	}

	tok, _ := LoadHFTokenizer(path, "</s>", "<s>")
	ids, _ := tok.Encode("What is 2+2? ")
	t.Logf("Our IDs:  %v", ids)
	// HF produces [1, 1724, 338, 29871, 29906, 29974, 29906, 29973, 29871]
	// with BOS=1 prepended. Our tokenizer does not prepend BOS.
	expected := []int32{1724, 338, 29871, 29906, 29974, 29906, 29973, 29871}
	if len(ids) != len(expected) {
		t.Fatalf("length mismatch: got %d, want %d\nOur: %v\nHF:  %v", len(ids), len(expected), ids, expected)
	}
	for i := range ids {
		if ids[i] != expected[i] {
			t.Errorf("ids[%d] = %d, want %d", i, ids[i], expected[i])
		}
	}
}

// Test with the real TinyLlama tokenizer if available.
func TestHFTokenizerRealLLaMA(t *testing.T) {
	path := "/home/anuragj/Desktop/GitHub/gollmgo/models/TinyLlama-1.1B-Chat-v1.0/tokenizer.json"
	if _, err := os.Stat(path); err != nil {
		t.Skip("TinyLlama tokenizer not available")
	}

	tok, err := LoadHFTokenizer(path, "</s>", "<s>")
	if err != nil {
		t.Fatal(err)
	}

	if tok.VocabSize() != 32000 {
		t.Fatalf("expected vocab 32000, got %d", tok.VocabSize())
	}

	text := "Hello, world!"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 {
		t.Fatal("expected non-empty encoding")
	}
	t.Logf("'%s' -> %v (%d tokens)", text, ids, len(ids))

	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != text {
		t.Fatalf("round-trip failed: %q -> %v -> %q", text, ids, decoded)
	}
}
