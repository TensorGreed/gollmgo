package model

import (
	"encoding/json"
	"os"
	"testing"
)

func writeTestTokenizerJSON(t *testing.T) string {
	t.Helper()
	// Minimal BPE tokenizer with a small vocab.
	// "hello" with these merges:
	// h e l l o -> "he" l l o -> "he" "ll" o -> "he" "llo" -> "hello"
	data := map[string]any{
		"model": map[string]any{
			"type": "BPE",
			"vocab": map[string]int{
				"h":     0,
				"e":     1,
				"l":     2,
				"o":     3,
				"he":    4,
				"ll":    5,
				"llo":   6,
				"hello": 7,
			},
			"merges": []string{
				"h e",    // rank 0: h+e -> he
				"l l",    // rank 1: l+l -> ll
				"ll o",   // rank 2: ll+o -> llo
				"he llo", // rank 3: he+llo -> hello
			},
		},
		"added_tokens": []map[string]any{
			{"id": 8, "content": "<eos>", "special": true},
			{"id": 9, "content": "<bos>", "special": true},
		},
	}

	path := t.TempDir() + "/tokenizer.json"
	raw, _ := json.Marshal(data)
	os.WriteFile(path, raw, 0644)
	return path
}

func TestHFTokenizerLoad(t *testing.T) {
	path := writeTestTokenizerJSON(t)
	tok, err := LoadHFTokenizer(path, "<eos>", "<bos>")
	if err != nil {
		t.Fatal(err)
	}

	if tok.VocabSize() != 10 { // ids 0..9
		t.Fatalf("expected vocab 10, got %d", tok.VocabSize())
	}
	if tok.EOSTokenID() != 8 {
		t.Fatalf("expected EOS=8, got %d", tok.EOSTokenID())
	}
}

func TestHFTokenizerEncodeDecode(t *testing.T) {
	path := writeTestTokenizerJSON(t)
	tok, _ := LoadHFTokenizer(path, "<eos>", "<bos>")

	// "hello" should merge fully to token 8.
	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != 7 {
		t.Fatalf("expected [7] for 'hello', got %v", ids)
	}

	// Decode back.
	text, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("expected 'hello', got %q", text)
	}
}

func TestHFTokenizerPartialMerge(t *testing.T) {
	path := writeTestTokenizerJSON(t)
	tok, _ := LoadHFTokenizer(path, "<eos>", "<bos>")

	// "helo" = h+e+l+o -> "he"+l+o (only "h e" merge applies)
	ids, err := tok.Encode("helo")
	if err != nil {
		t.Fatal(err)
	}
	// "h"+"e" -> "he", then "l" and "o" stay as-is => [4, 2, 3]
	if len(ids) != 3 || ids[0] != 4 || ids[1] != 2 || ids[2] != 3 {
		t.Fatalf("expected [4, 2, 3] for 'helo', got %v", ids)
	}
}

func TestHFTokenizerEmpty(t *testing.T) {
	path := writeTestTokenizerJSON(t)
	tok, _ := LoadHFTokenizer(path, "<eos>", "<bos>")

	ids, _ := tok.Encode("")
	if ids != nil {
		t.Fatalf("expected nil for empty, got %v", ids)
	}
}

func TestHFTokenizerMissingFile(t *testing.T) {
	_, err := LoadHFTokenizer("/nonexistent/tokenizer.json", "<eos>", "<bos>")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}
