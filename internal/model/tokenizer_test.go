package model

import "testing"

func TestByteLevelTokenizerRoundTrip(t *testing.T) {
	tok := NewByteLevelTokenizer(32000, 2)

	text := "Hello, world!"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != len(text) {
		t.Fatalf("expected %d tokens, got %d", len(text), len(ids))
	}

	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != text {
		t.Fatalf("expected %q, got %q", text, decoded)
	}
}

func TestByteLevelTokenizerEmpty(t *testing.T) {
	tok := NewByteLevelTokenizer(256, 0)

	ids, err := tok.Encode("")
	if err != nil {
		t.Fatal(err)
	}
	if ids != nil {
		t.Fatalf("expected nil for empty input, got %v", ids)
	}

	decoded, err := tok.Decode(nil)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != "" {
		t.Fatalf("expected empty string, got %q", decoded)
	}
}

func TestByteLevelTokenizerOutOfRange(t *testing.T) {
	tok := NewByteLevelTokenizer(256, 0)
	_, err := tok.Decode([]int32{300})
	if err == nil {
		t.Fatal("expected error for out-of-range token ID")
	}
}

func TestByteLevelTokenizerVocabAndEOS(t *testing.T) {
	tok := NewByteLevelTokenizer(32000, 2)
	if tok.VocabSize() != 32000 {
		t.Fatalf("expected vocab 32000, got %d", tok.VocabSize())
	}
	if tok.EOSTokenID() != 2 {
		t.Fatalf("expected EOS 2, got %d", tok.EOSTokenID())
	}
}

func TestByteLevelTokenizerMinVocab(t *testing.T) {
	tok := NewByteLevelTokenizer(10, 0) // should be bumped to 256
	if tok.VocabSize() != 256 {
		t.Fatalf("expected min vocab 256, got %d", tok.VocabSize())
	}
}
