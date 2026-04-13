package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"testing"
)

// writeGGUFv3File builds a minimal GGUF v3 file with the given metadata KVs.
func writeGGUFv3File(t *testing.T, kv map[string]any) string {
	t.Helper()
	var buf bytes.Buffer

	// Magic.
	binary.Write(&buf, binary.LittleEndian, uint32(ggufMagic))
	// Version 3.
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	// Tensor count = 0.
	binary.Write(&buf, binary.LittleEndian, uint64(0))
	// KV count.
	binary.Write(&buf, binary.LittleEndian, uint64(len(kv)))

	for key, val := range kv {
		writeGGUFTestString(&buf, key)
		switch v := val.(type) {
		case string:
			binary.Write(&buf, binary.LittleEndian, uint32(ggufTypeString))
			writeGGUFTestString(&buf, v)
		case uint32:
			binary.Write(&buf, binary.LittleEndian, uint32(ggufTypeUint32))
			binary.Write(&buf, binary.LittleEndian, v)
		}
	}

	path := t.TempDir() + "/test.gguf"
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeGGUFTestString(buf *bytes.Buffer, s string) {
	binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
}

func TestGGUFLoaderParsesMetadata(t *testing.T) {
	path := writeGGUFv3File(t, map[string]any{
		"general.architecture":             "llama",
		"general.name":                     "test-llama-7b",
		"llama.block_count":                uint32(32),
		"llama.embedding_length":           uint32(4096),
		"llama.attention.head_count":       uint32(32),
		"llama.attention.head_count_kv":    uint32(32),
		"llama.context_length":             uint32(2048),
	})

	loader := &GGUFLoader{}
	meta, err := loader.Load(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}

	if meta.Family != "llama" {
		t.Fatalf("expected family llama, got %q", meta.Family)
	}
	if meta.Name != "test-llama-7b" {
		t.Fatalf("expected name test-llama-7b, got %q", meta.Name)
	}
	if meta.NumLayers != 32 {
		t.Fatalf("expected 32 layers, got %d", meta.NumLayers)
	}
	if meta.HiddenSize != 4096 {
		t.Fatalf("expected hidden 4096, got %d", meta.HiddenSize)
	}
	if meta.NumHeads != 32 {
		t.Fatalf("expected 32 heads, got %d", meta.NumHeads)
	}
	if meta.NumKVHeads != 32 {
		t.Fatalf("expected 32 KV heads, got %d", meta.NumKVHeads)
	}
	if meta.MaxSeqLen != 2048 {
		t.Fatalf("expected max seq 2048, got %d", meta.MaxSeqLen)
	}
}

func TestGGUFLoaderBadMagic(t *testing.T) {
	path := t.TempDir() + "/bad.gguf"
	os.WriteFile(path, []byte("not a gguf file at all!!!!"), 0644)

	loader := &GGUFLoader{}
	_, err := loader.Load(context.Background(), path)
	if err == nil {
		t.Fatal("expected error for bad magic")
	}
}

func TestGGUFLoaderMissingFile(t *testing.T) {
	loader := &GGUFLoader{}
	_, err := loader.Load(context.Background(), "/nonexistent/model.gguf")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}
