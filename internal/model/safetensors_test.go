package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"testing"
)

func TestSafetensorsLoaderParsesHeader(t *testing.T) {
	// Build a minimal safetensors file with a header only.
	header := map[string]any{
		"__metadata__": map[string]string{
			"model_type": "llama",
		},
		"model.embed_tokens.weight": map[string]any{
			"dtype":        "F16",
			"shape":        []int{32000, 4096},
			"data_offsets": []int{0, 262144000},
		},
		"model.layers.0.self_attn.q_proj.weight": map[string]any{
			"dtype":        "F16",
			"shape":        []int{4096, 4096},
			"data_offsets": []int{0, 0},
		},
		"model.layers.31.self_attn.q_proj.weight": map[string]any{
			"dtype":        "F16",
			"shape":        []int{4096, 4096},
			"data_offsets": []int{0, 0},
		},
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON)))
	buf.Write(headerJSON)

	tmpFile := t.TempDir() + "/test.safetensors"
	if err := os.WriteFile(tmpFile, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	loader := &SafetensorsLoader{}
	meta, err := loader.Load(context.Background(), tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	if meta.Family != "llama" {
		t.Fatalf("expected family llama, got %q", meta.Family)
	}
	if meta.VocabSize != 32000 {
		t.Fatalf("expected vocab 32000, got %d", meta.VocabSize)
	}
	if meta.HiddenSize != 4096 {
		t.Fatalf("expected hidden 4096, got %d", meta.HiddenSize)
	}
	if meta.NumLayers != 32 {
		t.Fatalf("expected 32 layers, got %d", meta.NumLayers)
	}
	if meta.Dtype != "F16" {
		t.Fatalf("expected dtype F16, got %q", meta.Dtype)
	}
}

func TestSafetensorsLoaderMissingFile(t *testing.T) {
	loader := &SafetensorsLoader{}
	_, err := loader.Load(context.Background(), "/nonexistent/file.safetensors")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}
