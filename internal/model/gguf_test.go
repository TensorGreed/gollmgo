package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
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
		"general.architecture":          "llama",
		"general.name":                  "test-llama-7b",
		"llama.block_count":             uint32(32),
		"llama.embedding_length":        uint32(4096),
		"llama.feed_forward_length":     uint32(11008),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(32),
		"llama.context_length":          uint32(2048),
		"tokenizer.ggml.bos_token_id":   uint32(1),
		"tokenizer.ggml.eos_token_id":   uint32(2),
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
	if meta.IntermediateSize != 11008 {
		t.Fatalf("expected intermediate 11008, got %d", meta.IntermediateSize)
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
	if meta.BOSTokenID != 1 || meta.EOSTokenID != 2 {
		t.Fatalf("expected BOS/EOS 1/2, got %d/%d", meta.BOSTokenID, meta.EOSTokenID)
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

// writeGGUFv3WithTensors builds a GGUF v3 file with metadata and F32 tensor data.
func writeGGUFv3WithTensors(t *testing.T, kv map[string]any, tensors []struct {
	name  string
	shape []uint64
	dtype uint32
	data  []byte
}) string {
	t.Helper()
	var buf bytes.Buffer

	binary.Write(&buf, binary.LittleEndian, uint32(ggufMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(len(tensors)))
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

	// Tensor descriptors — offsets relative to data section start.
	offset := uint64(0)
	for _, tensor := range tensors {
		writeGGUFTestString(&buf, tensor.name)
		binary.Write(&buf, binary.LittleEndian, uint32(len(tensor.shape)))
		for _, d := range tensor.shape {
			binary.Write(&buf, binary.LittleEndian, d)
		}
		binary.Write(&buf, binary.LittleEndian, tensor.dtype)
		binary.Write(&buf, binary.LittleEndian, offset)
		offset += uint64(len(tensor.data))
	}

	// Align to 32 bytes.
	for buf.Len()%32 != 0 {
		buf.WriteByte(0)
	}

	// Tensor data.
	for _, tensor := range tensors {
		buf.Write(tensor.data)
	}

	path := t.TempDir() + "/model.gguf"
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadGGUFWeights(t *testing.T) {
	// Create a minimal GGUF file with 2 F32 tensors.
	tensorData1 := make([]byte, 4*4)   // [4] F32
	tensorData2 := make([]byte, 2*3*4) // [2,3] F32
	// Write known values.
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(tensorData1[i*4:], math.Float32bits(float32(i+1)))
	}
	for i := 0; i < 6; i++ {
		binary.LittleEndian.PutUint32(tensorData2[i*4:], math.Float32bits(float32(10+i)))
	}

	path := writeGGUFv3WithTensors(t, map[string]any{
		"general.architecture":          "llama",
		"general.name":                  "test",
		"llama.block_count":             uint32(2),
		"llama.embedding_length":        uint32(64),
		"llama.attention.head_count":    uint32(4),
		"llama.attention.head_count_kv": uint32(4),
		"llama.context_length":          uint32(512),
	}, []struct {
		name  string
		shape []uint64
		dtype uint32
		data  []byte
	}{
		{name: "weight_a", shape: []uint64{4}, dtype: ggufDTypeF32, data: tensorData1},
		{name: "weight_b", shape: []uint64{2, 3}, dtype: ggufDTypeF32, data: tensorData2},
	})

	tensors, meta, err := LoadGGUFWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.Family != "llama" {
		t.Fatalf("expected llama, got %q", meta.Family)
	}
	if meta.NumLayers != 2 {
		t.Fatalf("expected 2 layers, got %d", meta.NumLayers)
	}
	if len(tensors) != 2 {
		t.Fatalf("expected 2 tensors, got %d", len(tensors))
	}

	// Verify first tensor.
	if tensors[0].Name != "weight_a" {
		t.Fatalf("expected weight_a, got %q", tensors[0].Name)
	}
	if tensors[0].Dtype != "F32" {
		t.Fatalf("expected F32 dtype, got %q", tensors[0].Dtype)
	}
	if len(tensors[0].Data) != 16 {
		t.Fatalf("expected 16 bytes, got %d", len(tensors[0].Data))
	}
	v := math.Float32frombits(binary.LittleEndian.Uint32(tensors[0].Data[0:4]))
	if v != 1.0 {
		t.Fatalf("expected first value 1.0, got %f", v)
	}

	// Verify second tensor.
	if tensors[1].Name != "weight_b" {
		t.Fatalf("expected weight_b, got %q", tensors[1].Name)
	}
	if len(tensors[1].Shape) != 2 || tensors[1].Shape[0] != 2 || tensors[1].Shape[1] != 3 {
		t.Fatalf("expected shape [2,3], got %v", tensors[1].Shape)
	}
}

func TestLoadGGUFWeightsQuantizedError(t *testing.T) {
	// Q4_0 tensors should be rejected.
	path := writeGGUFv3WithTensors(t, map[string]any{
		"general.architecture": "llama",
	}, []struct {
		name  string
		shape []uint64
		dtype uint32
		data  []byte
	}{
		{name: "quant_weight", shape: []uint64{4}, dtype: ggufDTypeQ4_0, data: make([]byte, 20)},
	})

	_, _, err := LoadGGUFWeights(path)
	if err == nil {
		t.Fatal("expected error for quantized tensor")
	}
}

func TestNormalizeGGUFTensorName(t *testing.T) {
	cases := map[string]string{
		"token_embd.weight":        "model.embed_tokens.weight",
		"output_norm.weight":       "model.norm.weight",
		"output.weight":            "lm_head.weight",
		"blk.0.attn_norm.weight":   "model.layers.0.input_layernorm.weight",
		"blk.3.attn_q.weight":      "model.layers.3.self_attn.q_proj.weight",
		"blk.3.attn_k.weight":      "model.layers.3.self_attn.k_proj.weight",
		"blk.3.attn_v.weight":      "model.layers.3.self_attn.v_proj.weight",
		"blk.3.attn_output.weight": "model.layers.3.self_attn.o_proj.weight",
		"blk.3.ffn_norm.weight":    "model.layers.3.post_attention_layernorm.weight",
		"blk.3.ffn_gate.weight":    "model.layers.3.mlp.gate_proj.weight",
		"blk.3.ffn_up.weight":      "model.layers.3.mlp.up_proj.weight",
		"blk.3.ffn_down.weight":    "model.layers.3.mlp.down_proj.weight",
		"some.other.weight":        "some.other.weight",
	}
	for input, want := range cases {
		if got := normalizeGGUFTensorName(input); got != want {
			t.Fatalf("normalizeGGUFTensorName(%q)=%q, want %q", input, got, want)
		}
	}
}
