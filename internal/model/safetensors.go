package model

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// SafetensorsLoader extracts the JSON header of a .safetensors file (tensor
// names, shapes, dtypes, byte offsets). It is used by Loader callers that
// only need metadata. For end-to-end weight upload to GPU memory, see
// LoadSafetensorsWeights in weights.go — that's the path the serve command
// actually uses (cmd/gollmgo/gpu.go).
type SafetensorsLoader struct{}

// safetensorsHeader is the JSON metadata at the start of a .safetensors file.
// The format is: 8-byte LE uint64 header_size, then header_size bytes of JSON.
type safetensorsHeader struct {
	Metadata map[string]json.RawMessage `json:"__metadata__,omitempty"`
	// Remaining keys are tensor descriptors.
	Tensors map[string]safetensorsTensorInfo
}

type safetensorsTensorInfo struct {
	Dtype       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

func (l *SafetensorsLoader) Load(_ context.Context, path string) (*ModelMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: open %s: %w", path, err)
	}
	defer f.Close()

	// Read header size (first 8 bytes, little-endian uint64).
	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("safetensors: read header size: %w", err)
	}

	// Sanity check: headers over 100MB are likely corrupt.
	if headerSize > 100*1024*1024 {
		return nil, fmt.Errorf("safetensors: header too large (%d bytes)", headerSize)
	}

	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}

	// Parse as generic JSON map (tensor name -> info).
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	meta := &ModelMeta{
		Dtype: "unknown",
	}

	// Infer model shape from known tensor patterns.
	for name, data := range raw {
		if name == "__metadata__" {
			// Try to extract model config from metadata.
			var mdMap map[string]string
			if json.Unmarshal(data, &mdMap) == nil {
				if v, ok := mdMap["model_type"]; ok {
					meta.Family = v
				}
			}
			continue
		}

		var info safetensorsTensorInfo
		if json.Unmarshal(data, &info) != nil {
			continue
		}

		// Use embedding layer to infer vocab size and hidden size.
		if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
			if len(info.Shape) == 2 {
				if name == "model.embed_tokens.weight" {
					meta.VocabSize = info.Shape[0]
					meta.HiddenSize = info.Shape[1]
				}
			}
			if meta.Dtype == "unknown" {
				meta.Dtype = info.Dtype
			}
		}

		// Count layers by matching pattern "model.layers.N."
		// (heuristic: count unique layer indices)
	}

	// Count layers by scanning for highest layer index.
	meta.NumLayers = countLayers(raw)

	return meta, nil
}

func countLayers(raw map[string]json.RawMessage) int {
	maxLayer := -1
	for name := range raw {
		// Pattern: "model.layers.42.self_attn..."
		var idx int
		if n, _ := fmt.Sscanf(name, "model.layers.%d.", &idx); n == 1 {
			if idx > maxLayer {
				maxLayer = idx
			}
		}
	}
	if maxLayer < 0 {
		return 0
	}
	return maxLayer + 1
}
