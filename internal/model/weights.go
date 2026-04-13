package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// WeightTensor holds raw weight data and metadata for one tensor.
type WeightTensor struct {
	Name  string
	Dtype string
	Shape []int
	Data  []byte // raw bytes, in the file's native dtype
}

// LoadSafetensorsWeights reads all weight tensors from a safetensors file.
// Returns the list of tensors with their raw byte data.
func LoadSafetensorsWeights(path string) ([]WeightTensor, *ModelMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("safetensors: open %s: %w", path, err)
	}
	defer f.Close()

	// Read header size.
	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, nil, fmt.Errorf("safetensors: read header size: %w", err)
	}
	if headerSize > 100*1024*1024 {
		return nil, nil, fmt.Errorf("safetensors: header too large (%d bytes)", headerSize)
	}

	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, nil, fmt.Errorf("safetensors: read header: %w", err)
	}

	// Data starts right after the header.
	dataOffset := int64(8 + headerSize)

	// Parse header.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	meta := &ModelMeta{Dtype: "unknown"}
	var tensors []WeightTensor

	for name, data := range raw {
		if name == "__metadata__" {
			var mdMap map[string]string
			if json.Unmarshal(data, &mdMap) == nil {
				if v, ok := mdMap["model_type"]; ok {
					meta.Family = v
				}
			}
			continue
		}

		var info struct {
			Dtype       string  `json:"dtype"`
			Shape       []int   `json:"shape"`
			DataOffsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(data, &info); err != nil {
			continue
		}

		// Infer metadata from known tensors.
		if name == "model.embed_tokens.weight" && len(info.Shape) == 2 {
			meta.VocabSize = info.Shape[0]
			meta.HiddenSize = info.Shape[1]
			if meta.Dtype == "unknown" {
				meta.Dtype = info.Dtype
			}
		}

		size := info.DataOffsets[1] - info.DataOffsets[0]
		if size <= 0 {
			continue
		}

		// Read tensor data from file.
		tensorData := make([]byte, size)
		if _, err := f.ReadAt(tensorData, dataOffset+info.DataOffsets[0]); err != nil {
			return nil, nil, fmt.Errorf("safetensors: read tensor %q: %w", name, err)
		}

		tensors = append(tensors, WeightTensor{
			Name:  name,
			Dtype: info.Dtype,
			Shape: info.Shape,
			Data:  tensorData,
		})
	}

	meta.NumLayers = countLayersFromTensors(tensors)

	return tensors, meta, nil
}

func countLayersFromTensors(tensors []WeightTensor) int {
	maxLayer := -1
	for _, t := range tensors {
		var idx int
		if n, _ := fmt.Sscanf(t.Name, "model.layers.%d.", &idx); n == 1 {
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
