package model

import (
	"encoding/json"
	"fmt"
	"os"
)

// HFConfig represents the fields from a HuggingFace config.json.
type HFConfig struct {
	Architectures      []string `json:"architectures"`
	HiddenSize         int      `json:"hidden_size"`
	IntermediateSize   int      `json:"intermediate_size"`
	NumHiddenLayers    int      `json:"num_hidden_layers"`
	NumAttentionHeads  int      `json:"num_attention_heads"`
	NumKeyValueHeads   int      `json:"num_key_value_heads"`
	VocabSize          int      `json:"vocab_size"`
	MaxPositionEmbeddings int   `json:"max_position_embeddings"`
	RMSNormEps         float64  `json:"rms_norm_eps"`
	RopeTheta          float64  `json:"rope_theta"`
	ModelType          string   `json:"model_type"`
	TorchDtype         string   `json:"torch_dtype"`
	BosTokenID         int      `json:"bos_token_id"`
	EosTokenID         int      `json:"eos_token_id"`
}

// LoadHFConfig reads a HuggingFace config.json file.
func LoadHFConfig(path string) (*HFConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("hf_config: read %s: %w", path, err)
	}
	var cfg HFConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("hf_config: parse %s: %w", path, err)
	}
	return &cfg, nil
}

// ToModelMeta converts HFConfig to a ModelMeta.
func (c *HFConfig) ToModelMeta() *ModelMeta {
	return &ModelMeta{
		Family:     c.ModelType,
		NumLayers:  c.NumHiddenLayers,
		HiddenSize: c.HiddenSize,
		NumHeads:   c.NumAttentionHeads,
		NumKVHeads: c.NumKeyValueHeads,
		VocabSize:  c.VocabSize,
		MaxSeqLen:  c.MaxPositionEmbeddings,
		Dtype:      c.TorchDtype,
	}
}
