// Package config provides configuration loading and validation for gollmgo.
package config

// Config holds all server configuration.
type Config struct {
	// Server settings.
	Host string `json:"host"`
	Port int    `json:"port"`

	// Model settings.
	ModelPath     string `json:"model_path"`
	TokenizerPath string `json:"tokenizer_path"`

	// Scheduler settings.
	MaxBatchSize   int `json:"max_batch_size"`
	MaxTokenBudget int `json:"max_token_budget"`
	MaxQueueDepth  int `json:"max_queue_depth"`

	// KV cache settings.
	BlockSize         int     `json:"block_size"`
	MaxMemoryFraction float64 `json:"max_memory_fraction"`

	// Quantization: "" (none), "fp8", "int8".
	Quantization string `json:"quantization"`

	// Speculative decoding.
	Speculative SpeculativeConfig `json:"speculative"`

	// Logging.
	LogLevel string `json:"log_level"`
}

// SpeculativeConfig controls speculative decoding behavior.
type SpeculativeConfig struct {
	Enabled        bool    `json:"enabled"`
	Mode           string  `json:"mode"`             // "ngram" or "draft"
	NGramSize      int     `json:"ngram_size"`        // default 3
	NumDraftTokens int     `json:"num_draft_tokens"`  // K, default 4
	KillThreshold  float64 `json:"kill_threshold"`    // auto-disable below this acceptance rate
}

// DefaultConfig returns a Config with production-reasonable defaults.
func DefaultConfig() Config {
	return Config{
		Host:              "0.0.0.0",
		Port:              8080,
		MaxBatchSize:      64,
		MaxTokenBudget:    4096,
		MaxQueueDepth:     256,
		BlockSize:         16,
		MaxMemoryFraction: 0.9,
		LogLevel:          "info",
	}
}

// Validate checks that the config is internally consistent.
func (c *Config) Validate() error {
	if c.Port <= 0 || c.Port > 65535 {
		return ErrInvalidPort
	}
	if c.MaxBatchSize <= 0 {
		return ErrInvalidBatchSize
	}
	if c.BlockSize <= 0 {
		return ErrInvalidBlockSize
	}
	if c.MaxMemoryFraction <= 0 || c.MaxMemoryFraction > 1.0 {
		return ErrInvalidMemoryFraction
	}
	return nil
}
