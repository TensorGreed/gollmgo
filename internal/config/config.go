// Package config provides configuration loading and validation for gollmgo.
package config

import "strings"

// Config holds all server configuration.
type Config struct {
	// Server settings.
	Host string `json:"host"`
	Port int    `json:"port"`

	// Model settings.
	ModelPath     string `json:"model_path"`
	TokenizerPath string `json:"tokenizer_path"`

	// Scheduler settings.
	MaxBatchSize     int    `json:"max_batch_size"`
	MaxTokenBudget   int    `json:"max_token_budget"`
	MaxQueueDepth    int    `json:"max_queue_depth"`
	PrefillChunkSize int    `json:"prefill_chunk_size"`
	SchedulerPolicy  string `json:"scheduler_policy"` // "fcfs" (default), "sjf", "priority"
	PreemptMode      string `json:"preempt_mode"`     // "recompute" (default), "swap"
	AutoPreempt      bool   `json:"auto_preempt"`     // priority-only: high-priority arrivals preempt

	// KV cache settings.
	BlockSize            int     `json:"block_size"`
	MaxMemoryFraction    float64 `json:"max_memory_fraction"`
	PrefixCaching        bool    `json:"prefix_caching"`
	PrefixCacheMaxBlocks int     `json:"prefix_cache_max_blocks"`

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
	NGramSize      int     `json:"ngram_size"`       // default 3
	NumDraftTokens int     `json:"num_draft_tokens"` // K, default 4
	KillThreshold  float64 `json:"kill_threshold"`   // auto-disable below this acceptance rate
}

// DefaultConfig returns a Config with production-reasonable defaults.
func DefaultConfig() Config {
	return Config{
		Host:              "0.0.0.0",
		Port:              8080,
		MaxBatchSize:      64,
		MaxTokenBudget:    4096,
		MaxQueueDepth:     256,
		PrefillChunkSize:  512,
		SchedulerPolicy:   "fcfs",
		PreemptMode:       "recompute",
		BlockSize:         16,
		MaxMemoryFraction: 0.9,
		PrefixCaching:     false,
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
	if c.PrefixCacheMaxBlocks < 0 {
		return ErrInvalidPrefixCacheCap
	}
	switch strings.ToLower(strings.TrimSpace(c.SchedulerPolicy)) {
	case "", "fcfs", "sjf", "priority":
	default:
		return ErrInvalidSchedulerPolicy
	}
	if c.AutoPreempt && strings.ToLower(strings.TrimSpace(c.SchedulerPolicy)) != "priority" {
		return ErrAutoPreemptRequiresPrio
	}
	switch strings.ToLower(strings.TrimSpace(c.PreemptMode)) {
	case "", "recompute", "swap":
	default:
		return ErrInvalidPreemptMode
	}
	if c.Speculative.Enabled {
		switch strings.ToLower(strings.TrimSpace(c.Speculative.Mode)) {
		case "", "ngram":
			// draft mode is reserved for future work (needs a draft model).
		case "draft":
			// Accepted syntactically; engine may gate on runner capability.
		default:
			return ErrInvalidSpeculativeMode
		}
		if c.Speculative.NGramSize < 0 {
			return ErrInvalidNGramSize
		}
		if c.Speculative.NumDraftTokens < 0 {
			return ErrInvalidNumDraftTokens
		}
	}
	return nil
}
