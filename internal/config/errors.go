package config

import "errors"

var (
	ErrInvalidPort             = errors.New("config: port must be between 1 and 65535")
	ErrInvalidBatchSize        = errors.New("config: max_batch_size must be positive")
	ErrInvalidBlockSize        = errors.New("config: block_size must be positive")
	ErrInvalidMemoryFraction   = errors.New("config: max_memory_fraction must be in (0, 1]")
	ErrInvalidSchedulerPolicy  = errors.New("config: scheduler_policy must be one of fcfs, sjf, priority")
	ErrInvalidPreemptMode      = errors.New("config: preempt_mode must be one of recompute, swap")
	ErrAutoPreemptRequiresPrio = errors.New("config: auto_preempt requires scheduler_policy=priority")
	ErrInvalidPrefixCacheCap   = errors.New("config: prefix_cache_max_blocks must be non-negative")
	ErrInvalidSpeculativeMode  = errors.New("config: speculative.mode must be \"ngram\"")
	ErrInvalidNGramSize        = errors.New("config: speculative.ngram_size must be >= 2")
	ErrInvalidNumDraftTokens   = errors.New("config: speculative.num_draft_tokens must be positive")
)
