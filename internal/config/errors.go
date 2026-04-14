package config

import "errors"

var (
	ErrInvalidPort             = errors.New("config: port must be between 1 and 65535")
	ErrInvalidBatchSize        = errors.New("config: max_batch_size must be positive")
	ErrInvalidBlockSize        = errors.New("config: block_size must be positive")
	ErrInvalidMemoryFraction   = errors.New("config: max_memory_fraction must be in (0, 1]")
	ErrInvalidSchedulerPolicy  = errors.New("config: scheduler_policy must be one of fcfs, sjf, priority")
	ErrInvalidPreemptMode      = errors.New("config: preempt_mode must be recompute")
	ErrAutoPreemptRequiresPrio = errors.New("config: auto_preempt requires scheduler_policy=priority")
	ErrInvalidPrefixCacheCap   = errors.New("config: prefix_cache_max_blocks must be non-negative")
	ErrSpeculativeUnsupported  = errors.New("config: speculative decoding is not implemented in the serving path yet")
)
