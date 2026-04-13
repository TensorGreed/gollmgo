package config

import "errors"

var (
	ErrInvalidPort           = errors.New("config: port must be between 1 and 65535")
	ErrInvalidBatchSize      = errors.New("config: max_batch_size must be positive")
	ErrInvalidBlockSize      = errors.New("config: block_size must be positive")
	ErrInvalidMemoryFraction = errors.New("config: max_memory_fraction must be in (0, 1]")
)
