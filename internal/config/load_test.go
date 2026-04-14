package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFileBenchmarkServerEnvelope(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bench_config.json")
	data := []byte(`{
		"benchmark": {
			"mode": "serving",
			"num_prompts": 10
		},
		"server": {
			"max_batch_size": 32,
			"max_token_budget": 2048,
			"max_queue_depth": 64,
			"prefill_chunk_size": 256,
			"scheduler_policy": "priority",
			"auto_preempt": true,
			"preempt_mode": "recompute",
			"block_size": 32,
			"max_memory_fraction": 0.8,
			"prefix_caching": true,
			"prefix_cache_max_blocks": 128
		}
	}`)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MaxBatchSize != 32 || cfg.PrefillChunkSize != 256 {
		t.Fatalf("expected nested server settings to be loaded, got batch=%d chunk=%d", cfg.MaxBatchSize, cfg.PrefillChunkSize)
	}
	if cfg.SchedulerPolicy != "priority" || !cfg.AutoPreempt {
		t.Fatalf("expected priority+auto_preempt from server envelope, got policy=%q auto=%v", cfg.SchedulerPolicy, cfg.AutoPreempt)
	}
	if !cfg.PrefixCaching || cfg.PrefixCacheMaxBlocks != 128 {
		t.Fatalf("expected prefix cache settings from server envelope, got enabled=%v cap=%d", cfg.PrefixCaching, cfg.PrefixCacheMaxBlocks)
	}
}
