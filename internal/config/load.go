package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// LoadFile reads and parses a JSON config file, applying defaults first.
func LoadFile(path string) (Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("config: read %s: %w", path, err)
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("config: parse %s: %w", path, err)
	}
	// Benchmark configs store server settings under a nested "server" object.
	// Overlay those fields onto the flat runtime config when present.
	var envelope struct {
		Server json.RawMessage `json:"server"`
	}
	if err := json.Unmarshal(data, &envelope); err == nil && len(envelope.Server) > 0 {
		if err := json.Unmarshal(envelope.Server, &cfg); err != nil {
			return cfg, fmt.Errorf("config: parse %s server section: %w", path, err)
		}
	}

	if err := cfg.Validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}
