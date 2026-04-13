//go:build !gpu

package main

import (
	"fmt"
	"log/slog"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/config"
	"github.com/TensorGreed/gollmgo/internal/model"
)

func initGPURunner(_ *slog.Logger, cfg config.Config, _ int) (backend.Runner, model.Tokenizer, int, error) {
	return nil, nil, 0, fmt.Errorf("GPU support not compiled (build with -tags gpu). model path: %s", cfg.ModelPath)
}
