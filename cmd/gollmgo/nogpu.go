//go:build !gpu

package main

import (
	"fmt"
	"log/slog"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/model"
)

func initGPURunner(_ *slog.Logger, modelPath, _ string, _ int) (backend.Runner, model.Tokenizer, error) {
	return nil, nil, fmt.Errorf("GPU support not compiled (build with -tags gpu). model path: %s", modelPath)
}
