//go:build gpu

package cuda

import (
	"context"
	"testing"

	"github.com/TensorGreed/gollmgo/internal/backend"
)

func TestCUDARunnerLifecycle(t *testing.T) {
	runner, err := New(0)
	if err != nil {
		t.Fatalf("create failed: %v", err)
	}
	defer runner.Close()

	info := runner.DeviceInfo()
	if info.Name == "" {
		t.Fatal("expected non-empty device name")
	}
	t.Logf("device: %s (compute %d.%d, %.1f GB total, %.1f GB free)",
		info.Name, info.ComputeMajor, info.ComputeMinor,
		float64(info.TotalMemoryBytes)/(1<<30),
		float64(info.FreeMemoryBytes)/(1<<30))

	err = runner.Warmup(context.Background(), backend.WarmupProfile{
		MaxBatchSize: 32,
		MaxSeqLen:    2048,
		BlockSize:    16,
	})
	if err != nil {
		t.Fatalf("warmup failed: %v", err)
	}

	caps := runner.Capabilities()
	if !caps.FP16 {
		t.Fatal("expected FP16 capability")
	}
}
