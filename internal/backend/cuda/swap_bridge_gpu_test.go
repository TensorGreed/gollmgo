//go:build gpu

package cuda

import (
	"bytes"
	"context"
	"testing"
)

func TestCUDARunnerWithModelKVSwapRoundTrip(t *testing.T) {
	runner, err := New(0)
	if err != nil {
		t.Fatalf("create runner: %v", err)
	}
	defer runner.Close()

	cache, err := NewCUDAKVCache(runner, 1, 8, 1, 2, 2)
	if err != nil {
		t.Fatalf("create KV cache: %v", err)
	}
	defer cache.Close()

	fullRunner := &CUDARunnerWithModel{
		CUDARunner: runner,
		KVCache:    cache,
	}
	if !fullRunner.Capabilities().KVSwap {
		t.Fatal("expected KVSwap capability when full runner has a KV cache")
	}

	k := []uint16{1, 2, 3, 4}
	v := []uint16{5, 6, 7, 8}
	slotMapping := []int32{0, 1}
	if err := cache.Write(k, v, slotMapping, len(slotMapping)); err != nil {
		t.Fatalf("seed cache: %v", err)
	}

	ctx := context.Background()
	origSnap, err := fullRunner.SnapshotKV(ctx, []int32{0})
	if err != nil {
		t.Fatalf("snapshot original block: %v", err)
	}
	defer origSnap.Release()

	hostOrig, ok := origSnap.(*kvHostSnapshot)
	if !ok {
		t.Fatalf("expected kvHostSnapshot, got %T", origSnap)
	}
	if hostOrig.BytesOnHost() == 0 {
		t.Fatal("expected non-empty host snapshot")
	}

	if err := fullRunner.RestoreKV(ctx, origSnap, []int32{1}); err != nil {
		t.Fatalf("restore block: %v", err)
	}

	restoredSnap, err := fullRunner.SnapshotKV(ctx, []int32{1})
	if err != nil {
		t.Fatalf("snapshot restored block: %v", err)
	}
	defer restoredSnap.Release()

	hostRestored, ok := restoredSnap.(*kvHostSnapshot)
	if !ok {
		t.Fatalf("expected kvHostSnapshot, got %T", restoredSnap)
	}
	if !bytes.Equal(hostOrig.data, hostRestored.data) {
		t.Fatal("restored KV bytes do not match original snapshot")
	}
}
