//go:build gpu

package cuda

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// TestPagedAttentionParity drives the v1-vs-v2 correctness sweep in
// kernels/gollmgo_paged_attn_test.cu via the Makefile target so a GPU run
// can gate on kernel parity without rebuilding the binary by hand.
//
// This is the Epic 2 M2 acceptance gate: long-context attention path
// matches v1 within tolerance across multiple shapes and GQA ratios.
// The actual numerical comparison happens inside the C++ harness; this
// test just runs it, relays pass/fail, and captures output on failure.
func TestPagedAttentionParity(t *testing.T) {
	// Walk up to the repo root so we can invoke `make` regardless of where
	// `go test` was invoked from.
	_, thisFile, _, _ := runtime.Caller(0)
	repoRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", ".."))

	cmd := exec.Command("make", "-C", repoRoot, "test-paged-attn-parity")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("parity sweep failed: %v\n---- output ----\n%s", err, string(out))
	}
	t.Logf("parity sweep output:\n%s", string(out))
}
