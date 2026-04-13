package benchcheck

import (
	"strings"
	"testing"
)

var testThresholds = []byte(`{
  "description": "test thresholds",
  "thresholds": {
    "throughput_tokens_per_sec": {
      "min": 0,
      "regression_pct": 5.0
    },
    "ttft_p50_ms": {
      "max": 15.0,
      "regression_pct": 20.0
    },
    "ttft_p99_ms": {
      "max": 50.0,
      "regression_pct": 20.0
    },
    "itl_p50_ms": {
      "max": 10.0,
      "regression_pct": 20.0
    },
    "itl_p99_ms": {
      "max": 30.0,
      "regression_pct": 20.0
    },
    "error_rate_pct": {
      "max": 1.0
    }
  }
}`)

func baseline() []byte {
	return []byte(`{"metrics":{"throughput_tokens_per_sec":1000,"ttft_p50_ms":10,"ttft_p99_ms":30,"itl_p50_ms":7,"itl_p99_ms":20,"error_rate_pct":0.5}}`)
}

func TestCheckAllPass(t *testing.T) {
	current := []byte(`{"metrics":{"throughput_tokens_per_sec":1000,"ttft_p50_ms":10,"ttft_p99_ms":30,"itl_p50_ms":7,"itl_p99_ms":20,"error_rate_pct":0.5}}`)
	report, err := CheckBytes(baseline(), current, testThresholds)
	if err != nil {
		t.Fatal(err)
	}
	if report.Overall != VerdictPass {
		t.Errorf("expected PASS, got %s\n%s", report.Overall, report.String())
	}
}

func TestCheckThroughputRegression(t *testing.T) {
	current := []byte(`{"metrics":{"throughput_tokens_per_sec":900,"ttft_p50_ms":10,"ttft_p99_ms":30,"itl_p50_ms":7,"itl_p99_ms":20,"error_rate_pct":0.5}}`)
	report, err := CheckBytes(baseline(), current, testThresholds)
	if err != nil {
		t.Fatal(err)
	}
	if report.Overall != VerdictFail {
		t.Errorf("expected FAIL, got %s\n%s", report.Overall, report.String())
	}
	for _, r := range report.Results {
		if r.Name == "throughput_tokens_per_sec" {
			if r.Verdict != VerdictFail {
				t.Errorf("expected throughput FAIL, got %s", r.Verdict)
			}
			return
		}
	}
	t.Error("throughput metric not found in results")
}

func TestCheckAbsoluteThresholdFail(t *testing.T) {
	current := []byte(`{"metrics":{"throughput_tokens_per_sec":1000,"ttft_p50_ms":20,"ttft_p99_ms":30,"itl_p50_ms":7,"itl_p99_ms":20,"error_rate_pct":0.5}}`)
	report, err := CheckBytes(baseline(), current, testThresholds)
	if err != nil {
		t.Fatal(err)
	}
	if report.Overall != VerdictFail {
		t.Errorf("expected FAIL, got %s\n%s", report.Overall, report.String())
	}
	for _, r := range report.Results {
		if r.Name == "ttft_p50_ms" {
			if r.Verdict != VerdictFail {
				t.Errorf("expected ttft_p50_ms FAIL, got %s", r.Verdict)
			}
			if !strings.Contains(r.Reason, "exceeds max") {
				t.Errorf("expected reason to mention exceeds max, got: %s", r.Reason)
			}
			return
		}
	}
	t.Error("ttft_p50_ms metric not found in results")
}

func TestCheckReportFormatting(t *testing.T) {
	current := []byte(`{"metrics":{"throughput_tokens_per_sec":1000,"ttft_p50_ms":10,"ttft_p99_ms":30,"itl_p50_ms":7,"itl_p99_ms":20,"error_rate_pct":0.5}}`)
	report, err := CheckBytes(baseline(), current, testThresholds)
	if err != nil {
		t.Fatal(err)
	}
	s := report.String()
	if !strings.Contains(s, "Regression Check Report") {
		t.Error("missing report header")
	}
	if !strings.Contains(s, "Overall: PASS") {
		t.Error("missing overall verdict")
	}
	if !strings.Contains(s, "Metric") {
		t.Error("missing table header")
	}
	if !strings.Contains(s, "throughput_tokens_per_sec") {
		t.Error("missing throughput metric in output")
	}
}
