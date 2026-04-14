package benchcheck

import (
	"strings"
	"testing"
)

var testThresholds = []byte(`{
  "description": "test thresholds",
  "thresholds": {
    "tokens_per_second": {
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

// baseline matches the BenchResult schema from cmd/gollmgo. 1000 tok/s,
// 10ms TTFT P50, 7ms ITL P50, 0.5% error rate (5 errors / 1000 prompts).
func baseline() []byte {
	return []byte(`{
		"placeholder": false,
		"num_prompts": 1000,
		"error_count": 5,
		"tokens_per_second": 1000,
		"ttft_p50_ms": 10, "ttft_p99_ms": 30,
		"itl_p50_ms": 7, "itl_p99_ms": 20
	}`)
}

func TestCheckAllPass(t *testing.T) {
	current := baseline()
	report, err := CheckBytes(baseline(), current, testThresholds)
	if err != nil {
		t.Fatal(err)
	}
	if report.Overall != VerdictPass {
		t.Errorf("expected PASS, got %s\n%s", report.Overall, report.String())
	}
}

func TestCheckThroughputRegression(t *testing.T) {
	// 900 tok/s vs baseline 1000 = 10% drop, exceeds 5% threshold.
	current := []byte(`{
		"num_prompts": 1000, "error_count": 5,
		"tokens_per_second": 900,
		"ttft_p50_ms": 10, "ttft_p99_ms": 30,
		"itl_p50_ms": 7, "itl_p99_ms": 20
	}`)
	report, err := CheckBytes(baseline(), current, testThresholds)
	if err != nil {
		t.Fatal(err)
	}
	if report.Overall != VerdictFail {
		t.Errorf("expected FAIL, got %s\n%s", report.Overall, report.String())
	}
	for _, r := range report.Results {
		if r.Name == "tokens_per_second" {
			if r.Verdict != VerdictFail {
				t.Errorf("expected tokens_per_second FAIL, got %s", r.Verdict)
			}
			return
		}
	}
	t.Error("tokens_per_second metric not found in results")
}

func TestCheckAbsoluteThresholdFail(t *testing.T) {
	// TTFT P50 of 20ms exceeds the 15ms cap.
	current := []byte(`{
		"num_prompts": 1000, "error_count": 5,
		"tokens_per_second": 1000,
		"ttft_p50_ms": 20, "ttft_p99_ms": 30,
		"itl_p50_ms": 7, "itl_p99_ms": 20
	}`)
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

func TestCheckErrorRateDerived(t *testing.T) {
	// 50 errors / 1000 prompts = 5% > 1% cap.
	current := []byte(`{
		"num_prompts": 1000, "error_count": 50,
		"tokens_per_second": 1000,
		"ttft_p50_ms": 10, "ttft_p99_ms": 30,
		"itl_p50_ms": 7, "itl_p99_ms": 20
	}`)
	report, err := CheckBytes(baseline(), current, testThresholds)
	if err != nil {
		t.Fatal(err)
	}
	if report.Overall != VerdictFail {
		t.Fatalf("expected FAIL on derived error_rate_pct, got %s\n%s", report.Overall, report.String())
	}
	var found bool
	for _, r := range report.Results {
		if r.Name == "error_rate_pct" {
			found = true
			if r.Current != 5 {
				t.Errorf("expected error_rate_pct=5, got %v", r.Current)
			}
			if r.Verdict != VerdictFail {
				t.Errorf("expected error_rate_pct FAIL, got %s", r.Verdict)
			}
		}
	}
	if !found {
		t.Error("error_rate_pct metric not in results")
	}
}

func TestCheckReportFormatting(t *testing.T) {
	current := baseline()
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
	if !strings.Contains(s, "tokens_per_second") {
		t.Error("missing throughput metric in output")
	}
}

func TestCheckRejectsPlaceholderBaseline(t *testing.T) {
	placeholder := []byte(`{
		"placeholder": true,
		"num_prompts": 1,
		"error_count": 0,
		"tokens_per_second": 1
	}`)
	_, err := CheckBytes(placeholder, baseline(), testThresholds)
	if err == nil || !strings.Contains(err.Error(), "placeholder") {
		t.Fatalf("expected placeholder baseline error, got %v", err)
	}
}

func TestCheckErrorRateUsesMeasuredRequestsWhenPresent(t *testing.T) {
	baseline := []byte(`{
		"placeholder": false,
		"num_prompts": 1000,
		"measured_requests": 3000,
		"error_count": 15,
		"tokens_per_second": 1000,
		"ttft_p50_ms": 10, "ttft_p99_ms": 30,
		"itl_p50_ms": 7, "itl_p99_ms": 20
	}`)
	current := []byte(`{
		"num_prompts": 1000,
		"measured_requests": 3000,
		"error_count": 60,
		"tokens_per_second": 1000,
		"ttft_p50_ms": 10, "ttft_p99_ms": 30,
		"itl_p50_ms": 7, "itl_p99_ms": 20
	}`)

	report, err := CheckBytes(baseline, current, testThresholds)
	if err != nil {
		t.Fatal(err)
	}

	var found bool
	for _, r := range report.Results {
		if r.Name == "error_rate_pct" {
			found = true
			if r.Current != 2 {
				t.Fatalf("expected error_rate_pct=2, got %v", r.Current)
			}
			if r.Verdict != VerdictFail {
				t.Fatalf("expected error_rate_pct FAIL, got %s", r.Verdict)
			}
		}
	}
	if !found {
		t.Fatal("error_rate_pct metric not found")
	}
}
