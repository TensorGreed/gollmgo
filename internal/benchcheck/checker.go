// Package benchcheck compares benchmark results against baselines and thresholds.
package benchcheck

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"
)

// Verdict represents the outcome of a metric check.
type Verdict string

const (
	VerdictPass Verdict = "PASS"
	VerdictFail Verdict = "FAIL"
	VerdictWarn Verdict = "WARN"
)

// MetricResult holds the comparison result for a single metric.
type MetricResult struct {
	Name     string
	Baseline float64
	Current  float64
	DeltaPct float64
	Verdict  Verdict
	Reason   string
}

// Report holds the full regression check report.
type Report struct {
	Overall   Verdict
	Results   []MetricResult
	Timestamp time.Time
}

// String formats a human-readable table of the report.
func (r *Report) String() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Regression Check Report  %s\n", r.Timestamp.Format(time.RFC3339)))
	b.WriteString(fmt.Sprintf("Overall: %s\n\n", r.Overall))
	b.WriteString(fmt.Sprintf("%-30s %12s %12s %10s %8s  %s\n",
		"Metric", "Baseline", "Current", "Delta%", "Verdict", "Reason"))
	b.WriteString(strings.Repeat("-", 100) + "\n")
	for _, m := range r.Results {
		b.WriteString(fmt.Sprintf("%-30s %12.2f %12.2f %9.1f%% %8s  %s\n",
			m.Name, m.Baseline, m.Current, m.DeltaPct, m.Verdict, m.Reason))
	}
	return b.String()
}

// Thresholds is the top-level threshold config.
type Thresholds struct {
	Description string                       `json:"description"`
	Thresholds  map[string]MetricThreshold   `json:"thresholds"`
}

// MetricThreshold defines limits for a single metric.
type MetricThreshold struct {
	Min           *float64 `json:"min,omitempty"`
	Max           *float64 `json:"max,omitempty"`
	RegressionPct float64  `json:"regression_pct"`
	Description   string   `json:"description,omitempty"`
}

// BaselineResult is the JSON structure for baseline/current results.
type BaselineResult struct {
	Metrics map[string]float64 `json:"metrics"`
}

// Check loads files from disk and runs the comparison.
func Check(baselinePath, currentPath, thresholdsPath string) (*Report, error) {
	baselineData, err := os.ReadFile(baselinePath)
	if err != nil {
		return nil, fmt.Errorf("reading baseline: %w", err)
	}
	currentData, err := os.ReadFile(currentPath)
	if err != nil {
		return nil, fmt.Errorf("reading current: %w", err)
	}
	thresholdData, err := os.ReadFile(thresholdsPath)
	if err != nil {
		return nil, fmt.Errorf("reading thresholds: %w", err)
	}
	return CheckBytes(baselineData, currentData, thresholdData)
}

// CheckBytes performs the comparison from raw JSON bytes.
func CheckBytes(baselineData, currentData, thresholdData []byte) (*Report, error) {
	var baseline BaselineResult
	if err := json.Unmarshal(baselineData, &baseline); err != nil {
		return nil, fmt.Errorf("parsing baseline: %w", err)
	}
	var current BaselineResult
	if err := json.Unmarshal(currentData, &current); err != nil {
		return nil, fmt.Errorf("parsing current: %w", err)
	}
	var thresholds Thresholds
	if err := json.Unmarshal(thresholdData, &thresholds); err != nil {
		return nil, fmt.Errorf("parsing thresholds: %w", err)
	}

	report := &Report{
		Overall:   VerdictPass,
		Timestamp: time.Now().UTC(),
	}

	for name, thresh := range thresholds.Thresholds {
		baseVal := baseline.Metrics[name]
		curVal := current.Metrics[name]

		mr := MetricResult{
			Name:     name,
			Baseline: baseVal,
			Current:  curVal,
			Verdict:  VerdictPass,
		}

		// Compute delta percentage relative to baseline.
		if baseVal != 0 {
			mr.DeltaPct = ((curVal - baseVal) / math.Abs(baseVal)) * 100
		}

		// Check absolute max threshold.
		if thresh.Max != nil && curVal > *thresh.Max {
			mr.Verdict = VerdictFail
			mr.Reason = fmt.Sprintf("exceeds max %.2f", *thresh.Max)
		}

		// Check absolute min threshold.
		if thresh.Min != nil && curVal < *thresh.Min {
			mr.Verdict = VerdictFail
			mr.Reason = fmt.Sprintf("below min %.2f", *thresh.Min)
		}

		// Check regression percentage (only if baseline is nonzero).
		if baseVal != 0 && thresh.RegressionPct > 0 {
			// For "higher is better" metrics (has min or name contains "throughput"),
			// a drop is a regression. For "lower is better" metrics (has max),
			// an increase is a regression.
			isHigherBetter := thresh.Min != nil || strings.Contains(name, "throughput")

			var regressionDetected bool
			if isHigherBetter {
				// Regression if current is lower than baseline by more than threshold.
				regressionDetected = (baseVal-curVal)/math.Abs(baseVal)*100 > thresh.RegressionPct
			} else {
				// Regression if current is higher than baseline by more than threshold.
				regressionDetected = (curVal-baseVal)/math.Abs(baseVal)*100 > thresh.RegressionPct
			}

			if regressionDetected && mr.Verdict != VerdictFail {
				mr.Verdict = VerdictFail
				mr.Reason = fmt.Sprintf("regressed >%.1f%% vs baseline", thresh.RegressionPct)
			} else if regressionDetected {
				mr.Reason += fmt.Sprintf("; also regressed >%.1f%%", thresh.RegressionPct)
			}
		}

		if mr.Verdict == VerdictFail {
			report.Overall = VerdictFail
		}

		report.Results = append(report.Results, mr)
	}

	return report, nil
}
