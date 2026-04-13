// Command benchcheck runs regression checks comparing benchmark results against a baseline.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/TensorGreed/gollmgo/internal/benchcheck"
)

func main() {
	baselinePath := flag.String("baseline", "bench/baseline_result.json", "path to baseline result JSON")
	currentPath := flag.String("current", "", "path to current result JSON (required)")
	thresholdsPath := flag.String("thresholds", "bench/thresholds.json", "path to thresholds JSON")
	flag.Parse()

	if *currentPath == "" {
		fmt.Fprintln(os.Stderr, "error: --current is required")
		flag.Usage()
		os.Exit(1)
	}

	report, err := benchcheck.Check(*baselinePath, *currentPath, *thresholdsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Print(report.String())

	if report.Overall == benchcheck.VerdictFail {
		os.Exit(1)
	}
}
