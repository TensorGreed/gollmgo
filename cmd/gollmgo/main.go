// Package main is the CLI entrypoint for gollmgo.
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: gollmgo <command> [args]")
		fmt.Fprintln(os.Stderr, "commands: serve, bench, doctor")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		fmt.Println("gollmgo serve: not yet implemented")
	case "bench":
		fmt.Println("gollmgo bench: not yet implemented")
	case "doctor":
		fmt.Println("gollmgo doctor: not yet implemented")
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}
