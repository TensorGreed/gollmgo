// Package hfhub resolves model specifications — either HuggingFace Hub
// repository IDs or local filesystem paths — into a local directory the
// rest of gollmgo can load from. It caches downloads under
// $GOLLMGO_CACHE_DIR (default ~/.cache/gollmgo/hub).
package hfhub

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Spec identifies a model to load.
type Spec struct {
	// IsLocal is true when the input was a local filesystem path.
	IsLocal bool
	// LocalPath is the resolved local path (file or directory). Set only
	// when IsLocal is true.
	LocalPath string
	// Repo is the HuggingFace repo id ("org/model"). Set only when
	// IsLocal is false.
	Repo string
	// Revision is the branch/tag/commit to fetch. Defaults to "main".
	Revision string
}

// ParseSpec classifies an input string as either a local path or an HF
// repo id, optionally with a "@revision" suffix. Rules:
//
//  1. Empty input → error.
//  2. Starts with "/", "./", "../", or "~" → local path.
//  3. Starts with a drive letter (Windows) → local path.
//  4. Exists on disk → local path.
//  5. Matches "owner/repo[@revision]" with exactly one '/' separator
//     (and no extra slashes inside owner/repo) → HF repo.
//
// Examples:
//
//	/models/Llama-3-8B                     → local dir
//	./weights/model.safetensors            → local file
//	meta-llama/Llama-3.1-8B-Instruct       → HF repo, revision "main"
//	meta-llama/Llama-3.1-8B-Instruct@v0.1  → HF repo, revision "v0.1"
func ParseSpec(input string) (Spec, error) {
	s := strings.TrimSpace(input)
	if s == "" {
		return Spec{}, fmt.Errorf("hfhub: empty spec")
	}

	// Explicit local-path prefixes win.
	if strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./") ||
		strings.HasPrefix(s, "../") || strings.HasPrefix(s, "~") {
		return Spec{IsLocal: true, LocalPath: expandHome(s)}, nil
	}
	if len(s) >= 2 && s[1] == ':' { // windows drive
		return Spec{IsLocal: true, LocalPath: s}, nil
	}

	// If the string maps to an existing file or directory, treat as local.
	if _, err := os.Stat(s); err == nil {
		return Spec{IsLocal: true, LocalPath: s}, nil
	}

	// Split off "@revision" suffix if present.
	repo, rev := s, "main"
	if i := strings.LastIndex(s, "@"); i > 0 {
		repo = s[:i]
		rev = s[i+1:]
		if rev == "" {
			return Spec{}, fmt.Errorf("hfhub: empty revision in %q", s)
		}
	}

	// HF repo must look like "owner/name" — exactly one slash, no spaces,
	// neither side empty, no path traversal.
	parts := strings.Split(repo, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return Spec{}, fmt.Errorf("hfhub: %q is not a local path and not a valid HF repo id (expected owner/name)", s)
	}
	if strings.ContainsAny(repo, " \t\\") {
		return Spec{}, fmt.Errorf("hfhub: invalid characters in repo id %q", repo)
	}

	return Spec{Repo: repo, Revision: rev}, nil
}

// expandHome replaces a leading "~" with $HOME.
func expandHome(p string) string {
	if !strings.HasPrefix(p, "~") {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return p
	}
	if p == "~" {
		return home
	}
	if strings.HasPrefix(p, "~/") {
		return filepath.Join(home, p[2:])
	}
	return p
}
