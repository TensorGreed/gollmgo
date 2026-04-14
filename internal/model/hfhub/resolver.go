package hfhub

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Endpoint is the HuggingFace Hub base URL. Overridable for tests and
// self-hosted mirrors (e.g. HF_ENDPOINT).
var Endpoint = "https://huggingface.co"

// Resolver fetches HF Hub repos to a local cache directory.
type Resolver struct {
	// CacheDir is the root of the local cache. A repo "org/name" is
	// cached under CacheDir/models--org--name/.
	CacheDir string
	// Token is the HF bearer token (for gated/private repos). Optional.
	Token string
	// HTTPClient is used for all requests. Defaults to a sensible client.
	HTTPClient *http.Client
	// Concurrency limits in-flight downloads. Default 4.
	Concurrency int
	// Log receives progress lines. Optional.
	Log *slog.Logger
}

// DefaultCacheDir returns the cache directory used when none is configured.
// Order: $GOLLMGO_CACHE_DIR, $XDG_CACHE_HOME/gollmgo/hub, ~/.cache/gollmgo/hub.
func DefaultCacheDir() string {
	if dir := os.Getenv("GOLLMGO_CACHE_DIR"); dir != "" {
		return dir
	}
	if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
		return filepath.Join(xdg, "gollmgo", "hub")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "gollmgo-hub")
	}
	return filepath.Join(home, ".cache", "gollmgo", "hub")
}

// TokenFromEnv returns the HF bearer token from common env vars, or "".
func TokenFromEnv() string {
	for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"} {
		if v := strings.TrimSpace(os.Getenv(k)); v != "" {
			return v
		}
	}
	return ""
}

// Handle describes a resolved model ready for loading.
type Handle struct {
	// RepoID is the HF repo ("org/name"), or "" for local specs.
	RepoID string
	// Revision is the fetched revision, or "" for local specs.
	Revision string
	// LocalDir is the directory containing the model files.
	LocalDir string
	// ConfigPath is "<LocalDir>/config.json" if it exists, else "".
	ConfigPath string
	// TokenizerPath points at tokenizer.json if present, else "".
	TokenizerPath string
	// WeightsFiles lists all .safetensors files in LocalDir (sorted).
	WeightsFiles []string
	// IsSharded is true when there are multiple safetensors shards.
	IsSharded bool
}

// Resolve ensures the model files for spec are present locally and returns
// a Handle describing them. For HF repos it downloads missing files to
// CacheDir and reuses anything already cached. For local specs it just
// inspects the filesystem and builds a Handle.
func (r *Resolver) Resolve(ctx context.Context, spec Spec) (*Handle, error) {
	if r.CacheDir == "" {
		r.CacheDir = DefaultCacheDir()
	}
	if r.Concurrency <= 0 {
		r.Concurrency = 4
	}
	if r.HTTPClient == nil {
		r.HTTPClient = &http.Client{Timeout: 30 * time.Minute}
	}
	if r.Log == nil {
		r.Log = slog.New(slog.NewTextHandler(io.Discard, nil))
	}

	if spec.IsLocal {
		return handleFromDir(spec.LocalPath)
	}

	dir, err := r.fetchRepo(ctx, spec.Repo, spec.Revision)
	if err != nil {
		return nil, err
	}
	h, err := handleFromDir(dir)
	if err != nil {
		return nil, err
	}
	h.RepoID = spec.Repo
	h.Revision = spec.Revision
	return h, nil
}

// handleFromDir builds a Handle from an existing local directory or file.
// Accepts either a directory (HF layout) or a single .safetensors file.
func handleFromDir(path string) (*Handle, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("hfhub: stat %s: %w", path, err)
	}
	// Single safetensors file → treat its directory as the model dir.
	if !info.IsDir() {
		dir := filepath.Dir(path)
		if !strings.HasSuffix(strings.ToLower(path), ".safetensors") {
			return nil, fmt.Errorf("hfhub: %s is not a directory or .safetensors file", path)
		}
		h := &Handle{LocalDir: dir, WeightsFiles: []string{path}}
		h.ConfigPath = ifExists(filepath.Join(dir, "config.json"))
		h.TokenizerPath = ifExists(filepath.Join(dir, "tokenizer.json"))
		return h, nil
	}

	h := &Handle{LocalDir: path}
	h.ConfigPath = ifExists(filepath.Join(path, "config.json"))
	h.TokenizerPath = ifExists(filepath.Join(path, "tokenizer.json"))

	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, fmt.Errorf("hfhub: readdir %s: %w", path, err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.HasSuffix(e.Name(), ".safetensors") {
			h.WeightsFiles = append(h.WeightsFiles, filepath.Join(path, e.Name()))
		}
	}
	if len(h.WeightsFiles) == 0 {
		return nil, fmt.Errorf("hfhub: no .safetensors files in %s", path)
	}
	h.IsSharded = len(h.WeightsFiles) > 1
	return h, nil
}

func ifExists(p string) string {
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return ""
}

// fetchRepo ensures all relevant files for repo@revision exist in the
// cache dir and returns that directory.
func (r *Resolver) fetchRepo(ctx context.Context, repo, revision string) (string, error) {
	dir := r.repoDir(repo)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("hfhub: mkdir %s: %w", dir, err)
	}

	files, err := r.listFiles(ctx, repo, revision)
	if err != nil {
		return "", err
	}

	// Filter to things we actually care about for serving.
	want := filterRelevant(files)
	if len(want) == 0 {
		return "", fmt.Errorf("hfhub: %s@%s has no safetensors / config / tokenizer files", repo, revision)
	}

	r.Log.Info("hfhub: resolving repo",
		"repo", repo, "revision", revision, "files", len(want), "cache", dir)

	// Download in parallel, bounded by Concurrency.
	sem := make(chan struct{}, r.Concurrency)
	var wg sync.WaitGroup
	errCh := make(chan error, len(want))

	for _, f := range want {
		f := f
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			dest := filepath.Join(dir, f.Path)
			if err := r.downloadOne(ctx, repo, revision, f, dest); err != nil {
				errCh <- fmt.Errorf("hfhub: download %s: %w", f.Path, err)
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return "", err
		}
	}
	return dir, nil
}

// repoDir returns the on-disk cache directory for repo. Format mirrors the
// HF Hub layout loosely ("models--org--name") so users can recognise it.
func (r *Resolver) repoDir(repo string) string {
	safe := strings.ReplaceAll(repo, "/", "--")
	return filepath.Join(r.CacheDir, "models--"+safe)
}

// treeEntry is one file from GET /api/models/<repo>/tree/<rev>.
type treeEntry struct {
	Type string `json:"type"` // "file" or "directory"
	Path string `json:"path"`
	Size int64  `json:"size"`
	OID  string `json:"oid"`
}

// listFiles walks the repo's file tree at revision. Recurses into
// subdirectories so nested tokenizer/config files aren't missed.
func (r *Resolver) listFiles(ctx context.Context, repo, revision string) ([]treeEntry, error) {
	var out []treeEntry
	seen := map[string]bool{}

	var walk func(subpath string) error
	walk = func(subpath string) error {
		u := fmt.Sprintf("%s/api/models/%s/tree/%s", Endpoint,
			url.PathEscape(repo), url.PathEscape(revision))
		if subpath != "" {
			u += "/" + url.PathEscape(subpath)
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			return err
		}
		r.authorize(req)
		resp, err := r.HTTPClient.Do(req)
		if err != nil {
			return fmt.Errorf("list tree: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
			return fmt.Errorf("list tree: %s — model may be gated; set HF_TOKEN", resp.Status)
		}
		if resp.StatusCode == http.StatusNotFound {
			return fmt.Errorf("list tree: %s — repo %q or revision %q not found", resp.Status, repo, revision)
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("list tree: %s", resp.Status)
		}
		var entries []treeEntry
		if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
			return fmt.Errorf("parse tree: %w", err)
		}
		for _, e := range entries {
			if e.Type == "directory" {
				if err := walk(e.Path); err != nil {
					return err
				}
				continue
			}
			if seen[e.Path] {
				continue
			}
			seen[e.Path] = true
			out = append(out, e)
		}
		return nil
	}
	if err := walk(""); err != nil {
		return nil, err
	}
	return out, nil
}

// filterRelevant keeps only files that our loaders will actually consume:
// config/tokenizer JSON, sentencepiece model, and safetensors weights.
// It deliberately skips PyTorch/flax/onnx duplicates that bloat downloads.
func filterRelevant(files []treeEntry) []treeEntry {
	keep := []treeEntry{}
	for _, f := range files {
		if f.Type != "file" {
			continue
		}
		name := filepath.Base(f.Path)
		low := strings.ToLower(name)
		switch {
		case strings.HasSuffix(low, ".safetensors"),
			strings.HasSuffix(low, ".json"),
			low == "tokenizer.model",
			low == "special_tokens_map.json":
			keep = append(keep, f)
		default:
			// Skip: .bin, .pt, .pth, .h5, .msgpack, .ot, .onnx, README, etc.
		}
	}
	return keep
}

// downloadOne fetches a single file, skipping the HTTP request if the cache
// already has a file of the expected size. It writes atomically to a .part
// file and renames on success.
func (r *Resolver) downloadOne(ctx context.Context, repo, revision string, f treeEntry, dest string) error {
	// Fast path: existing file with correct size.
	if info, err := os.Stat(dest); err == nil {
		if f.Size > 0 && info.Size() == f.Size {
			r.Log.Debug("hfhub: cache hit", "path", f.Path, "size", info.Size())
			return nil
		}
		r.Log.Info("hfhub: size mismatch, re-downloading", "path", f.Path,
			"have", info.Size(), "want", f.Size)
	}

	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}

	u := fmt.Sprintf("%s/%s/resolve/%s/%s", Endpoint,
		url.PathEscape(repo), url.PathEscape(revision),
		// Use url.PathEscape per path segment so nested paths survive.
		escapePath(f.Path))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	r.authorize(req)

	resp, err := r.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return fmt.Errorf("%s — gated? set HF_TOKEN", resp.Status)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", u, resp.Status)
	}

	start := time.Now()
	tmp := dest + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	n, err := io.Copy(out, resp.Body)
	cerr := out.Close()
	if err != nil {
		os.Remove(tmp)
		return err
	}
	if cerr != nil {
		os.Remove(tmp)
		return cerr
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return err
	}

	speedMBps := float64(n) / (1 << 20) / time.Since(start).Seconds()
	r.Log.Info("hfhub: downloaded",
		"path", f.Path,
		"size_mb", fmt.Sprintf("%.1f", float64(n)/(1<<20)),
		"mbps", fmt.Sprintf("%.1f", speedMBps))
	return nil
}

func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

func (r *Resolver) authorize(req *http.Request) {
	if r.Token != "" {
		req.Header.Set("Authorization", "Bearer "+r.Token)
	}
	req.Header.Set("User-Agent", "gollmgo/0.1 (+https://github.com/TensorGreed/gollmgo)")
}
