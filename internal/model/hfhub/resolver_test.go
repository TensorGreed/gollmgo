package hfhub

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// mockHFServer stands in for huggingface.co. It serves the tree API for
// a single repo and returns bodies from a files map.
type mockHFServer struct {
	mu       sync.Mutex
	repo     string
	revision string
	files    map[string][]byte // path → body (read-only after construction)
	tokenSet string            // if non-empty, 401 unless Authorization matches
	hits     map[string]int    // guarded by mu
	t        *testing.T
}

func newMockHFServer(t *testing.T, repo, rev string, files map[string][]byte) *httptest.Server {
	m := &mockHFServer{
		repo: repo, revision: rev, files: files,
		hits: map[string]int{},
		t:    t,
	}
	return httptest.NewServer(http.HandlerFunc(m.handle))
}

func (m *mockHFServer) handle(w http.ResponseWriter, r *http.Request) {
	if m.tokenSet != "" {
		if r.Header.Get("Authorization") != "Bearer "+m.tokenSet {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
	}
	m.mu.Lock()
	m.hits[r.URL.Path]++
	m.mu.Unlock()
	// Tree API.
	treePrefix := fmt.Sprintf("/api/models/%s/tree/%s", m.repo, m.revision)
	if strings.HasPrefix(r.URL.Path, treePrefix) {
		var entries []treeEntry
		for path, body := range m.files {
			entries = append(entries, treeEntry{
				Type: "file", Path: path, Size: int64(len(body)), OID: "abc",
			})
		}
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(entries)
		return
	}
	// Resolve/download API.
	resolvePrefix := fmt.Sprintf("/%s/resolve/%s/", m.repo, m.revision)
	if strings.HasPrefix(r.URL.Path, resolvePrefix) {
		path := strings.TrimPrefix(r.URL.Path, resolvePrefix)
		body, ok := m.files[path]
		if !ok {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		_, _ = w.Write(body)
		return
	}
	http.NotFound(w, r)
}

func testResolver(t *testing.T, srv *httptest.Server, tmp string) *Resolver {
	t.Helper()
	return &Resolver{
		CacheDir:    tmp,
		HTTPClient:  srv.Client(),
		Concurrency: 2,
		Log:         slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
}

func TestResolverDownloadsHFRepo(t *testing.T) {
	repo := "fake-org/fake-model"
	rev := "main"
	files := map[string][]byte{
		"config.json":            []byte(`{"hidden_size":128}`),
		"tokenizer.json":         []byte(`{"model":{"type":"BPE"}}`),
		"model.safetensors":      make([]byte, 4096),
		"ignored.bin":            []byte("pytorch weights — should be skipped"),
		"special_tokens_map.json": []byte(`{}`),
	}
	srv := newMockHFServer(t, repo, rev, files)
	defer srv.Close()

	prevEndpoint := Endpoint
	Endpoint = srv.URL
	defer func() { Endpoint = prevEndpoint }()

	tmp := t.TempDir()
	r := testResolver(t, srv, tmp)

	h, err := r.Resolve(context.Background(), Spec{Repo: repo, Revision: rev})
	if err != nil {
		t.Fatal(err)
	}
	if h.RepoID != repo {
		t.Errorf("RepoID=%q", h.RepoID)
	}
	if h.ConfigPath == "" {
		t.Error("expected config.json to be present")
	}
	if h.TokenizerPath == "" {
		t.Error("expected tokenizer.json to be present")
	}
	if len(h.WeightsFiles) != 1 {
		t.Errorf("WeightsFiles=%v", h.WeightsFiles)
	}
	if h.IsSharded {
		t.Error("single-file model should not report sharded")
	}
	// .bin must have been skipped.
	if _, err := os.Stat(filepath.Join(h.LocalDir, "ignored.bin")); !os.IsNotExist(err) {
		t.Error(".bin files should not be downloaded")
	}
}

func TestResolverShardedModel(t *testing.T) {
	repo := "fake/sharded"
	files := map[string][]byte{
		"config.json":                       []byte(`{}`),
		"tokenizer.json":                    []byte(`{}`),
		"model-00001-of-00003.safetensors":  make([]byte, 1024),
		"model-00002-of-00003.safetensors":  make([]byte, 1024),
		"model-00003-of-00003.safetensors":  make([]byte, 1024),
		"model.safetensors.index.json":      []byte(`{"weight_map":{}}`),
	}
	srv := newMockHFServer(t, repo, "main", files)
	defer srv.Close()
	prev := Endpoint
	Endpoint = srv.URL
	defer func() { Endpoint = prev }()

	tmp := t.TempDir()
	r := testResolver(t, srv, tmp)

	h, err := r.Resolve(context.Background(), Spec{Repo: repo, Revision: "main"})
	if err != nil {
		t.Fatal(err)
	}
	if !h.IsSharded {
		t.Error("expected IsSharded=true")
	}
	if len(h.WeightsFiles) != 3 {
		t.Errorf("expected 3 shards, got %d", len(h.WeightsFiles))
	}
	// Files must be sorted so load order is deterministic.
	for i := 1; i < len(h.WeightsFiles); i++ {
		if h.WeightsFiles[i-1] >= h.WeightsFiles[i] {
			t.Errorf("shards not sorted: %v", h.WeightsFiles)
		}
	}
}

func TestResolverCacheReuse(t *testing.T) {
	repo := "fake/reused"
	files := map[string][]byte{
		"config.json":       []byte(`{}`),
		"tokenizer.json":    []byte(`{}`),
		"model.safetensors": make([]byte, 2048),
	}
	srv := newMockHFServer(t, repo, "main", files)
	defer srv.Close()
	prev := Endpoint
	Endpoint = srv.URL
	defer func() { Endpoint = prev }()

	tmp := t.TempDir()
	r := testResolver(t, srv, tmp)

	// First resolve downloads everything.
	_, err := r.Resolve(context.Background(), Spec{Repo: repo, Revision: "main"})
	if err != nil {
		t.Fatal(err)
	}

	// Note which paths the mock server served weights for.
	resolvePath := "/" + repo + "/resolve/main/model.safetensors"
	firstCount := srv.Config.Handler.(http.HandlerFunc)
	_ = firstCount // silence unused; we'll use the mock's hits map directly below

	// Reach into the underlying mock via a type assertion on the handler.
	// httptest wraps the HandlerFunc so we can't get back the *mockHFServer.
	// Instead, re-check by replacing one file's bytes on disk and ensuring
	// the resolver doesn't re-download (size matches → fast path).
	//
	// Fault-injection: mutate the cached file and verify it's preserved.
	cached := filepath.Join(tmp, "models--"+strings.ReplaceAll(repo, "/", "--"), "model.safetensors")
	want := make([]byte, 2048)
	for i := range want {
		want[i] = 0xAB
	}
	if err := os.WriteFile(cached, want, 0o644); err != nil {
		t.Fatal(err)
	}

	h, err := r.Resolve(context.Background(), Spec{Repo: repo, Revision: "main"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(h.WeightsFiles[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) || got[0] != 0xAB {
		t.Error("resolver re-downloaded file that already had the right size")
	}
	_ = resolvePath
}

func TestResolverAuthRequired(t *testing.T) {
	repo := "gated/model"
	files := map[string][]byte{
		"config.json":       []byte(`{}`),
		"tokenizer.json":    []byte(`{}`),
		"model.safetensors": make([]byte, 64),
	}
	m := &mockHFServer{
		repo: repo, revision: "main", files: files,
		tokenSet: "secret",
		hits:     map[string]int{},
		t:        t,
	}
	srv := httptest.NewServer(http.HandlerFunc(m.handle))
	defer srv.Close()
	prev := Endpoint
	Endpoint = srv.URL
	defer func() { Endpoint = prev }()

	tmp := t.TempDir()

	// Without token → should fail with an auth hint.
	r := &Resolver{
		CacheDir: tmp, HTTPClient: srv.Client(),
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	_, err := r.Resolve(context.Background(), Spec{Repo: repo, Revision: "main"})
	if err == nil {
		t.Fatal("expected error without token")
	}
	if !strings.Contains(err.Error(), "HF_TOKEN") {
		t.Errorf("auth error should mention HF_TOKEN, got: %v", err)
	}

	// With token → succeeds.
	r.Token = "secret"
	if _, err := r.Resolve(context.Background(), Spec{Repo: repo, Revision: "main"}); err != nil {
		t.Fatalf("expected success with token, got %v", err)
	}
}

func TestDefaultCacheDirHonorsEnv(t *testing.T) {
	prev := os.Getenv("GOLLMGO_CACHE_DIR")
	defer os.Setenv("GOLLMGO_CACHE_DIR", prev)
	os.Setenv("GOLLMGO_CACHE_DIR", "/custom/cache")
	if got := DefaultCacheDir(); got != "/custom/cache" {
		t.Errorf("DefaultCacheDir=%q, want /custom/cache", got)
	}
}

func TestTokenFromEnv(t *testing.T) {
	for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"} {
		t.Setenv(k, "")
	}
	if TokenFromEnv() != "" {
		t.Fatal("expected empty token when no env vars set")
	}
	t.Setenv("HF_TOKEN", "hf_xxx")
	if TokenFromEnv() != "hf_xxx" {
		t.Errorf("TokenFromEnv=%q, want hf_xxx", TokenFromEnv())
	}
}

// Pre-populated local directory should resolve without any HTTP.
func TestResolverLocalPath(t *testing.T) {
	tmp := t.TempDir()
	os.WriteFile(filepath.Join(tmp, "config.json"), []byte(`{}`), 0o644)
	os.WriteFile(filepath.Join(tmp, "tokenizer.json"), []byte(`{}`), 0o644)
	os.WriteFile(filepath.Join(tmp, "model.safetensors"), make([]byte, 128), 0o644)

	r := &Resolver{
		HTTPClient: &http.Client{Timeout: time.Second},
		Log:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	h, err := r.Resolve(context.Background(), Spec{IsLocal: true, LocalPath: tmp})
	if err != nil {
		t.Fatal(err)
	}
	if h.LocalDir != tmp {
		t.Errorf("LocalDir=%q, want %q", h.LocalDir, tmp)
	}
	if len(h.WeightsFiles) != 1 {
		t.Errorf("expected 1 weight file, got %d", len(h.WeightsFiles))
	}
	if h.ConfigPath == "" || h.TokenizerPath == "" {
		t.Error("expected config and tokenizer paths")
	}
}
