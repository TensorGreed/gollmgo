package hfhub

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseSpecHFRepo(t *testing.T) {
	s, err := ParseSpec("meta-llama/Llama-3.1-8B-Instruct")
	if err != nil {
		t.Fatal(err)
	}
	if s.IsLocal {
		t.Fatal("expected HF repo, got local")
	}
	if s.Repo != "meta-llama/Llama-3.1-8B-Instruct" {
		t.Errorf("Repo=%q", s.Repo)
	}
	if s.Revision != "main" {
		t.Errorf("Revision=%q, want main", s.Revision)
	}
}

func TestParseSpecHFRepoWithRevision(t *testing.T) {
	s, err := ParseSpec("meta-llama/Llama-3.1-8B-Instruct@v0.1-release")
	if err != nil {
		t.Fatal(err)
	}
	if s.IsLocal {
		t.Fatal("expected HF repo")
	}
	if s.Repo != "meta-llama/Llama-3.1-8B-Instruct" {
		t.Errorf("Repo=%q", s.Repo)
	}
	if s.Revision != "v0.1-release" {
		t.Errorf("Revision=%q", s.Revision)
	}
}

func TestParseSpecLocalAbsolute(t *testing.T) {
	s, err := ParseSpec("/models/my-llama")
	if err != nil {
		t.Fatal(err)
	}
	if !s.IsLocal {
		t.Fatal("expected local")
	}
	if s.LocalPath != "/models/my-llama" {
		t.Errorf("LocalPath=%q", s.LocalPath)
	}
}

func TestParseSpecLocalRelative(t *testing.T) {
	for _, prefix := range []string{"./weights", "../models/foo"} {
		s, err := ParseSpec(prefix)
		if err != nil {
			t.Fatalf("%q: %v", prefix, err)
		}
		if !s.IsLocal {
			t.Errorf("%q should be local", prefix)
		}
	}
}

func TestParseSpecExistingPathIsLocal(t *testing.T) {
	// Create a path that has no "/" separator — but because it exists on
	// disk, ParseSpec must classify it as local, not as a bogus repo id.
	tmp, err := os.MkdirTemp("", "gollmgo-spec-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	f := filepath.Join(tmp, "weights.safetensors")
	if err := os.WriteFile(f, []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Note: f is absolute, so "/" prefix rule already catches it, but do the
	// inverse: a bare name that exists on disk.
	origDir, _ := os.Getwd()
	defer os.Chdir(origDir)
	_ = os.Chdir(tmp)

	s, err := ParseSpec("weights.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if !s.IsLocal {
		t.Errorf("existing filename should be classified local; got repo=%q", s.Repo)
	}
}

func TestParseSpecRejectsBadInput(t *testing.T) {
	cases := []string{
		"",                 // empty
		"no-slash",         // doesn't look like owner/name, doesn't exist
		"too/many/slashes", // not a valid HF repo id
		"owner/",           // empty name
		"meta-llama/foo@",  // empty revision
		"has spaces/in-it", // illegal char
	}
	for _, c := range cases {
		if _, err := ParseSpec(c); err == nil {
			t.Errorf("expected error for %q, got nil", c)
		}
	}
}

func TestExpandHome(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no HOME available")
	}
	got := expandHome("~/foo")
	if got != filepath.Join(home, "foo") {
		t.Errorf("expandHome(~/foo)=%q, want %q", got, filepath.Join(home, "foo"))
	}
	if expandHome("~") != home {
		t.Errorf("expandHome(~)=%q, want %q", expandHome("~"), home)
	}
	if expandHome("/absolute") != "/absolute" {
		t.Error("expandHome should leave absolute paths alone")
	}
}
