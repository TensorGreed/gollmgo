package api

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/TensorGreed/gollmgo/internal/config"
	"github.com/TensorGreed/gollmgo/internal/engine"
	"github.com/TensorGreed/gollmgo/internal/model"
)

// Server is the HTTP server for the OpenAI-compatible API.
type Server struct {
	cfg         config.Config
	engine      engine.Engine
	tokenizer   model.Tokenizer
	mux         *http.ServeMux
	srv         *http.Server
	log         *slog.Logger
	modelID     string
	startedUnix int64
}

// NewServer creates a new API server wired to the given engine. The model
// id is the identifier reported by /v1/models and accepted in chat
// completion requests; pass the basename of the loaded model directory or
// the literal "mock" in dev mode.
func NewServer(cfg config.Config, eng engine.Engine, tok model.Tokenizer, log *slog.Logger, modelID string) *Server {
	if modelID == "" {
		modelID = "gollmgo-default"
	}
	s := &Server{
		cfg:         cfg,
		engine:      eng,
		tokenizer:   tok,
		mux:         http.NewServeMux(),
		log:         log,
		modelID:     modelID,
		startedUnix: time.Now().Unix(),
	}
	s.registerRoutes()
	s.srv = &http.Server{
		Addr:              net.JoinHostPort(cfg.Host, fmt.Sprintf("%d", cfg.Port)),
		Handler:           s.mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	return s
}

// ModelID returns the id this server reports via /v1/models.
func (s *Server) ModelID() string { return s.modelID }

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("GET /health/live", LiveHandler())
	s.mux.HandleFunc("GET /health/ready", ReadyHandler())
	s.mux.HandleFunc("GET /v1/models", s.ModelsHandler())
	s.mux.HandleFunc("POST /v1/chat/completions", s.ChatCompletionsHandler())
	s.mux.HandleFunc("GET /metrics", s.MetricsHandler())
}

// ListenAndServe starts the HTTP server.
func (s *Server) ListenAndServe() error {
	s.log.Info("server starting", "addr", s.srv.Addr)
	return s.srv.ListenAndServe()
}

// Shutdown gracefully stops the server.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.srv.Shutdown(ctx)
}

// Handler returns the underlying http.Handler for testing.
func (s *Server) Handler() http.Handler {
	return s.mux
}
