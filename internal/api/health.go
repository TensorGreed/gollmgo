// Package api provides the OpenAI-compatible HTTP server and operator endpoints.
// This file contains health and readiness endpoint skeletons.
package api

import (
	"encoding/json"
	"net/http"
)

// HealthResponse is the payload for health endpoints.
type HealthResponse struct {
	Status string `json:"status"`
}

// LiveHandler returns 200 if the process is alive.
func LiveHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
	}
}

// ReadyHandler returns 200 when the server is ready to accept requests.
// For now it always returns ok; gated readiness will come with backend bring-up.
func ReadyHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
	}
}
