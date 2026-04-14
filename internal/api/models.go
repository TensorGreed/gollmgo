package api

import "net/http"

// ModelsResponse is the /v1/models response.
type ModelsResponse struct {
	Object string       `json:"object"`
	Data   []ModelEntry `json:"data"`
}

// ModelEntry describes one available model.
type ModelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
	// Created is a unix timestamp; OpenAI clients expect it.
	Created int64 `json:"created"`
}

// ModelsHandler returns the list of loaded models. The server reports the
// actual model id derived from the loaded weights (or "mock" in dev mode).
// gollmgo currently serves a single model per process; the response is a
// list with one entry to match the OpenAI schema.
func (s *Server) ModelsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		resp := ModelsResponse{
			Object: "list",
			Data: []ModelEntry{{
				ID:      s.modelID,
				Object:  "model",
				OwnedBy: "local",
				Created: s.startedUnix,
			}},
		}
		writeJSON(w, http.StatusOK, resp)
	}
}
