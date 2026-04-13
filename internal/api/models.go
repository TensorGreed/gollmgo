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
}

// ModelsHandler returns the list of loaded models.
// For now returns a hardcoded placeholder; wired to real model registry later.
func (s *Server) ModelsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		resp := ModelsResponse{
			Object: "list",
			Data: []ModelEntry{{
				ID:      "gollmgo-default",
				Object:  "model",
				OwnedBy: "local",
			}},
		}
		writeJSON(w, http.StatusOK, resp)
	}
}
