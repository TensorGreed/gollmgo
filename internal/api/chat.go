package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/TensorGreed/gollmgo/internal/engine"
	"github.com/TensorGreed/gollmgo/internal/metrics"
)

// --- OpenAI-compatible request/response types ---

// ChatMessage is a single message in the conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest is the /v1/chat/completions request body.
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

// ChatCompletionChoice is one completion choice.
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatCompletionResponse is the non-streaming response.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   UsageInfo              `json:"usage"`
}

// UsageInfo reports token counts.
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamChunkDelta is the delta in a streaming chunk.
type StreamChunkDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// StreamChunkChoice is one choice in a streaming chunk.
type StreamChunkChoice struct {
	Index        int              `json:"index"`
	Delta        StreamChunkDelta `json:"delta"`
	FinishReason *string          `json:"finish_reason"`
}

// StreamChunk is one SSE chunk in a streaming response.
type StreamChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []StreamChunkChoice `json:"choices"`
}

// --- Validation ---

// APIError is a structured error response.
type APIError struct {
	Error APIErrorBody `json:"error"`
}

// APIErrorBody holds the error detail.
type APIErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func (r *ChatCompletionRequest) validate() *APIError {
	if len(r.Messages) == 0 {
		return &APIError{Error: APIErrorBody{
			Message: "messages must not be empty",
			Type:    "invalid_request_error",
			Code:    "invalid_messages",
		}}
	}
	for i, m := range r.Messages {
		if m.Role == "" {
			return &APIError{Error: APIErrorBody{
				Message: fmt.Sprintf("messages[%d].role must not be empty", i),
				Type:    "invalid_request_error",
				Code:    "invalid_messages",
			}}
		}
	}
	if r.MaxTokens < 0 {
		return &APIError{Error: APIErrorBody{
			Message: "max_tokens must be non-negative",
			Type:    "invalid_request_error",
			Code:    "invalid_max_tokens",
		}}
	}
	return nil
}

// --- Handler ---

// ChatCompletionsHandler handles /v1/chat/completions.
func (s *Server) ChatCompletionsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		metrics.Global.RequestsTotal.Add(1)

		var req ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, APIError{Error: APIErrorBody{
				Message: "invalid JSON: " + err.Error(),
				Type:    "invalid_request_error",
				Code:    "invalid_json",
			}})
			return
		}

		if apiErr := req.validate(); apiErr != nil {
			writeJSON(w, http.StatusBadRequest, apiErr)
			return
		}

		maxTokens := req.MaxTokens
		if maxTokens == 0 {
			maxTokens = 256
		}

		// Build prompt string and tokenize using the real tokenizer.
		prompt := ""
		for _, m := range req.Messages {
			prompt += m.Content + " "
		}

		tokenIDs, err := s.tokenizer.Encode(prompt)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, APIError{Error: APIErrorBody{
				Message: "tokenization failed: " + err.Error(),
				Type:    "invalid_request_error",
				Code:    "tokenization_error",
			}})
			return
		}

		engineReq := &engine.Request{
			ID:        fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
			TokenIDs:  tokenIDs,
			MaxTokens: maxTokens,
		}

		// Enqueue and get a per-request handle. Tokens arrive on
		// handle.Tokens — isolated from every other request.
		handle, err := s.engine.Enqueue(r.Context(), engineReq)
		if err != nil {
			writeJSON(w, http.StatusServiceUnavailable, APIError{Error: APIErrorBody{
				Message: "engine unavailable: " + err.Error(),
				Type:    "server_error",
				Code:    "engine_error",
			}})
			return
		}

		if req.Stream {
			s.handleStream(w, r, &req, engineReq, handle)
		} else {
			s.handleNonStream(w, r, &req, engineReq, handle)
		}
	}
}

func (s *Server) handleNonStream(w http.ResponseWriter, r *http.Request, req *ChatCompletionRequest, engineReq *engine.Request, handle *engine.RequestHandle) {
	var generated []int32

	// Block on the per-request channel until finished, error, or context done.
	for {
		select {
		case <-r.Context().Done():
			writeJSON(w, http.StatusGatewayTimeout, APIError{Error: APIErrorBody{
				Message: "request cancelled",
				Type:    "server_error",
				Code:    "cancelled",
			}})
			return

		case tok, ok := <-handle.Tokens:
			if !ok {
				// Channel closed without a Finished token — engine shutdown.
				writeJSON(w, http.StatusServiceUnavailable, APIError{Error: APIErrorBody{
					Message: "engine shut down before request completed",
					Type:    "server_error",
					Code:    "engine_shutdown",
				}})
				return
			}
			if tok.Err != nil {
				writeJSON(w, http.StatusInternalServerError, APIError{Error: APIErrorBody{
					Message: "generation error: " + tok.Err.Error(),
					Type:    "server_error",
					Code:    "generation_error",
				}})
				return
			}
			generated = append(generated, tok.TokenID)
			if tok.Finished {
				goto done
			}
		}
	}

done:
	resp := ChatCompletionResponse{
		ID:      engineReq.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []ChatCompletionChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: s.detokenize(generated)},
			FinishReason: "stop",
		}},
		Usage: UsageInfo{
			PromptTokens:     len(engineReq.TokenIDs),
			CompletionTokens: len(generated),
			TotalTokens:      len(engineReq.TokenIDs) + len(generated),
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleStream(w http.ResponseWriter, r *http.Request, req *ChatCompletionRequest, engineReq *engine.Request, handle *engine.RequestHandle) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, APIError{Error: APIErrorBody{
			Message: "streaming not supported",
			Type:    "server_error",
			Code:    "streaming_unsupported",
		}})
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	// Send initial role chunk.
	writeSSE(w, flusher, StreamChunk{
		ID:      engineReq.ID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []StreamChunkChoice{{
			Index: 0,
			Delta: StreamChunkDelta{Role: "assistant"},
		}},
	})

	// sendFinish terminates the stream with a finish_reason and [DONE].
	sendFinish := func(reason string) {
		writeSSE(w, flusher, StreamChunk{
			ID: engineReq.ID, Object: "chat.completion.chunk",
			Created: time.Now().Unix(), Model: req.Model,
			Choices: []StreamChunkChoice{{
				Index:        0,
				Delta:        StreamChunkDelta{},
				FinishReason: &reason,
			}},
		})
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}

	// Block on per-request channel for each token.
	for {
		select {
		case <-r.Context().Done():
			sendFinish("length")
			return
		case tok, ok := <-handle.Tokens:
			if !ok {
				// Channel closed without Finished — engine shut down.
				sendFinish("error")
				return
			}
			if tok.Err != nil {
				// Backend error — terminate cleanly, not as content.
				sendFinish("error")
				return
			}
			content := s.detokenize([]int32{tok.TokenID})
			if tok.Finished {
				// Final content token + finish_reason in same chunk.
				stop := "stop"
				writeSSE(w, flusher, StreamChunk{
					ID:      engineReq.ID,
					Object:  "chat.completion.chunk",
					Created: time.Now().Unix(),
					Model:   req.Model,
					Choices: []StreamChunkChoice{{
						Index:        0,
						Delta:        StreamChunkDelta{Content: content},
						FinishReason: &stop,
					}},
				})
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return
			}
			writeSSE(w, flusher, StreamChunk{
				ID:      engineReq.ID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   req.Model,
				Choices: []StreamChunkChoice{{
					Index: 0,
					Delta: StreamChunkDelta{Content: content},
				}},
			})
		}
	}
}

// --- Helpers ---

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func writeSSE(w http.ResponseWriter, flusher http.Flusher, chunk StreamChunk) {
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func (s *Server) detokenize(ids []int32) string {
	text, err := s.tokenizer.Decode(ids)
	if err != nil {
		return ""
	}
	return text
}
