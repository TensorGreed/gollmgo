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

		// Build a placeholder prompt string for the engine.
		prompt := ""
		for _, m := range req.Messages {
			prompt += m.Content + " "
		}

		engineReq := &engine.Request{
			ID:        fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
			TokenIDs:  placeholderTokenize(prompt),
			MaxTokens: maxTokens,
		}

		if req.Stream {
			s.handleStream(w, r, &req, engineReq)
		} else {
			s.handleNonStream(w, r, &req, engineReq)
		}
	}
}

func (s *Server) handleNonStream(w http.ResponseWriter, r *http.Request, req *ChatCompletionRequest, engineReq *engine.Request) {
	if err := s.engine.Enqueue(r.Context(), engineReq); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, APIError{Error: APIErrorBody{
			Message: "engine unavailable: " + err.Error(),
			Type:    "server_error",
			Code:    "engine_error",
		}})
		return
	}

	// Collect all tokens until finished.
	var generated []int32
	for {
		tokens, err := s.engine.NextTokens(r.Context())
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, APIError{Error: APIErrorBody{
				Message: "generation error: " + err.Error(),
				Type:    "server_error",
				Code:    "generation_error",
			}})
			return
		}
		finished := false
		for _, tok := range tokens {
			generated = append(generated, tok.TokenID)
			if tok.Finished {
				finished = true
			}
		}
		if finished || len(tokens) == 0 {
			break
		}
	}

	resp := ChatCompletionResponse{
		ID:      engineReq.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []ChatCompletionChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: placeholderDetokenize(generated)},
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

func (s *Server) handleStream(w http.ResponseWriter, r *http.Request, req *ChatCompletionRequest, engineReq *engine.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, APIError{Error: APIErrorBody{
			Message: "streaming not supported",
			Type:    "server_error",
			Code:    "streaming_unsupported",
		}})
		return
	}

	if err := s.engine.Enqueue(r.Context(), engineReq); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, APIError{Error: APIErrorBody{
			Message: "engine unavailable: " + err.Error(),
			Type:    "server_error",
			Code:    "engine_error",
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

	for {
		tokens, err := s.engine.NextTokens(r.Context())
		if err != nil {
			break
		}
		finished := false
		for _, tok := range tokens {
			content := placeholderDetokenize([]int32{tok.TokenID})
			var finishReason *string
			if tok.Finished {
				s := "stop"
				finishReason = &s
				finished = true
			}
			writeSSE(w, flusher, StreamChunk{
				ID:      engineReq.ID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   req.Model,
				Choices: []StreamChunkChoice{{
					Index:        0,
					Delta:        StreamChunkDelta{Content: content},
					FinishReason: finishReason,
				}},
			})
		}
		if finished || len(tokens) == 0 {
			break
		}
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
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

// placeholderTokenize is a stub until real tokenizer is wired in.
func placeholderTokenize(text string) []int32 {
	ids := make([]int32, len(text))
	for i, b := range []byte(text) {
		ids[i] = int32(b)
	}
	return ids
}

// placeholderDetokenize is a stub until real tokenizer is wired in.
func placeholderDetokenize(ids []int32) string {
	bs := make([]byte, len(ids))
	for i, id := range ids {
		bs[i] = byte(id)
	}
	return string(bs)
}
