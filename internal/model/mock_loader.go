package model

import "context"

// MockLoader returns fixed metadata for testing.
type MockLoader struct {
	Meta *ModelMeta
	Err  error
}

func (m *MockLoader) Load(_ context.Context, _ string) (*ModelMeta, error) {
	if m.Err != nil {
		return nil, m.Err
	}
	return m.Meta, nil
}

// MockTokenizer is a trivial test tokenizer.
type MockTokenizer struct {
	Vocab int
	EOS   int32
}

func (t *MockTokenizer) Encode(text string) ([]int32, error) {
	// One token per byte, for testing only.
	ids := make([]int32, len(text))
	for i, b := range []byte(text) {
		ids[i] = int32(b)
	}
	return ids, nil
}

func (t *MockTokenizer) Decode(ids []int32) (string, error) {
	bs := make([]byte, len(ids))
	for i, id := range ids {
		bs[i] = byte(id)
	}
	return string(bs), nil
}

func (t *MockTokenizer) VocabSize() int  { return t.Vocab }
func (t *MockTokenizer) EOSTokenID() int32 { return t.EOS }
