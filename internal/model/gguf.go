package model

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// GGUFLoader loads model metadata from GGUF format files.
// Full weight loading deferred to M5; this extracts the header metadata.

// GGUF magic number: "GGUF" in little-endian.
const ggufMagic = 0x46475547 // "GGUF" LE

// GGUF metadata value types.
const (
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12
)

// GGUFLoader implements Loader for GGUF files.
type GGUFLoader struct{}

func (l *GGUFLoader) Load(_ context.Context, path string) (*ModelMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open %s: %w", path, err)
	}
	defer f.Close()

	// Read and validate magic.
	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("gguf: invalid magic 0x%08X (expected 0x%08X)", magic, ggufMagic)
	}

	// Read version.
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}

	// Read tensor count and metadata KV count.
	var tensorCount, kvCount uint64
	if version == 3 {
		if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
			return nil, fmt.Errorf("gguf: read tensor_count: %w", err)
		}
		if err := binary.Read(f, binary.LittleEndian, &kvCount); err != nil {
			return nil, fmt.Errorf("gguf: read kv_count: %w", err)
		}
	} else {
		// v2 uses uint32 for counts.
		var tc32, kv32 uint32
		binary.Read(f, binary.LittleEndian, &tc32)
		binary.Read(f, binary.LittleEndian, &kv32)
		tensorCount = uint64(tc32)
		kvCount = uint64(kv32)
	}

	// Parse metadata key-value pairs.
	meta := &ModelMeta{}
	kv := make(map[string]any)

	for i := uint64(0); i < kvCount; i++ {
		key, err := readGGUFString(f)
		if err != nil {
			return nil, fmt.Errorf("gguf: read kv key %d: %w", i, err)
		}
		val, err := readGGUFValue(f)
		if err != nil {
			return nil, fmt.Errorf("gguf: read kv value for %q: %w", key, err)
		}
		kv[key] = val
	}

	// Extract known keys.
	if v, ok := kv["general.architecture"].(string); ok {
		meta.Family = v
	}
	if v, ok := kv["general.name"].(string); ok {
		meta.Name = v
	}
	if v, ok := getUint32(kv, meta.Family+".block_count"); ok {
		meta.NumLayers = int(v)
	}
	if v, ok := getUint32(kv, meta.Family+".embedding_length"); ok {
		meta.HiddenSize = int(v)
	}
	if v, ok := getUint32(kv, meta.Family+".attention.head_count"); ok {
		meta.NumHeads = int(v)
	}
	if v, ok := getUint32(kv, meta.Family+".attention.head_count_kv"); ok {
		meta.NumKVHeads = int(v)
	}
	if v, ok := getUint32(kv, meta.Family+".context_length"); ok {
		meta.MaxSeqLen = int(v)
	}

	_ = tensorCount // will be used for weight loading in M5

	return meta, nil
}

func readGGUFString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<20 { // 1MB sanity limit
		return "", fmt.Errorf("gguf: string too long (%d)", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readGGUFValue(r io.Reader) (any, error) {
	var vtype uint32
	if err := binary.Read(r, binary.LittleEndian, &vtype); err != nil {
		return nil, err
	}
	return readGGUFTypedValue(r, vtype)
}

func readGGUFTypedValue(r io.Reader, vtype uint32) (any, error) {
	switch vtype {
	case ggufTypeUint8:
		var v uint8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt8:
		var v int8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeUint16:
		var v uint16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt16:
		var v int16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeUint32:
		var v uint32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt32:
		var v int32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeFloat32:
		var v float32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case ggufTypeString:
		return readGGUFString(r)
	case ggufTypeUint64:
		var v uint64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt64:
		var v int64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeFloat64:
		var v float64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeArray:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var length uint64
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return nil, err
		}
		// Skip array contents for now (metadata extraction doesn't need them).
		// We still need to consume the bytes to keep the reader aligned.
		arr := make([]any, 0, min(length, 64)) // cap small for metadata arrays
		for j := uint64(0); j < length; j++ {
			v, err := readGGUFTypedValue(r, elemType)
			if err != nil {
				return nil, err
			}
			if j < 64 {
				arr = append(arr, v)
			}
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("gguf: unknown value type %d", vtype)
	}
}

func getUint32(kv map[string]any, key string) (uint32, bool) {
	v, ok := kv[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case uint32:
		return val, true
	case int32:
		return uint32(val), true
	case uint64:
		return uint32(val), true
	default:
		return 0, false
	}
}
