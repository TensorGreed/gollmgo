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

	_ = tensorCount // used by LoadGGUFWeights

	return meta, nil
}

// GGUF tensor dtype codes.
const (
	ggufDTypeF32  = 0
	ggufDTypeF16  = 1
	ggufDTypeQ4_0 = 2
	ggufDTypeQ4_1 = 3
	ggufDTypeQ5_0 = 6
	ggufDTypeQ5_1 = 7
	ggufDTypeQ8_0 = 8
	ggufDTypeQ8_1 = 9
	ggufDTypeQ2_K = 10
	ggufDTypeQ3_K = 11
	ggufDTypeQ4_K = 12
	ggufDTypeQ5_K = 13
	ggufDTypeQ6_K = 14
	ggufDTypeIQ2  = 15
	ggufDTypeIQ3  = 16
	ggufDTypeBF16 = 30
)

// ggufDtypeName maps GGUF type codes to dtype strings matching safetensors conventions.
func ggufDtypeName(code uint32) string {
	switch code {
	case ggufDTypeF32:
		return "F32"
	case ggufDTypeF16:
		return "F16"
	case ggufDTypeBF16:
		return "BF16"
	default:
		return fmt.Sprintf("GGUF_Q%d", code)
	}
}

// ggufDtypeSize returns bytes per element for unquantized GGUF types.
// Returns 0 for quantized types (not yet supported for direct upload).
func ggufDtypeSize(code uint32) int {
	switch code {
	case ggufDTypeF32:
		return 4
	case ggufDTypeF16, ggufDTypeBF16:
		return 2
	default:
		return 0
	}
}

// ggufTensorDesc describes one tensor from the GGUF header.
type ggufTensorDesc struct {
	Name   string
	NDims  uint32
	Shape  []uint64
	Dtype  uint32
	Offset uint64
}

// LoadGGUFWeights reads all weight tensors from a GGUF file.
// Returns tensors with raw byte data and normalized metadata.
// Only F32, F16, and BF16 tensors are supported; quantized types return an error.
func LoadGGUFWeights(path string) ([]WeightTensor, *ModelMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: open %s: %w", path, err)
	}
	defer f.Close()

	// Parse header.
	var magic, version uint32
	binary.Read(f, binary.LittleEndian, &magic)
	if magic != ggufMagic {
		return nil, nil, fmt.Errorf("gguf: invalid magic 0x%08X", magic)
	}
	binary.Read(f, binary.LittleEndian, &version)
	if version < 2 || version > 3 {
		return nil, nil, fmt.Errorf("gguf: unsupported version %d", version)
	}

	var tensorCount, kvCount uint64
	if version == 3 {
		binary.Read(f, binary.LittleEndian, &tensorCount)
		binary.Read(f, binary.LittleEndian, &kvCount)
	} else {
		var tc32, kv32 uint32
		binary.Read(f, binary.LittleEndian, &tc32)
		binary.Read(f, binary.LittleEndian, &kv32)
		tensorCount = uint64(tc32)
		kvCount = uint64(kv32)
	}

	// Parse metadata KVs.
	kv := make(map[string]any)
	for i := uint64(0); i < kvCount; i++ {
		key, err := readGGUFString(f)
		if err != nil {
			return nil, nil, fmt.Errorf("gguf: read kv key %d: %w", i, err)
		}
		val, err := readGGUFValue(f)
		if err != nil {
			return nil, nil, fmt.Errorf("gguf: read kv value for %q: %w", key, err)
		}
		kv[key] = val
	}

	// Build metadata.
	meta := &ModelMeta{}
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
	if v, ok := getUint32(kv, meta.Family+".vocab_size"); ok {
		meta.VocabSize = int(v)
	}

	// Parse tensor descriptors.
	descs := make([]ggufTensorDesc, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readGGUFString(f)
		if err != nil {
			return nil, nil, fmt.Errorf("gguf: read tensor name %d: %w", i, err)
		}
		var ndims uint32
		if err := binary.Read(f, binary.LittleEndian, &ndims); err != nil {
			return nil, nil, fmt.Errorf("gguf: read tensor ndims: %w", err)
		}
		shape := make([]uint64, ndims)
		for d := uint32(0); d < ndims; d++ {
			if err := binary.Read(f, binary.LittleEndian, &shape[d]); err != nil {
				return nil, nil, fmt.Errorf("gguf: read tensor shape: %w", err)
			}
		}
		var dtype uint32
		if err := binary.Read(f, binary.LittleEndian, &dtype); err != nil {
			return nil, nil, fmt.Errorf("gguf: read tensor dtype: %w", err)
		}
		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			return nil, nil, fmt.Errorf("gguf: read tensor offset: %w", err)
		}
		descs[i] = ggufTensorDesc{
			Name:   name,
			NDims:  ndims,
			Shape:  shape,
			Dtype:  dtype,
			Offset: offset,
		}
	}

	// Data section starts at alignment boundary after header.
	// GGUF v3 aligns data to 32 bytes.
	headerEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: seek header end: %w", err)
	}
	alignment := int64(32)
	dataStart := ((headerEnd + alignment - 1) / alignment) * alignment

	// Read tensor data.
	tensors := make([]WeightTensor, 0, tensorCount)
	for _, desc := range descs {
		elemSize := ggufDtypeSize(desc.Dtype)
		if elemSize == 0 {
			return nil, nil, fmt.Errorf("gguf: tensor %q has quantized dtype %s which is not yet supported for direct upload",
				desc.Name, ggufDtypeName(desc.Dtype))
		}

		// Compute total elements.
		numElements := uint64(1)
		for _, d := range desc.Shape {
			numElements *= d
		}
		sizeBytes := int64(numElements) * int64(elemSize)

		data := make([]byte, sizeBytes)
		if _, err := f.ReadAt(data, dataStart+int64(desc.Offset)); err != nil {
			return nil, nil, fmt.Errorf("gguf: read tensor %q data: %w", desc.Name, err)
		}

		intShape := make([]int, len(desc.Shape))
		for i, d := range desc.Shape {
			intShape[i] = int(d)
		}

		// Set dtype on meta from first tensor if not already set.
		dtypeName := ggufDtypeName(desc.Dtype)
		if meta.Dtype == "" {
			meta.Dtype = dtypeName
		}

		tensors = append(tensors, WeightTensor{
			Name:  desc.Name,
			Dtype: dtypeName,
			Shape: intShape,
			Data:  data,
		})
	}

	return tensors, meta, nil
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
