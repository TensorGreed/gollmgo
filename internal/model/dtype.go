package model

import (
	"encoding/binary"
	"math"
)

// ConvertBF16ToFP16 converts a byte slice of BF16 values to FP16 in-place.
// BF16 layout: 1 sign + 8 exponent + 7 mantissa (same exponent as FP32)
// FP16 layout: 1 sign + 5 exponent + 10 mantissa
// Conversion: BF16 -> FP32 (shift left 16) -> FP16 (standard narrowing).
func ConvertBF16ToFP16(data []byte) []byte {
	if len(data)%2 != 0 {
		return data
	}

	out := make([]byte, len(data))
	n := len(data) / 2

	for i := 0; i < n; i++ {
		// Read BF16 as uint16 (little-endian).
		bf16 := binary.LittleEndian.Uint16(data[i*2:])

		// BF16 -> FP32: the BF16 value is the upper 16 bits of an FP32.
		f32bits := uint32(bf16) << 16
		f32 := math.Float32frombits(f32bits)

		// FP32 -> FP16.
		fp16 := float32ToFP16(f32)

		binary.LittleEndian.PutUint16(out[i*2:], fp16)
	}

	return out
}

// float32ToFP16 converts a float32 to IEEE 754 half-precision (FP16).
func float32ToFP16(f float32) uint16 {
	bits := math.Float32bits(f)

	sign := (bits >> 31) & 1
	exp := int((bits>>23)&0xFF) - 127 // unbiased exponent
	mantissa := bits & 0x7FFFFF       // 23-bit mantissa

	if exp > 15 {
		// Overflow -> infinity.
		return uint16(sign<<15 | 0x7C00)
	}
	if exp < -14 {
		// Underflow / subnormal.
		if exp < -24 {
			return uint16(sign << 15) // zero
		}
		// Subnormal FP16.
		mantissa |= 0x800000 // implicit leading 1
		shift := uint(-exp - 14 + 13)
		mantissa >>= shift
		return uint16(sign<<15 | (mantissa & 0x3FF))
	}

	// Normal case: rebias exponent to FP16 (bias 15).
	fp16Exp := uint16(exp + 15)
	fp16Mantissa := uint16(mantissa >> 13) // keep top 10 bits

	return uint16(sign<<15) | (fp16Exp << 10) | fp16Mantissa
}

// DtypeElementSize returns the byte size per element for a dtype string.
func DtypeElementSize(dtype string) int {
	switch dtype {
	case "F16", "BF16":
		return 2
	case "F32":
		return 4
	case "F64":
		return 8
	case "I8", "U8":
		return 1
	case "I16":
		return 2
	case "I32":
		return 4
	default:
		return 0
	}
}
