package engine

import (
	"math"
	"math/rand/v2"
	"sort"
)

// SamplingParams controls token selection from logits.
type SamplingParams struct {
	Temperature float64 // 0 = greedy, >0 = scaled softmax
	TopK        int     // 0 = disabled, >0 = keep top-k
	TopP        float64 // 0 = disabled, (0,1] = nucleus sampling
	Seed        *uint64 // nil = non-deterministic, set = deterministic
}

// DefaultSamplingParams returns greedy decoding.
func DefaultSamplingParams() SamplingParams {
	return SamplingParams{Temperature: 0}
}

// Sampler selects tokens from logits.
type Sampler struct {
	params SamplingParams
	rng    *rand.Rand
}

// NewSampler creates a sampler. If params.Seed is set, the sampler is deterministic.
func NewSampler(params SamplingParams) *Sampler {
	var rng *rand.Rand
	if params.Seed != nil {
		rng = rand.New(rand.NewPCG(*params.Seed, 0))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return &Sampler{params: params, rng: rng}
}

// Sample selects one token ID from the given logits.
func (s *Sampler) Sample(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}

	// Greedy: temperature == 0 or temperature very small.
	if s.params.Temperature <= 1e-9 {
		return argmax(logits)
	}

	// Apply temperature.
	scaled := make([]float64, len(logits))
	invT := 1.0 / s.params.Temperature
	for i, l := range logits {
		scaled[i] = float64(l) * invT
	}

	// Top-K filtering.
	if s.params.TopK > 0 && s.params.TopK < len(scaled) {
		scaled = topKFilter(scaled, s.params.TopK)
	}

	// Convert to probabilities via softmax.
	probs := softmax(scaled)

	// Top-P (nucleus) filtering.
	if s.params.TopP > 0 && s.params.TopP < 1.0 {
		probs = topPFilter(probs, s.params.TopP)
	}

	// Categorical sample.
	return categoricalSample(probs, s.rng)
}

// argmax returns the index of the largest element.
func argmax(logits []float32) int32 {
	best := int32(0)
	bestVal := logits[0]
	for i := int32(1); i < int32(len(logits)); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			best = i
		}
	}
	return best
}

// topKFilter keeps the top-k logits and sets the rest to -inf.
func topKFilter(logits []float64, k int) []float64 {
	type iv struct {
		idx int
		val float64
	}
	items := make([]iv, len(logits))
	for i, v := range logits {
		items[i] = iv{i, v}
	}
	sort.Slice(items, func(a, b int) bool {
		return items[a].val > items[b].val
	})

	result := make([]float64, len(logits))
	for i := range result {
		result[i] = math.Inf(-1)
	}
	for i := 0; i < k && i < len(items); i++ {
		result[items[i].idx] = items[i].val
	}
	return result
}

// softmax converts logits to probabilities.
func softmax(logits []float64) []float64 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	probs := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		if math.IsInf(v, -1) {
			probs[i] = 0
		} else {
			probs[i] = math.Exp(v - maxVal)
			sum += probs[i]
		}
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}
	return probs
}

// topPFilter zeroes probabilities outside the nucleus and renormalizes.
func topPFilter(probs []float64, p float64) []float64 {
	type iv struct {
		idx  int
		prob float64
	}
	items := make([]iv, len(probs))
	for i, v := range probs {
		items[i] = iv{i, v}
	}
	sort.Slice(items, func(a, b int) bool {
		return items[a].prob > items[b].prob
	})

	result := make([]float64, len(probs))
	cumulative := 0.0
	for _, item := range items {
		if cumulative >= p && cumulative > 0 {
			break
		}
		result[item.idx] = item.prob
		cumulative += item.prob
	}

	// Renormalize.
	sum := 0.0
	for _, v := range result {
		sum += v
	}
	if sum > 0 {
		for i := range result {
			result[i] /= sum
		}
	}
	return result
}

// categoricalSample draws one index from a probability distribution.
func categoricalSample(probs []float64, rng *rand.Rand) int32 {
	r := rng.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return int32(i)
		}
	}
	// Fallback: return last non-zero.
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return int32(i)
		}
	}
	return 0
}
