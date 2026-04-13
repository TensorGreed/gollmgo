# Skill: Scheduler Development

## Use This Skill When
- Editing `internal/scheduler/`
- Modifying admission, batching, or preemption logic
- Investigating TTFT/ITL/throughput regressions

---

## Scheduler Mission
Keep GPU useful every tick while preserving fairness and predictable latency.

Primary responsibilities:
1. Admit requests safely under KV/memory limits.
2. Build mixed prefill/decode batches.
3. Progress every active sequence without starvation.
4. Release resources immediately on finish/cancel.

---

## State Model
`WAITING -> PREFILLING -> DECODING -> FINISHED`

Preemption path:
`DECODING -> PREEMPTED -> WAITING`

All transitions must go through explicit transition helpers. No ad-hoc state mutation.

---

## Tick Template
```go
func (s *Scheduler) tick(ctx context.Context) {
    decoding := s.collectDecoding()
    admitted := s.admit(decoding)
    prefill := s.allocatePrefillBudget(admitted)

    batch := s.buildBatch(prefill, decoding)
    out, err := s.runner.Step(ctx, batch)
    if err != nil { s.handleRunnerError(err); return }

    s.applyOutput(batch, out)
    s.cleanupFinishedOrCanceled()
}
```

---

## Invariants
- Sequence block allocation matches token growth.
- No batch exceeds configured limits.
- Scheduler tick contains no blocking network/disk calls.
- Canceled sequences free resources quickly.

---

## Debug Checklist

### Throughput too low
- Check average batch fill.
- Check queue depth versus admission limits.
- Check KV utilization and preemption frequency.

### TTFT too high
- Reduce long-prefill starvation with chunked prefill tuning.
- Confirm graph replay path is active.

### Tail latency unstable
- Review policy/preemption interactions.
- Verify cancellation and cleanup are not delayed.

---

## Before Merge
- [ ] Scheduler unit tests updated
- [ ] Overload/preemption tests pass
- [ ] Benchmarks attached for impacted workloads
- [ ] Config docs updated for new policy knobs
