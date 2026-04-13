# gollmgo - Agent Entry Guide

This repository supports both Claude Code and Codex workflows.

## Start Here
- Product direction: `docs/roadmap.md`
- System shape: `docs/architecture.md`
- Engineering standards: `docs/engineering.md`
- Execution backlog: `epics/`
- Phase 1 fast-track plan: `epics/epic-01-execution-plan.md`
- Remaining execution plans: `epics/epic-02-execution-plan.md`, `epics/epic-03-execution-plan.md`, `epics/epic-04-execution-plan.md`, `epics/epic-05-execution-plan.md`, `epics/epic-06-execution-plan.md`
- Decisions and constraints: `decisions/adr.md`

## Agent-Specific Memory
- Claude Code: `CLAUDE.md`
- Codex: `CODEX.md`

## Shared Working Rules
- Optimize for single-GPU performance on DGX Spark first.
- Preserve backend abstraction for future AMD ROCm support.
- Keep changes benchmark-backed when touching hot paths.
- Update docs in the same patch as behavior/config changes.

## If You Touch Critical Paths
- Scheduler: read `skills/scheduler.md`
- Kernels/backend: read `skills/cuda-kernels.md`
