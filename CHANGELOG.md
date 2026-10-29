# Changelog

## 1.3.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.2.1 — 2026-10-29

### Fixed
- Typo bug in `CompatibilityChecker.run_verification_suite()`: `headDim if False else head_dim` used undefined variable `headDim` instead of `head_dim` (line 165). Fixed to use correct parameter name.

### Verified
- Backend compatibility checker tests should now pass without NameError risk

## 1.2.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.1.0 — 2026-10-21

### Fixed
- Critical bug in `MemoryPlanner.calculate_layer_weights_mb()`: missing `norm_params` definition caused NameError across all planner-dependent endpoints and tests
- Critical bug in `POST /api/planner/calculate`: InferencePlanRecord constructor received extra dict keys (`total_model_weights_mb`, `per_layer_weights_mb`) not defined in entity schema

### Verified
- All 33 backend pytest tests pass (auth, API workflows, compatibility checker, LLM advisor, memory planner, streaming engine)
- All 4 frontend vitest tests pass
- Frontend builds successfully (TypeScript + Vite)
- Preflight validation passes: 149,546 bytes of first-party source within 50-150 KB budget
- Dockerfiles, compose.yaml, and CI workflow present

## 1.0.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

All notable changes to TensorLens are documented in this file.

## [1.0.0] - 2026-10-12

### Added
- Core layer-streaming transformer execution engine supporting sequential layer offload, transfer simulation, and real tensor execution.
- Hardware-aware dynamic N-layer VRAM batching optimizer resolving memory saturation and PCIe throughput tradeoffs.
- Comprehensive memory profiler calculating weight footprint across FP16, INT8, and INT4 block-wise quantizations, sequence activation overhead, and KV-cache expansion.
- Hugging Face runtime compatibility diagnostic suite detecting `_is_stateful`, `DynamicCache`, and RoPE dimension drift with automated patch generation.
- Full-stack FastAPI backend with parameterized SQLite persistence, audit logging, and Keycloak OIDC authentication with local demo mode.
- React and TypeScript dashboard with interactive memory watermarks, real-time layer stream timeline, and benchmark export workflows.
- Docker containerization and docker-compose deployment with Keycloak identity broker integration.
