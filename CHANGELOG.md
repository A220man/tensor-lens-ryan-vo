# Changelog

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
