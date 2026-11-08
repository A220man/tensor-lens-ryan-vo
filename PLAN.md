# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

layer-streaming-engine

## Roadmap

### layer-streaming-engine — Layer-Streaming Inference Engine and VRAM Optimization Planner

Status: done

Acceptance:
- Simulate and execute layer-by-layer transformer streaming with configurable layer chunking (N-layer batching)
- Compute exact layer parameter memory, activation footprints, and KV-cache allocations across FP16, INT8, and INT4 quantizations
- Expose REST API endpoints for model architectures, hardware profiles, plan calculations, and benchmark executions
- Deliver responsive React dashboard with real-time memory watermarking, transfer timeline visualization, and benchmark export workflows

Implementation: `backend/app/main.py`, `backend/app/services/memory_planner.py`, `backend/app/services/streaming_engine.py`, `backend/app/api/routes.py`, `frontend/src/App.tsx`, `frontend/src/components/MemoryPlanner.tsx`

Tests: `backend/tests/test_memory_planner.py`, `backend/tests/test_streaming_engine.py`, `backend/tests/test_api_workflows.py`, `frontend/src/App.test.tsx`

### hf-runtime-diagnostics — Hugging Face Compatibility Diagnostics and DynamicCache Patch Generator

Status: done

Acceptance:
- Analyze installed transformers environment for known breaking changes like _is_stateful and DynamicCache indexing
- Generate automated patch shims for RoPE rotary embeddings and layer offloading hooks
- Provide verification test suite for patched model definitions

Implementation: `backend/app/services/compatibility_checker.py`, `backend/app/api/routes.py (endpoints: /api/compatibility/audit, /api/compatibility/verify)`, `frontend/src/components/CompatibilityDiagnostics.tsx`, `backend/app/models/entities.py (CompatibilityAuditRecord)`, `backend/app/models/schemas.py (CompatibilityAuditRequest/Response, PatchVerificationRequest/Response)`

Tests: `backend/tests/test_compatibility_checker.py`

### llm-advisor — LLM-Powered Optimization Advisory Grounded in Benchmark Telemetry

Status: done

Acceptance:
- Generate opt-in advisory analysis strictly grounded in verified benchmark and plan records
- Provider-agnostic adapter supporting OpenAI-compatible endpoints and deterministic offline heuristics
- Integrate advisory access from Planner and Benchmarks tabs via modal dialog

Implementation: `backend/app/services/llm_advisor.py`, `backend/app/api/routes.py (endpoint: /api/advisor/analyze)`, `frontend/src/components/LlmAdvisorModal.tsx`, `backend/app/models/schemas.py (AdvisorAnalysisRequest/Response)`

Tests: `backend/tests/test_llm_advisor.py`

### multi-gpu-heterogeneous-scheduler — Heterogeneous Multi-GPU Pipeline Scheduler

Status: planned

Acceptance:
- Distribute partitioned layers across asymmetric GPUs (e.g. K80 + T4 combinations)
- Schedule pipelined transfers across PCI-e and network interconnects
- Provide live multi-device memory telemetry

