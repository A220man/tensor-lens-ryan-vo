"""FastAPI REST API routes for TensorLens."""
import csv
import io
import time
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.database import get_db
from app.core.auth import get_current_user, require_role, create_demo_token, UserPrincipal
from app.models.entities import (
    HardwareProfileRecord,
    ModelSpecRecord,
    InferencePlanRecord,
    BenchmarkRunRecord,
    AuditLogRecord,
)
from app.models.schemas import (
    HardwareProfileCreate,
    HardwareProfileResponse,
    ModelSpecCreate,
    ModelSpecResponse,
    PlanCalculateRequest,
    InferencePlanResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkStreamSimulationResponse,
    CompatibilityAuditRequest,
    CompatibilityAuditResponse,
    PatchVerificationRequest,
    PatchVerificationResponse,
    AdvisorAnalysisRequest,
    AdvisorAnalysisResponse,
    AuditLogResponse,
    UserInfoResponse,
    DemoTokenRequest,
    TokenResponse,
)
from app.services.memory_planner import MemoryPlanner
from app.services.streaming_engine import LayerStreamingEngine
from app.services.compatibility_checker import CompatibilityChecker
from app.services.llm_advisor import LlmAdvisor
from app.services.storage import log_audit_action

api_router = APIRouter(prefix="/api")


# --- Health & Discovery ---
@api_router.get("/health")
def health_check() -> dict:
    """System health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.ENVIRONMENT,
        "demo_mode": settings.DEMO_MODE,
        "product": "tensor-lens-ryan-vo",
    }


# --- Authentication & User Info ---
@api_router.post("/auth/demo-token", response_model=TokenResponse)
def get_demo_token(req: DemoTokenRequest) -> TokenResponse:
    """Generate local demo JWT token for viewer, analyst, or admin roles."""
    token = create_demo_token(role=req.role)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=28800,
        role=req.role,
    )


@api_router.get("/auth/userinfo", response_model=UserInfoResponse)
def get_user_info(current_user: UserPrincipal = Depends(get_current_user)) -> UserInfoResponse:
    """Return authenticated user profile and roles."""
    return UserInfoResponse(
        sub=current_user.sub,
        email=current_user.email,
        roles=current_user.roles,
        is_demo=current_user.is_demo,
    )


# --- Hardware Profiles ---
@api_router.get("/hardware", response_model=List[HardwareProfileResponse])
def list_hardware(
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> List[HardwareProfileResponse]:
    """Retrieve all hardware profiles."""
    return db.query(HardwareProfileRecord).order_by(HardwareProfileRecord.vram_gb.asc()).all()


@api_router.get("/hardware/{hardware_id}", response_model=HardwareProfileResponse)
def get_hardware(
    hardware_id: str,
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> HardwareProfileResponse:
    """Retrieve specific hardware profile."""
    item = db.query(HardwareProfileRecord).filter(HardwareProfileRecord.id == hardware_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Hardware profile not found")
    return item


@api_router.post("/hardware", response_model=HardwareProfileResponse, status_code=status.HTTP_201_CREATED)
def create_hardware(
    req: HardwareProfileCreate,
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(require_role("analyst"))
) -> HardwareProfileResponse:
    """Create a new custom hardware profile (analyst/admin only)."""
    hw_id = req.id or f"hw-{int(time.time() * 1000)}"
    if db.query(HardwareProfileRecord).filter(HardwareProfileRecord.id == hw_id).first():
        raise HTTPException(status_code=400, detail="Hardware profile ID already exists")

    new_hw = HardwareProfileRecord(id=hw_id, **req.model_dump(exclude={"id"}))
    db.add(new_hw)
    db.commit()
    db.refresh(new_hw)

    log_audit_action(
        db, user_id=current_user.sub, role=current_user.roles[0] if current_user.roles else "user",
        action="CREATE_HARDWARE", resource=hw_id, details=f"Created hardware profile {req.name} ({req.vram_gb} GB VRAM)"
    )
    return new_hw


# --- Model Specifications ---
@api_router.get("/models", response_model=List[ModelSpecResponse])
def list_models(
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> List[ModelSpecResponse]:
    """Retrieve all model architectures."""
    return db.query(ModelSpecRecord).order_by(ModelSpecRecord.parameter_count_billion.asc()).all()


@api_router.get("/models/{model_id}", response_model=ModelSpecResponse)
def get_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> ModelSpecResponse:
    """Retrieve specific model architecture."""
    item = db.query(ModelSpecRecord).filter(ModelSpecRecord.id == model_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Model specification not found")
    return item


@api_router.post("/models", response_model=ModelSpecResponse, status_code=status.HTTP_201_CREATED)
def create_model(
    req: ModelSpecCreate,
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(require_role("analyst"))
) -> ModelSpecResponse:
    """Register custom model architecture (analyst/admin only)."""
    m_id = req.id or f"model-{int(time.time() * 1000)}"
    if db.query(ModelSpecRecord).filter(ModelSpecRecord.id == m_id).first():
        raise HTTPException(status_code=400, detail="Model ID already exists")

    new_model = ModelSpecRecord(id=m_id, **req.model_dump(exclude={"id"}))
    db.add(new_model)
    db.commit()
    db.refresh(new_model)

    log_audit_action(
        db, user_id=current_user.sub, role=current_user.roles[0] if current_user.roles else "user",
        action="CREATE_MODEL", resource=m_id, details=f"Registered model {req.name} ({req.parameter_count_billion}B params)"
    )
    return new_model


# --- Memory Planner ---
@api_router.post("/planner/calculate", response_model=InferencePlanResponse)
def calculate_plan(
    req: PlanCalculateRequest,
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> InferencePlanResponse:
    """
    Solve optimal layer-by-layer batch chunk size ($N$) to maximize VRAM saturation
    without exceeding hardware limits.
    """
    model = db.query(ModelSpecRecord).filter(ModelSpecRecord.id == req.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    hardware = db.query(HardwareProfileRecord).filter(HardwareProfileRecord.id == req.hardware_id).first()
    if not hardware:
        raise HTTPException(status_code=404, detail="Hardware profile not found")

    plan_metrics = MemoryPlanner.solve_optimal_chunk_size(
        model=model,
        hardware=hardware,
        quantization=req.quantization,
        batch_size=req.batch_size,
        seq_len=req.seq_len,
        manual_chunk_size=req.manual_layers_per_chunk,
    )

    plan_id = f"plan-{int(time.time() * 1000)}"
    plan_record = InferencePlanRecord(
        id=plan_id,
        model_id=model.id,
        hardware_id=hardware.id,
        quantization=req.quantization,
        batch_size=req.batch_size,
        seq_len=req.seq_len,
        layers_per_chunk=plan_metrics["layers_per_chunk"],
        total_chunks=plan_metrics["total_chunks"],
        param_vram_mb=plan_metrics["param_vram_mb"],
        activation_vram_mb=plan_metrics["activation_vram_mb"],
        kv_cache_vram_mb=plan_metrics["kv_cache_vram_mb"],
        total_vram_mb=plan_metrics["total_vram_mb"],
        vram_utilization_percent=plan_metrics["vram_utilization_percent"],
        est_prefill_sec=plan_metrics["est_prefill_sec"],
        est_generation_sec=plan_metrics["est_generation_sec"],
        est_tokens_per_sec=plan_metrics["est_tokens_per_sec"],
    )
    db.add(plan_record)
    db.commit()
    db.refresh(plan_record)

    log_audit_action(
        db, user_id=current_user.sub, role=current_user.roles[0] if current_user.roles else "user",
        action="CALCULATE_PLAN", resource=plan_id,
        details=f"Calculated plan for {model.name} on {hardware.name} ({plan_metrics['layers_per_chunk']} layers/chunk)"
    )
    return plan_record


@api_router.get("/planner/plans", response_model=List[InferencePlanResponse])
def list_plans(
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> List[InferencePlanResponse]:
    """List calculated inference plans."""
    return db.query(InferencePlanRecord).order_by(InferencePlanRecord.created_at.desc()).limit(50).all()


# --- Benchmark & Layer Stream Simulation ---
@api_router.post("/benchmark/simulate", response_model=BenchmarkStreamSimulationResponse)
def simulate_stream(
    req: BenchmarkRunRequest,
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> BenchmarkStreamSimulationResponse:
    """Execute live layer streaming simulation with step-by-step telemetry."""
    model = db.query(ModelSpecRecord).filter(ModelSpecRecord.id == req.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    hardware = db.query(HardwareProfileRecord).filter(HardwareProfileRecord.id == req.hardware_id).first()
    if not hardware:
        raise HTTPException(status_code=404, detail="Hardware profile not found")

    result = LayerStreamingEngine.execute_stream_simulation(
        model=model,
        hardware=hardware,
        quantization=req.quantization,
        layers_per_chunk=req.layers_per_chunk,
        prompt_tokens=req.prompt_tokens,
        output_tokens=req.output_tokens,
        run_numerical_check=req.run_numerical_simulation,
    )
    return result


@api_router.post("/benchmark/run", response_model=BenchmarkRunResponse, status_code=status.HTTP_201_CREATED)
def record_benchmark_run(
    req: BenchmarkRunRequest,
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(require_role("analyst"))
) -> BenchmarkRunResponse:
    """Record verified benchmark execution run (analyst/admin only)."""
    model = db.query(ModelSpecRecord).filter(ModelSpecRecord.id == req.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    hardware = db.query(HardwareProfileRecord).filter(HardwareProfileRecord.id == req.hardware_id).first()
    if not hardware:
        raise HTTPException(status_code=404, detail="Hardware profile not found")

    sim = LayerStreamingEngine.execute_stream_simulation(
        model=model,
        hardware=hardware,
        quantization=req.quantization,
        layers_per_chunk=req.layers_per_chunk,
        prompt_tokens=req.prompt_tokens,
        output_tokens=req.output_tokens,
        run_numerical_check=req.run_numerical_simulation,
    )

    total_time_ms = sum(s.disk_read_ms + s.pcie_transfer_ms + s.compute_ms for s in sim.steps)
    disk_total_ms = sum(s.disk_read_ms for s in sim.steps)
    pcie_total_ms = sum(s.pcie_transfer_ms for s in sim.steps)
    compute_total_ms = sum(s.compute_ms for s in sim.steps)

    b_id = f"bench-{int(time.time() * 1000)}"
    bench_record = BenchmarkRunRecord(
        id=b_id, model_id=model.id, hardware_id=hardware.id, quantization=req.quantization,
        layers_per_chunk=req.layers_per_chunk, prompt_tokens=req.prompt_tokens, output_tokens=req.output_tokens,
        total_time_ms=round(total_time_ms, 2), tokens_per_sec=sim.final_tokens_per_sec, peak_vram_mb=sim.peak_vram_mb,
        disk_read_time_ms=round(disk_total_ms, 2), pcie_transfer_time_ms=round(pcie_total_ms, 2),
        compute_time_ms=round(compute_total_ms, 2), status="completed",
        notes=f"Simulated execution across {sim.chunks_count} chunks with numerical check {sim.numerical_check_passed}",
    )
    db.add(bench_record)
    db.commit()
    db.refresh(bench_record)

    log_audit_action(
        db, user_id=current_user.sub, role=current_user.roles[0] if current_user.roles else "user",
        action="RUN_BENCHMARK", resource=b_id, details=f"Executed benchmark run for {model.name} ({sim.final_tokens_per_sec} tok/s)"
    )
    return bench_record


@api_router.get("/benchmark/history", response_model=List[BenchmarkRunResponse])
def get_benchmark_history(
    model_id: Optional[str] = None,
    hardware_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> List[BenchmarkRunResponse]:
    """Retrieve benchmark run history with optional filtering."""
    query = db.query(BenchmarkRunRecord)
    if model_id:
        query = query.filter(BenchmarkRunRecord.model_id == model_id)
    if hardware_id:
        query = query.filter(BenchmarkRunRecord.hardware_id == hardware_id)
    return query.order_by(BenchmarkRunRecord.created_at.desc()).limit(limit).all()


@api_router.get("/benchmark/export")
def export_benchmarks(
    format: str = Query("json", pattern="^(json|csv)$"),
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> Response:
    """Export benchmark runs in JSON or CSV format."""
    runs = db.query(BenchmarkRunRecord).order_by(BenchmarkRunRecord.created_at.desc()).all()
    if format == "csv":
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["id", "model_id", "hardware_id", "quantization", "layers_per_chunk", "prompt_tokens", "output_tokens", "total_time_ms", "tokens_per_sec", "peak_vram_mb", "disk_read_ms", "pcie_transfer_ms", "compute_ms", "created_at"])
        for r in runs:
            w.writerow([r.id, r.model_id, r.hardware_id, r.quantization, r.layers_per_chunk, r.prompt_tokens, r.output_tokens, r.total_time_ms, r.tokens_per_sec, r.peak_vram_mb, r.disk_read_time_ms, r.pcie_transfer_time_ms, r.compute_time_ms, r.created_at.isoformat()])
        return Response(content=out.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=tensor_lens_benchmarks.csv"})

    import json
    data = [{"id": r.id, "model_id": r.model_id, "hardware_id": r.hardware_id, "quantization": r.quantization, "layers_per_chunk": r.layers_per_chunk, "tokens_per_sec": r.tokens_per_sec, "peak_vram_mb": r.peak_vram_mb, "total_time_ms": r.total_time_ms, "created_at": r.created_at.isoformat()} for r in runs]
    return Response(content=json.dumps(data, indent=2), media_type="application/json", headers={"Content-Disposition": "attachment; filename=tensor_lens_benchmarks.json"})


# --- Compatibility Diagnostics ---
@api_router.post("/compatibility/audit", response_model=CompatibilityAuditResponse)
def audit_compatibility(
    req: CompatibilityAuditRequest,
    current_user: UserPrincipal = Depends(get_current_user)
) -> CompatibilityAuditResponse:
    """
    Audit environment for Hugging Face version drift and generate automated patch shims.
    Directly addresses issues #258, #301, #317.
    """
    resp = CompatibilityChecker.audit_environment(
        model_name=req.model_name,
        transformers_version=req.transformers_version,
        torch_version=req.torch_version,
        target_device=req.target_device,
        head_dim=req.head_dim,
        rotary_dim=req.rotary_dim,
    )
    resp.created_at = datetime.now(timezone.utc)
    return resp


@api_router.post("/compatibility/verify", response_model=PatchVerificationResponse)
def verify_compatibility_patch(
    req: PatchVerificationRequest,
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> PatchVerificationResponse:
    """Run automated verification test suite on patched model definitions and shims."""
    resp = CompatibilityChecker.run_verification_suite(
        model_name=req.model_name,
        transformers_version=req.transformers_version,
        target_device=req.target_device,
        head_dim=req.head_dim,
        rotary_dim=req.rotary_dim,
        patch_code=req.patch_code or ""
    )
    log_audit_action(
        db,
        user_id=current_user.sub,
        role=current_user.roles[0] if current_user.roles else "user",
        action="VERIFY_PATCH",
        resource=resp.id,
        details=f"Ran patch verification for {req.model_name}: {resp.passed_tests}/{resp.total_tests} passed"
    )
    return resp


# --- LLM Advisory ---
@api_router.post("/advisor/analyze", response_model=AdvisorAnalysisResponse)
async def analyze_with_advisor(
    req: AdvisorAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(get_current_user)
) -> AdvisorAnalysisResponse:
    """Generate opt-in advisory analysis strictly grounded in verified run telemetry."""
    plan = None
    if req.plan_id:
        plan = db.query(InferencePlanRecord).filter(InferencePlanRecord.id == req.plan_id).first()

    benchmark = None
    if req.benchmark_id:
        benchmark = db.query(BenchmarkRunRecord).filter(BenchmarkRunRecord.id == req.benchmark_id).first()

    result = await LlmAdvisor.request_advisory(
        plan=plan,
        benchmark=benchmark,
        user_query=req.user_query,
    )

    return AdvisorAnalysisResponse(
        provider=result["provider"],
        model=result["model"],
        is_grounded=result["is_grounded"],
        advisory_label=result["advisory_label"],
        recommendations=result["recommendations"],
        analysis_text=result["analysis_text"],
        hardware_assessment=result["hardware_assessment"],
    )


# --- Audit Logs ---
@api_router.get("/audit-logs", response_model=List[AuditLogResponse])
def get_audit_logs(
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: UserPrincipal = Depends(require_role("admin"))
) -> List[AuditLogResponse]:
    """Retrieve security audit logs (admin only)."""
    return db.query(AuditLogRecord).order_by(AuditLogRecord.timestamp.desc()).limit(limit).all()
