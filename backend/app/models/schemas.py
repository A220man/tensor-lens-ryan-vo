from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)


class HardwareProfileBase(BaseSchema):
    name: str = Field(..., min_length=2, max_length=128)
    category: str = Field("gpu", description="gpu, multi_gpu, cpu, rocm")
    vram_gb: float = Field(..., gt=0.0)
    memory_bandwidth_gbps: float = Field(..., gt=0.0)
    pcie_gen: int = Field(4, ge=1, le=5)
    pcie_lanes: int = Field(16, ge=1, le=16)
    pcie_bandwidth_gbps: float = Field(..., gt=0.0)
    system_ram_gb: float = Field(..., gt=0.0)
    disk_read_mbps: float = Field(..., gt=0.0)
    compute_tflops_fp16: float = Field(..., gt=0.0)


class HardwareProfileCreate(HardwareProfileBase):
    id: Optional[str] = None


class HardwareProfileResponse(HardwareProfileBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ModelSpecBase(BaseSchema):
    name: str = Field(..., min_length=2, max_length=128)
    family: str = Field("llama", description="llama, qwen, mistral, deepseek")
    parameter_count_billion: float = Field(..., gt=0.0)
    hidden_dim: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    num_kv_heads: int = Field(..., gt=0)
    vocab_size: int = Field(..., gt=0)
    max_seq_length: int = Field(..., gt=0)
    intermediate_dim: int = Field(..., gt=0)
    default_dtype: str = "bfloat16"


class ModelSpecCreate(ModelSpecBase):
    id: Optional[str] = None


class ModelSpecResponse(ModelSpecBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class PlanCalculateRequest(BaseSchema):
    model_id: str
    hardware_id: str
    quantization: str = Field("int4", description="fp16, int8, int4")
    batch_size: int = Field(1, ge=1, le=128)
    seq_len: int = Field(512, ge=16, le=32768)
    manual_layers_per_chunk: Optional[int] = Field(None, ge=1)


class InferencePlanResponse(BaseSchema):
    id: str
    model_id: str
    hardware_id: str
    quantization: str
    batch_size: int
    seq_len: int
    layers_per_chunk: int
    total_chunks: int
    param_vram_mb: float
    activation_vram_mb: float
    kv_cache_vram_mb: float
    total_vram_mb: float
    vram_utilization_percent: float
    est_prefill_sec: float
    est_generation_sec: float
    est_tokens_per_sec: float
    created_at: datetime

    class Config:
        from_attributes = True


class BenchmarkRunRequest(BaseSchema):
    model_id: str
    hardware_id: str
    quantization: str = "int4"
    layers_per_chunk: int = 1
    prompt_tokens: int = Field(32, ge=1, le=4096)
    output_tokens: int = Field(16, ge=1, le=1024)
    run_numerical_simulation: bool = False


class BenchmarkRunResponse(BaseSchema):
    id: str
    model_id: str
    hardware_id: str
    plan_id: Optional[str]
    quantization: str
    layers_per_chunk: int
    prompt_tokens: int
    output_tokens: int
    total_time_ms: float
    tokens_per_sec: float
    peak_vram_mb: float
    disk_read_time_ms: float
    pcie_transfer_time_ms: float
    compute_time_ms: float
    status: str
    notes: str
    created_at: datetime

    class Config:
        from_attributes = True


class LayerStreamStep(BaseSchema):
    layer_idx: int
    chunk_idx: int
    stage: str
    disk_read_ms: float
    pcie_transfer_ms: float
    compute_ms: float
    vram_allocated_mb: float
    description: str


class BenchmarkStreamSimulationResponse(BaseSchema):
    benchmark_id: str
    total_layers: int
    chunks_count: int
    steps: List[LayerStreamStep]
    final_tokens_per_sec: float
    peak_vram_mb: float
    numerical_check_passed: bool


class CompatibilityAuditRequest(BaseSchema):
    model_name: str
    transformers_version: str = "4.45.0"
    torch_version: str = "2.4.0"
    target_device: str = "cuda"  # cuda, rocm, directml, cpu
    head_dim: int = 128
    rotary_dim: int = 128


class CompatibilityAuditResponse(BaseSchema):
    id: str
    target_model: str
    transformers_version: str
    torch_version: str
    has_stateful_attribute: bool
    dynamic_cache_compatible: bool
    rope_dimension_aligned: bool
    device_compatible: bool
    issues: List[str]
    recommendations: List[str]
    generated_patch: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdvisorAnalysisRequest(BaseSchema):
    benchmark_id: Optional[str] = None
    plan_id: Optional[str] = None
    user_query: Optional[str] = None


class AdvisorAnalysisResponse(BaseSchema):
    provider: str
    model: str
    is_grounded: bool
    advisory_label: str
    recommendations: List[str]
    analysis_text: str
    hardware_assessment: str


class AuditLogResponse(BaseSchema):
    id: str
    user_id: str
    role: str
    action: str
    resource: str
    details: str
    timestamp: datetime

    class Config:
        from_attributes = True


class UserInfoResponse(BaseSchema):
    sub: str
    email: str
    roles: List[str]
    is_demo: bool


class DemoTokenRequest(BaseSchema):
    role: str = Field("analyst", description="viewer, analyst, admin")


class TokenResponse(BaseSchema):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
