"""SQLAlchemy persistent entities for TensorLens."""
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base


def utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


class HardwareProfileRecord(Base):
    """Hardware configuration profiling GPU, system memory, PCIe, and disk I/O."""
    
    __tablename__ = "hardware_profiles"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    category = Column(String(64), nullable=False, default="gpu")  # gpu, multi_gpu, cpu, rocm
    vram_gb = Column(Float, nullable=False)
    memory_bandwidth_gbps = Column(Float, nullable=False)
    pcie_gen = Column(Integer, nullable=False, default=4)
    pcie_lanes = Column(Integer, nullable=False, default=16)
    pcie_bandwidth_gbps = Column(Float, nullable=False)
    system_ram_gb = Column(Float, nullable=False)
    disk_read_mbps = Column(Float, nullable=False)
    compute_tflops_fp16 = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    plans = relationship("InferencePlanRecord", back_populates="hardware", cascade="all, delete-orphan")
    benchmarks = relationship("BenchmarkRunRecord", back_populates="hardware", cascade="all, delete-orphan")


class ModelSpecRecord(Base):
    """Large Language Model architecture specification."""
    
    __tablename__ = "model_specs"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    family = Column(String(64), nullable=False)  # llama, qwen, mistral, deepseek
    parameter_count_billion = Column(Float, nullable=False)
    hidden_dim = Column(Integer, nullable=False)
    num_layers = Column(Integer, nullable=False)
    num_heads = Column(Integer, nullable=False)
    num_kv_heads = Column(Integer, nullable=False)
    vocab_size = Column(Integer, nullable=False)
    max_seq_length = Column(Integer, nullable=False)
    intermediate_dim = Column(Integer, nullable=False)
    default_dtype = Column(String(32), default="bfloat16", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    plans = relationship("InferencePlanRecord", back_populates="model", cascade="all, delete-orphan")
    benchmarks = relationship("BenchmarkRunRecord", back_populates="model", cascade="all, delete-orphan")


class InferencePlanRecord(Base):
    """Calculated layer-streaming execution plan and memory budget."""
    
    __tablename__ = "inference_plans"

    id = Column(String(64), primary_key=True, index=True)
    model_id = Column(String(64), ForeignKey("model_specs.id"), nullable=False)
    hardware_id = Column(String(64), ForeignKey("hardware_profiles.id"), nullable=False)
    quantization = Column(String(32), nullable=False)  # fp16, int8, int4
    batch_size = Column(Integer, default=1, nullable=False)
    seq_len = Column(Integer, default=512, nullable=False)
    layers_per_chunk = Column(Integer, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    param_vram_mb = Column(Float, nullable=False)
    activation_vram_mb = Column(Float, nullable=False)
    kv_cache_vram_mb = Column(Float, nullable=False)
    total_vram_mb = Column(Float, nullable=False)
    vram_utilization_percent = Column(Float, nullable=False)
    est_prefill_sec = Column(Float, nullable=False)
    est_generation_sec = Column(Float, nullable=False)
    est_tokens_per_sec = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    model = relationship("ModelSpecRecord", back_populates="plans")
    hardware = relationship("HardwareProfileRecord", back_populates="plans")
    benchmarks = relationship("BenchmarkRunRecord", back_populates="plan", cascade="all, delete-orphan")


class BenchmarkRunRecord(Base):
    """Recorded layer streaming benchmark execution."""
    
    __tablename__ = "benchmark_runs"

    id = Column(String(64), primary_key=True, index=True)
    model_id = Column(String(64), ForeignKey("model_specs.id"), nullable=False)
    hardware_id = Column(String(64), ForeignKey("hardware_profiles.id"), nullable=False)
    plan_id = Column(String(64), ForeignKey("inference_plans.id"), nullable=True)
    quantization = Column(String(32), nullable=False)
    layers_per_chunk = Column(Integer, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    total_time_ms = Column(Float, nullable=False)
    tokens_per_sec = Column(Float, nullable=False)
    peak_vram_mb = Column(Float, nullable=False)
    disk_read_time_ms = Column(Float, nullable=False)
    pcie_transfer_time_ms = Column(Float, nullable=False)
    compute_time_ms = Column(Float, nullable=False)
    status = Column(String(32), default="completed", nullable=False)
    notes = Column(Text, default="", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    model = relationship("ModelSpecRecord", back_populates="benchmarks")
    hardware = relationship("HardwareProfileRecord", back_populates="benchmarks")
    plan = relationship("InferencePlanRecord", back_populates="benchmarks")


class CompatibilityAuditRecord(Base):
    """Audit record of Hugging Face runtime compatibility analysis."""
    
    __tablename__ = "compatibility_audits"

    id = Column(String(64), primary_key=True, index=True)
    target_model = Column(String(128), nullable=False)
    transformers_version = Column(String(64), nullable=False)
    torch_version = Column(String(64), nullable=False)
    has_stateful_attribute = Column(Boolean, nullable=False)
    dynamic_cache_compatible = Column(Boolean, nullable=False)
    rope_dimension_aligned = Column(Boolean, nullable=False)
    issues_summary = Column(Text, nullable=False)
    patch_recommendation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)


class AuditLogRecord(Base):
    """Security and mutation audit trail."""
    
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(128), nullable=False)
    role = Column(String(64), nullable=False)
    action = Column(String(64), nullable=False)
    resource = Column(String(128), nullable=False)
    details = Column(Text, default="", nullable=False)
    timestamp = Column(DateTime, default=utc_now, nullable=False)
