"""Storage repository, default seeds, and audit logging for TensorLens."""
from typing import List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models.entities import (
    HardwareProfileRecord,
    ModelSpecRecord,
    InferencePlanRecord,
    BenchmarkRunRecord,
    AuditLogRecord,
)


def seed_defaults(db: Session) -> None:
    """Populate database with standard hardware profiles and model architectures."""
    # Seed hardware profiles if empty
    if db.query(HardwareProfileRecord).count() == 0:
        default_hardware = [
            HardwareProfileRecord(
                id="gpu-gtx-1650-4gb",
                name="NVIDIA GeForce GTX 1650 (4GB)",
                category="gpu",
                vram_gb=4.0,
                memory_bandwidth_gbps=128.0,
                pcie_gen=3,
                pcie_lanes=16,
                pcie_bandwidth_gbps=15.75,
                system_ram_gb=16.0,
                disk_read_mbps=550.0,
                compute_tflops_fp16=5.9,
            ),
            HardwareProfileRecord(
                id="gpu-tesla-k80-12gb",
                name="NVIDIA Tesla K80 (12GB per die)",
                category="gpu",
                vram_gb=12.0,
                memory_bandwidth_gbps=240.0,
                pcie_gen=3,
                pcie_lanes=16,
                pcie_bandwidth_gbps=15.75,
                system_ram_gb=64.0,
                disk_read_mbps=450.0,
                compute_tflops_fp16=8.7,
            ),
            HardwareProfileRecord(
                id="gpu-t4-16gb",
                name="NVIDIA Tesla T4 (16GB)",
                category="gpu",
                vram_gb=16.0,
                memory_bandwidth_gbps=320.0,
                pcie_gen=3,
                pcie_lanes=16,
                pcie_bandwidth_gbps=15.75,
                system_ram_gb=32.0,
                disk_read_mbps=1200.0,
                compute_tflops_fp16=65.0,
            ),
            HardwareProfileRecord(
                id="gpu-rtx-3060-12gb",
                name="NVIDIA GeForce RTX 3060 (12GB)",
                category="gpu",
                vram_gb=12.0,
                memory_bandwidth_gbps=360.0,
                pcie_gen=4,
                pcie_lanes=16,
                pcie_bandwidth_gbps=31.5,
                system_ram_gb=32.0,
                disk_read_mbps=2500.0,
                compute_tflops_fp16=12.7,
            ),
            HardwareProfileRecord(
                id="gpu-rx-7900-xtx-24gb",
                name="AMD Radeon RX 7900 XTX (24GB ROCm)",
                category="rocm",
                vram_gb=24.0,
                memory_bandwidth_gbps=960.0,
                pcie_gen=4,
                pcie_lanes=16,
                pcie_bandwidth_gbps=31.5,
                system_ram_gb=64.0,
                disk_read_mbps=5000.0,
                compute_tflops_fp16=122.8,
            ),
            HardwareProfileRecord(
                id="cpu-workstation-64gb",
                name="x86_64 High-Memory Workstation (64GB RAM)",
                category="cpu",
                vram_gb=32.0,
                memory_bandwidth_gbps=51.2,
                pcie_gen=4,
                pcie_lanes=16,
                pcie_bandwidth_gbps=31.5,
                system_ram_gb=64.0,
                disk_read_mbps=3500.0,
                compute_tflops_fp16=2.4,
            ),
        ]
        db.add_all(default_hardware)
        db.commit()

    # Seed model specs if empty
    if db.query(ModelSpecRecord).count() == 0:
        default_models = [
            ModelSpecRecord(
                id="model-llama3-70b",
                name="Meta Llama-3-70B-Instruct",
                family="llama",
                parameter_count_billion=70.6,
                hidden_dim=8192,
                num_layers=80,
                num_heads=64,
                num_kv_heads=8,
                vocab_size=128256,
                max_seq_length=8192,
                intermediate_dim=28672,
                default_dtype="bfloat16",
            ),
            ModelSpecRecord(
                id="model-llama2-70b",
                name="Llama-2-70B-Chat",
                family="llama",
                parameter_count_billion=68.9,
                hidden_dim=8192,
                num_layers=80,
                num_heads=64,
                num_kv_heads=8,
                vocab_size=32000,
                max_seq_length=4096,
                intermediate_dim=28672,
                default_dtype="float16",
            ),
            ModelSpecRecord(
                id="model-qwen25-72b",
                name="Qwen-2.5-72B-Instruct",
                family="qwen",
                parameter_count_billion=72.7,
                hidden_dim=8192,
                num_layers=80,
                num_heads=64,
                num_kv_heads=8,
                vocab_size=152064,
                max_seq_length=32768,
                intermediate_dim=29568,
                default_dtype="bfloat16",
            ),
            ModelSpecRecord(
                id="model-mistral-7b",
                name="Mistral-7B-Instruct-v0.3",
                family="mistral",
                parameter_count_billion=7.2,
                hidden_dim=4096,
                num_layers=32,
                num_heads=32,
                num_kv_heads=8,
                vocab_size=32768,
                max_seq_length=32768,
                intermediate_dim=14336,
                default_dtype="bfloat16",
            ),
        ]
        db.add_all(default_models)
        db.commit()


def log_audit_action(
    db: Session,
    user_id: str,
    role: str,
    action: str,
    resource: str,
    details: str = ""
) -> None:
    """Record an audit trail event for state modifications and secure operations."""
    audit_entry = AuditLogRecord(
        id=f"audit-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        user_id=user_id,
        role=role,
        action=action,
        resource=resource,
        details=details,
        timestamp=datetime.now(timezone.utc),
    )
    db.add(audit_entry)
    db.commit()
