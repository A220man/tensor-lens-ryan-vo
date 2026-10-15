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


HW_SEEDS = [
    ("gpu-gtx-1650-4gb", "NVIDIA GeForce GTX 1650 (4GB)", "gpu", 4.0, 128.0, 3, 16, 15.75, 16.0, 550.0, 5.9),
    ("gpu-tesla-k80-12gb", "NVIDIA Tesla K80 (12GB per die)", "gpu", 12.0, 240.0, 3, 16, 15.75, 64.0, 450.0, 8.7),
    ("gpu-t4-16gb", "NVIDIA Tesla T4 (16GB)", "gpu", 16.0, 320.0, 3, 16, 15.75, 32.0, 1200.0, 65.0),
    ("gpu-rtx-3060-12gb", "NVIDIA GeForce RTX 3060 (12GB)", "gpu", 12.0, 360.0, 4, 16, 31.5, 32.0, 2500.0, 12.7),
    ("gpu-rx-7900-xtx-24gb", "AMD Radeon RX 7900 XTX (24GB ROCm)", "rocm", 24.0, 960.0, 4, 16, 31.5, 64.0, 5000.0, 122.8),
    ("cpu-workstation-64gb", "x86_64 High-Memory Workstation (64GB RAM)", "cpu", 32.0, 51.2, 4, 16, 31.5, 64.0, 3500.0, 2.4),
]

MODEL_SEEDS = [
    ("model-llama3-70b", "Meta Llama-3-70B-Instruct", "llama", 70.6, 8192, 80, 64, 8, 128256, 8192, 28672, "bfloat16"),
    ("model-llama2-70b", "Llama-2-70B-Chat", "llama", 68.9, 8192, 80, 64, 8, 32000, 4096, 28672, "float16"),
    ("model-qwen25-72b", "Qwen-2.5-72B-Instruct", "qwen", 72.7, 8192, 80, 64, 8, 152064, 32768, 29568, "bfloat16"),
    ("model-mistral-7b", "Mistral-7B-Instruct-v0.3", "mistral", 7.2, 4096, 32, 32, 8, 32768, 32768, 14336, "bfloat16"),
]


def seed_defaults(db: Session) -> None:
    """Populate database with standard hardware profiles and model architectures."""
    if db.query(HardwareProfileRecord).count() == 0:
        db.add_all([
            HardwareProfileRecord(
                id=h[0], name=h[1], category=h[2], vram_gb=h[3], memory_bandwidth_gbps=h[4],
                pcie_gen=h[5], pcie_lanes=h[6], pcie_bandwidth_gbps=h[7], system_ram_gb=h[8],
                disk_read_mbps=h[9], compute_tflops_fp16=h[10]
            ) for h in HW_SEEDS
        ])
        db.commit()

    if db.query(ModelSpecRecord).count() == 0:
        db.add_all([
            ModelSpecRecord(
                id=m[0], name=m[1], family=m[2], parameter_count_billion=m[3], hidden_dim=m[4],
                num_layers=m[5], num_heads=m[6], num_kv_heads=m[7], vocab_size=m[8],
                max_seq_length=m[9], intermediate_dim=m[10], default_dtype=m[11]
            ) for m in MODEL_SEEDS
        ])
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
    now = datetime.now(timezone.utc)
    db.add(AuditLogRecord(
        id=f"audit-{int(now.timestamp() * 1000)}",
        user_id=user_id,
        role=role,
        action=action,
        resource=resource,
        details=details,
        timestamp=now,
    ))
    db.commit()
