import math
import pytest
from app.models.entities import ModelSpecRecord, HardwareProfileRecord
from app.services.memory_planner import MemoryPlanner, BYTES_PER_PARAM


@pytest.fixture
def sample_70b_model():
    return ModelSpecRecord(
        id="test-70b",
        name="Test-70B",
        family="llama",
        parameter_count_billion=70.0,
        hidden_dim=8192,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        vocab_size=32000,
        max_seq_length=4096,
        intermediate_dim=28672,
        default_dtype="bfloat16",
    )


@pytest.fixture
def gtx1650_4gb_hardware():
    return HardwareProfileRecord(
        id="test-gtx-1650",
        name="GTX 1650 (4GB)",
        category="gpu",
        vram_gb=4.0,
        memory_bandwidth_gbps=128.0,
        pcie_gen=3,
        pcie_lanes=16,
        pcie_bandwidth_gbps=15.75,
        system_ram_gb=16.0,
        disk_read_mbps=550.0,
        compute_tflops_fp16=5.9,
    )


@pytest.fixture
def t4_16gb_hardware():
    return HardwareProfileRecord(
        id="test-t4-16gb",
        name="Tesla T4 (16GB)",
        category="gpu",
        vram_gb=16.0,
        memory_bandwidth_gbps=320.0,
        pcie_gen=3,
        pcie_lanes=16,
        pcie_bandwidth_gbps=15.75,
        system_ram_gb=32.0,
        disk_read_mbps=1200.0,
        compute_tflops_fp16=65.0,
    )


def test_calculate_layer_weights_quantizations(sample_70b_model):
    """Verify weight calculations scale proportionally across FP16, INT8, and INT4."""
    fp16_total, fp16_layer = MemoryPlanner.calculate_layer_weights_mb(sample_70b_model, "fp16")
    int8_total, int8_layer = MemoryPlanner.calculate_layer_weights_mb(sample_70b_model, "int8")
    int4_total, int4_layer = MemoryPlanner.calculate_layer_weights_mb(sample_70b_model, "int4")

    # INT4 should be roughly 1/4th the size of FP16
    assert int4_total < int8_total < fp16_total
    assert int4_layer < int8_layer < fp16_layer
    assert int4_total > 30000.0  # 70B in 4-bit is ~35-40 GB total
    assert int4_layer > 400.0   # Per layer ~450-500 MB


def test_calculate_kv_cache_scaling(sample_70b_model):
    """Verify KV cache size doubles when sequence length doubles."""
    kv_512 = MemoryPlanner.calculate_kv_cache_mb(sample_70b_model, batch_size=1, seq_len=512)
    kv_1024 = MemoryPlanner.calculate_kv_cache_mb(sample_70b_model, batch_size=1, seq_len=1024)

    assert kv_512 > 0
    assert abs(kv_1024 - (kv_512 * 2)) < 0.1


def test_calculate_activation_scaling(sample_70b_model):
    """Verify activation memory scales with layer count in chunk."""
    act_1 = MemoryPlanner.calculate_activation_mb(sample_70b_model, batch_size=1, seq_len=512, layers_in_chunk=1)
    act_4 = MemoryPlanner.calculate_activation_mb(sample_70b_model, batch_size=1, seq_len=512, layers_in_chunk=4)

    assert act_4 >= act_1
    assert act_1 >= 32.0


def test_solve_optimal_chunk_size_on_4gb_gpu(sample_70b_model, gtx1650_4gb_hardware):
    """Verify solver finds a chunk size that strictly fits within 4GB VRAM."""
    plan = MemoryPlanner.solve_optimal_chunk_size(
        model=sample_70b_model,
        hardware=gtx1650_4gb_hardware,
        quantization="int4",
        batch_size=1,
        seq_len=512,
    )

    max_vram_mb = gtx1650_4gb_hardware.vram_gb * 1024.0
    assert plan["layers_per_chunk"] >= 1
    assert plan["total_vram_mb"] <= max_vram_mb
    assert plan["vram_utilization_percent"] <= 100.0
    assert plan["total_chunks"] == math.ceil(80 / plan["layers_per_chunk"])
    assert plan["est_tokens_per_sec"] > 0.0


def test_solve_optimal_chunk_size_on_16gb_gpu(sample_70b_model, t4_16gb_hardware):
    """Verify solver batches more layers on 16GB GPU (resolving issue #44)."""
    plan_t4 = MemoryPlanner.solve_optimal_chunk_size(
        model=sample_70b_model,
        hardware=t4_16gb_hardware,
        quantization="int4",
        batch_size=1,
        seq_len=512,
    )

    # With 16GB VRAM, layers per chunk should be significantly higher than 1
    assert plan_t4["layers_per_chunk"] > 1
    assert plan_t4["total_vram_mb"] <= (t4_16gb_hardware.vram_gb * 1024.0)
