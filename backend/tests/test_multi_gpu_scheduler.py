import pytest
from app.models.entities import ModelSpecRecord, HardwareProfileRecord
from app.services.multi_gpu_scheduler import MultiGPUScheduler


@pytest.fixture
def small_model():
    return ModelSpecRecord(
        id="test-small",
        name="Small-Test",
        family="llama",
        parameter_count_billion=7.0,
        hidden_dim=4096,
        num_layers=8,
        num_heads=32,
        num_kv_heads=8,
        vocab_size=32000,
        max_seq_length=2048,
        intermediate_dim=11008,
        default_dtype="float16",
    )


@pytest.fixture
def gpu_8gb():
    return HardwareProfileRecord(
        id="test-gpu-8gb",
        name="Test GPU 8GB",
        category="gpu",
        vram_gb=8.0,
        memory_bandwidth_gbps=256.0,
        pcie_gen=3,
        pcie_lanes=16,
        pcie_bandwidth_gbps=15.75,
        system_ram_gb=16.0,
        disk_read_mbps=800.0,
        compute_tflops_fp16=20.0,
    )


@pytest.fixture
def gpu_4gb():
    return HardwareProfileRecord(
        id="test-gpu-4gb",
        name="Test GPU 4GB",
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


def test_scheduler_assigns_layers(small_model, gpu_8gb, gpu_4gb):
    """Verify scheduler splits layers across GPUs respecting capacity."""
    assignments = MultiGPUScheduler.distribute_layers(
        model=small_model,
        hardware_list=[gpu_8gb, gpu_4gb],
        quantization="int4",
        batch_size=1,
        seq_len=512,
    )
    # Both GPUs should receive some layers (or at least one)
    assert len(assignments) == 2
    assert gpu_8gb.id in assignments
    assert gpu_4gb.id in assignments
    # Total assigned layers should equal model.num_layers (or less if capacity insufficient)
    total_assigned = sum(len(layers) for layers in assignments.values())
    assert total_assigned == small_model.num_layers
    # Verify each assigned layer index is within [0, num_layers)
    for hw_id, layers in assignments.items():
        for layer in layers:
            assert 0 <= layer < small_model.num_layers
    # Verify that the assignment is contiguous and non-overlapping
    all_assigned = sorted([l for layers in assignments.values() for l in layers])
    assert all_assigned == list(range(small_model.num_layers))
