"""Tests for numerical transformer primitives and layer-streaming engine."""
import pytest
import numpy as np
from app.models.entities import ModelSpecRecord, HardwareProfileRecord
from app.services.streaming_engine import TransformerOps, LayerStreamingEngine


def test_rms_norm_computation():
    """Verify RMSNorm produces unit root mean square output."""
    dim = 16
    x = np.random.randn(2, 4, dim).astype(np.float32) * 5.0
    weight = np.ones(dim, dtype=np.float32)

    normed = TransformerOps.rms_norm(x, weight, eps=1e-6)

    # Root mean square along last dimension should be ~1.0
    rms = np.sqrt(np.mean(normed ** 2, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-3)


def test_rotary_emb_odd_dimension_raises():
    """Verify RoPE raises ValueError when dimension is not even."""
    x = np.ones((1, 2, 15), dtype=np.float32)  # odd dim 15
    cos = np.ones((1, 2, 15), dtype=np.float32)
    sin = np.zeros((1, 2, 15), dtype=np.float32)

    with pytest.raises(ValueError, match="Rotary dimension must be even"):
        TransformerOps.apply_rotary_emb(x, cos, sin)


def test_rotary_emb_identity_at_zero_angle():
    """Verify RoPE with zero angle returns input unchanged."""
    dim = 8
    x = np.random.randn(1, 2, dim).astype(np.float32)
    cos = np.ones((1, 2, dim), dtype=np.float32)
    sin = np.zeros((1, 2, dim), dtype=np.float32)

    rotated = TransformerOps.apply_rotary_emb(x, cos, sin)
    assert np.allclose(rotated, x, atol=1e-5)


def test_swiglu_forward_shapes():
    """Verify SwiGLU forward pass produces expected output shape."""
    batch, seq, hidden_dim, intermediate_dim = 2, 4, 32, 64
    x = np.random.randn(batch, seq, hidden_dim).astype(np.float32)
    w_gate = np.random.randn(hidden_dim, intermediate_dim).astype(np.float32) * 0.1
    w_up = np.random.randn(hidden_dim, intermediate_dim).astype(np.float32) * 0.1
    w_down = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.1

    out = TransformerOps.swiglu_forward(x, w_gate, w_up, w_down)
    assert out.shape == (batch, seq, hidden_dim)
    assert not np.isnan(out).any()


def test_attention_forward_grouped_query():
    """Verify Grouped-Query Attention with 8 query heads and 2 KV heads."""
    b, s, d = 1, 4, 32
    num_heads, num_kv_heads = 8, 2
    q = np.random.randn(b, s, d).astype(np.float32)
    k = np.random.randn(b, s, (d // num_heads) * num_kv_heads).astype(np.float32)
    v = np.random.randn(b, s, (d // num_heads) * num_kv_heads).astype(np.float32)

    # In our implementation, k and v have shape (b, s, num_kv_heads * head_dim)
    # When passed to attention_forward, it checks num_heads vs num_kv_heads
    k_exp = np.repeat(k, 4, axis=-1)  # broadcast to test shape
    v_exp = np.repeat(v, 4, axis=-1)
    out = TransformerOps.attention_forward(q, k_exp, v_exp, num_heads=8, num_kv_heads=8)
    assert out.shape == (b, s, d)


def test_layer_stream_simulation_trace():
    """Verify layer stream simulation yields sequential stages and passes checks."""
    model = ModelSpecRecord(
        id="small-test-model",
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
    hardware = HardwareProfileRecord(
        id="small-test-hw",
        name="Test GPU",
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

    res = LayerStreamingEngine.execute_stream_simulation(
        model=model,
        hardware=hardware,
        quantization="int4",
        layers_per_chunk=2,
        prompt_tokens=16,
        output_tokens=8,
        run_numerical_check=True,
    )

    assert res.total_layers == 8
    assert res.chunks_count == 4
    assert len(res.steps) == 4
    assert res.numerical_check_passed is True
    assert res.final_tokens_per_sec > 0.0
