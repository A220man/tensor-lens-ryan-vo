"""Layer-streaming inference execution engine and numerical tensor simulator."""
import time
import math
from typing import List, Dict, Any, Tuple
import numpy as np
from app.models.entities import ModelSpecRecord, HardwareProfileRecord
from app.models.schemas import LayerStreamStep, BenchmarkStreamSimulationResponse
from app.services.memory_planner import MemoryPlanner


class TransformerOps:
    """Numerical implementations of modern transformer primitives."""

    @staticmethod
    def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Root Mean Square Layer Normalization."""
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(variance + eps) * weight

    @staticmethod
    def apply_rotary_emb(
        x: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray
    ) -> np.ndarray:
        """
        Apply Rotary Position Embeddings (RoPE) with explicit dimension validation.
        Addresses user issue #258 dimension mismatch.
        """
        d = x.shape[-1]
        if d % 2 != 0:
            raise ValueError(f"Rotary dimension must be even, got {d}")
        
        # Split into even/odd halves or rotate half
        half = d // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        # Rotate: [-x2, x1]
        x_rot = np.concatenate((-x2, x1), axis=-1)
        return (x * cos) + (x_rot * sin)

    @staticmethod
    def swiglu_forward(
        x: np.ndarray,
        w_gate: np.ndarray,
        w_up: np.ndarray,
        w_down: np.ndarray
    ) -> np.ndarray:
        """SwiGLU activation and feed-forward projection."""
        gate = x @ w_gate
        # SiLU(gate) = gate * sigmoid(gate)
        silu_gate = gate * (1.0 / (1.0 + np.exp(-np.clip(gate, -20.0, 20.0))))
        up = x @ w_up
        hidden = silu_gate * up
        return hidden @ w_down

    @staticmethod
    def attention_forward(
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        num_heads: int,
        num_kv_heads: int
    ) -> np.ndarray:
        """Grouped-Query Attention forward pass."""
        b, s, d = q.shape
        head_dim = d // num_heads
        
        # Reshape to (batch, heads, seq, head_dim)
        q_heads = q.reshape(b, s, num_heads, head_dim).swapaxes(1, 2)
        k_heads = k.reshape(b, s, num_kv_heads, head_dim).swapaxes(1, 2)
        v_heads = v.reshape(b, s, num_kv_heads, head_dim).swapaxes(1, 2)

        # Expand KV heads if Grouped-Query Attention
        if num_heads != num_kv_heads:
            ratio = num_heads // num_kv_heads
            k_heads = np.repeat(k_heads, ratio, axis=1)
            v_heads = np.repeat(v_heads, ratio, axis=1)

        scale = 1.0 / math.sqrt(head_dim)
        scores = (q_heads @ k_heads.swapaxes(-1, -2)) * scale
        
        # Softmax over last dimension
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        out = probs @ v_heads
        # Reshape back to (batch, seq, d)
        return out.swapaxes(1, 2).reshape(b, s, d)


class LayerStreamingEngine:
    """Simulates and executes layer streaming with chunking and offloading."""

    @classmethod
    def execute_stream_simulation(
        cls,
        model: ModelSpecRecord,
        hardware: HardwareProfileRecord,
        quantization: str = "int4",
        layers_per_chunk: int = 1,
        prompt_tokens: int = 32,
        output_tokens: int = 16,
        run_numerical_check: bool = True
    ) -> BenchmarkStreamSimulationResponse:
        """
        Simulates end-to-end layer streaming execution with staged transfer timings,
        VRAM allocation watermarks, and optional numerical tensor verification.
        """
        plan = MemoryPlanner.solve_optimal_chunk_size(
            model=model,
            hardware=hardware,
            quantization=quantization,
            batch_size=1,
            seq_len=prompt_tokens + output_tokens,
            manual_chunk_size=layers_per_chunk
        )

        n_chunk = plan["layers_per_chunk"]
        total_chunks = plan["total_chunks"]
        per_layer_mb = plan["per_layer_weights_mb"]

        steps: List[LayerStreamStep] = []
        
        # Compute transfer latencies per chunk
        chunk_weights_mb = n_chunk * per_layer_mb
        disk_time_ms = (chunk_weights_mb / max(50.0, hardware.disk_read_mbps)) * 1000.0
        pcie_bw_mb_s = (hardware.pcie_bandwidth_gbps * 1024.0) * 0.75
        pcie_time_ms = (chunk_weights_mb / max(100.0, pcie_bw_mb_s)) * 1000.0

        # Compute execution time per chunk
        # Chunk FLOPs = 2 * (params_per_chunk) * tokens
        params_per_chunk = (model.parameter_count_billion * 1e9 / model.num_layers) * n_chunk
        chunk_flops = 2.0 * params_per_chunk * 1  # per token
        gpu_flops_per_ms = (hardware.compute_tflops_fp16 * 1e12) / 1000.0
        compute_time_ms = max(0.5, chunk_flops / max(1e6, gpu_flops_per_ms))

        # Base VRAM: Reserved + KV cache
        kv_cache_mb = plan["kv_cache_vram_mb"]
        base_vram = 512.0 + kv_cache_mb

        for chunk_idx in range(total_chunks):
            start_layer = chunk_idx * n_chunk
            end_layer = min(model.num_layers, start_layer + n_chunk)
            layers_in_this_chunk = end_layer - start_layer

            # Dynamic VRAM during chunk execution: Base + chunk weights + activation
            active_weights_mb = layers_in_this_chunk * per_layer_mb
            active_act_mb = MemoryPlanner.calculate_activation_mb(
                model=model,
                batch_size=1,
                seq_len=prompt_tokens,
                layers_in_chunk=layers_in_this_chunk
            )
            vram_peak = round(base_vram + active_weights_mb + active_act_mb, 2)

            step = LayerStreamStep(
                layer_idx=start_layer,
                chunk_idx=chunk_idx,
                stage=f"Layers {start_layer}..{end_layer - 1}",
                disk_read_ms=round(disk_time_ms, 2),
                pcie_transfer_ms=round(pcie_time_ms, 2),
                compute_ms=round(compute_time_ms, 2),
                vram_allocated_mb=vram_peak,
                description=(
                    f"Prefetched {layers_in_this_chunk} layer(s) ({round(active_weights_mb, 1)} MB), "
                    f"streamed Host->Device via PCIe Gen{hardware.pcie_gen}, executed forward pass, evicted."
                )
            )
            steps.append(step)

        # Perform numerical sanity check on small synthetic dimensions
        numerical_ok = True
        if run_numerical_check:
            try:
                test_dim = 64
                test_seq = 4
                x = np.random.randn(1, test_seq, test_dim).astype(np.float32)
                norm_w = np.ones(test_dim, dtype=np.float32)
                normed = TransformerOps.rms_norm(x, norm_w)

                cos = np.ones((1, test_seq, test_dim), dtype=np.float32)
                sin = np.zeros((1, test_seq, test_dim), dtype=np.float32)
                rotated = TransformerOps.apply_rotary_emb(normed, cos, sin)

                attn_out = TransformerOps.attention_forward(
                    q=rotated, k=rotated, v=rotated, num_heads=4, num_kv_heads=4
                )
                numerical_ok = not np.isnan(attn_out).any() and attn_out.shape == (1, test_seq, test_dim)
            except Exception:
                numerical_ok = False

        return BenchmarkStreamSimulationResponse(
            benchmark_id=f"sim-{int(time.time())}",
            total_layers=model.num_layers,
            chunks_count=total_chunks,
            steps=steps,
            final_tokens_per_sec=plan["est_tokens_per_sec"],
            peak_vram_mb=plan["total_vram_mb"],
            numerical_check_passed=numerical_ok
        )
