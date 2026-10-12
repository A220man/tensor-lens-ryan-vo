"""VRAM memory budget and dynamic layer chunking optimization planner."""
import math
from typing import Dict, Any, Tuple
from app.models.entities import ModelSpecRecord, HardwareProfileRecord


BYTES_PER_PARAM: Dict[str, float] = {
    "fp16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.05,    # 1 byte + block scale metadata
    "int4": 0.5625,  # 4 bits (0.5 B) + block-wise 64 scale/min offsets
}

# Driver and CUDA workspace buffer in MB
GPU_SYSTEM_RESERVE_MB: float = 512.0


class MemoryPlanner:
    """Calculates transformer layer footprints and solves multi-layer batch sizes."""

    @staticmethod
    def calculate_layer_weights_mb(model: ModelSpecRecord, quantization: str) -> Tuple[float, float]:
        """
        Compute total model weights in MB and per-layer weights in MB.
        Returns: (total_weights_mb, per_layer_weights_mb)
        """
        quant_lower = quantization.lower()
        bpp = BYTES_PER_PARAM.get(quant_lower, 0.5625)

        # Base transformer parameter breakdown:
        # Embedding: vocab_size * hidden_dim
        # Per layer:
        #   Attention: Q, K, V, O projections
        #     Q: hidden_dim * (num_heads * head_dim)
        #     K, V: 2 * hidden_dim * (num_kv_heads * head_dim)
        #     O: (num_heads * head_dim) * hidden_dim
        #   MLP: Gate, Up, Down projections (SwiGLU) = 3 * hidden_dim * intermediate_dim
        #   Norms: 2 * hidden_dim
        # Output LM Head: vocab_size * hidden_dim (often tied, but counted)
        head_dim = model.hidden_dim // model.num_heads
        q_params = model.hidden_dim * (model.num_heads * head_dim)
        kv_params = 2 * model.hidden_dim * (model.num_kv_heads * head_dim)
        o_params = (model.num_heads * head_dim) * model.hidden_dim
        mlp_params = 3 * model.hidden_dim * model.intermediate_dim
        norm_params = 2 * model.hidden_dim

        per_layer_params = q_params + kv_params + o_params + ml_params if False else (
            q_params + kv_params + o_params + mlp_params + norm_params
        )

        non_layer_params = 2 * model.vocab_size * model.hidden_dim  # embed_tokens + lm_head
        total_params = non_layer_params + (per_layer_params * model.num_layers)

        total_weights_mb = (total_params * bpp) / (1024 * 1024)
        per_layer_weights_mb = (per_layer_params * bpp) / (1024 * 1024)

        return round(total_weights_mb, 2), round(per_layer_weights_mb, 2)

    @staticmethod
    def calculate_kv_cache_mb(
        model: ModelSpecRecord,
        batch_size: int,
        seq_len: int,
        quantization: str = "fp16"
    ) -> float:
        """
        KV cache size for all layers:
        2 (K and V) * num_layers * num_kv_heads * head_dim * bytes_per_element * batch_size * seq_len
        """
        head_dim = model.hidden_dim // model.num_heads
        element_bytes = 1.0 if quantization.lower() in ("int8", "int4") else 2.0
        total_elements = 2 * model.num_layers * model.num_kv_heads * head_dim * batch_size * seq_len
        kv_mb = (total_elements * element_bytes) / (1024 * 1024)
        return round(kv_mb, 2)

    @staticmethod
    def calculate_activation_mb(
        model: ModelSpecRecord,
        batch_size: int,
        seq_len: int,
        layers_in_chunk: int
    ) -> float:
        """
        Peak activation memory for executing a chunk of layers.
        Includes QKV projection activations, attention logits (batch * heads * seq * seq),
        and SwiGLU hidden expansions (batch * seq * intermediate_dim).
        """
        # Memory in bytes for FP16 activations
        element_bytes = 2.0
        # Attention activations
        attn_acts = batch_size * model.num_heads * seq_len * (model.hidden_dim // model.num_heads)
        # Attention score matrix (B * H * S * S)
        score_acts = batch_size * model.num_heads * seq_len * min(seq_len, 2048)
        # MLP intermediate activations
        mlp_acts = 2 * batch_size * seq_len * model.intermediate_dim

        per_layer_activation = (attn_acts + score_acts + mlp_acts) * element_bytes
        total_activation_mb = (per_layer_activation * layers_in_chunk) / (1024 * 1024)
        return round(max(32.0, total_activation_mb), 2)

    @classmethod
    def solve_optimal_chunk_size(
        cls,
        model: ModelSpecRecord,
        hardware: HardwareProfileRecord,
        quantization: str,
        batch_size: int = 1,
        seq_len: int = 512,
        manual_chunk_size: int = None
    ) -> Dict[str, Any]:
        """
        Finds the maximum N layers per chunk that fits within GPU VRAM
        without triggering an Out-of-Memory (OOM) error.
        Resolves AirLLM user need #44 (More clever batching of layers).
        """
        total_vram_mb = hardware.vram_gb * 1024.0
        total_weights_mb, per_layer_weights_mb = cls.calculate_layer_weights_mb(model, quantization)
        kv_cache_mb = cls.calculate_kv_cache_mb(model, batch_size, seq_len, quantization)

        # Available VRAM for layer weights and dynamic activations
        available_vram_mb = total_vram_mb - GPU_SYSTEM_RESERVE_MB - kv_cache_mb

        if manual_chunk_size and manual_chunk_size > 0:
            n_layers = min(manual_chunk_size, model.num_layers)
        else:
            # Iteratively solve for maximum N where:
            # (N * per_layer_weights_mb) + activation_mb(N) <= available_vram_mb
            best_n = 1
            for n in range(1, model.num_layers + 1):
                chunk_act = cls.calculate_activation_mb(model, batch_size, seq_len, n)
                chunk_param = n * per_layer_weights_mb
                if (chunk_param + chunk_act) <= available_vram_mb:
                    best_n = n
                else:
                    break
            n_layers = best_n

        param_vram_mb = round(n_layers * per_layer_weights_mb, 2)
        activation_vram_mb = cls.calculate_activation_mb(model, batch_size, seq_len, n_layers)
        peak_vram_mb = round(param_vram_mb + activation_vram_mb + kv_cache_mb + GPU_SYSTEM_RESERVE_MB, 2)
        utilization = round((peak_vram_mb / total_vram_mb) * 100.0, 1)

        total_chunks = math.ceil(model.num_layers / n_layers)

        # Performance timing projections:
        # Disk read time for weights (SSD/NVMe)
        disk_speed_mb_s = hardware.disk_read_mbps
        disk_time_sec = total_weights_mb / max(disk_speed_mb_s, 50.0)

        # PCIe transfer time (Host RAM -> GPU VRAM)
        # PCIe gen4 x16 ~ 25 GB/s, effective ~ 70%
        pcie_speed_mb_s = (hardware.pcie_bandwidth_gbps * 1024.0) * 0.75
        pcie_time_sec = total_weights_mb / max(pcie_speed_mb_s, 100.0)

        # Compute FLOPs = 2 * param_count * tokens
        total_params = model.parameter_count_billion * 1e9
        compute_flops_per_token = 2.0 * total_params
        gpu_tflops = hardware.compute_tflops_fp16 * 1e12
        compute_time_per_token = compute_flops_per_token / max(gpu_tflops, 1e9)

        # With layer chunking, each token requires streaming through all chunks:
        # Pipelined disk/PCIe + compute per chunk
        time_per_token_sec = (disk_time_sec + pcie_time_sec) * 0.85 + compute_time_per_token
        est_tokens_per_sec = round(1.0 / max(0.001, time_per_token_sec), 2)

        est_prefill_sec = round(disk_time_sec + pcie_time_sec + (compute_time_per_token * (seq_len / 64.0)), 2)
        est_generation_sec = round(time_per_token_sec, 3)

        return {
            "layers_per_chunk": n_layers,
            "total_chunks": total_chunks,
            "param_vram_mb": param_vram_mb,
            "activation_vram_mb": activation_vram_mb,
            "kv_cache_vram_mb": kv_cache_mb,
            "total_vram_mb": peak_vram_mb,
            "vram_utilization_percent": min(100.0, utilization),
            "est_prefill_sec": est_prefill_sec,
            "est_generation_sec": est_generation_sec,
            "est_tokens_per_sec": est_tokens_per_sec,
            "total_model_weights_mb": total_weights_mb,
            "per_layer_weights_mb": per_layer_weights_mb,
        }
