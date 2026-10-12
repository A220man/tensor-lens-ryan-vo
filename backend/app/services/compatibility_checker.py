import re
from datetime import datetime, timezone
from typing import List, Dict, Any
from app.models.schemas import CompatibilityAuditResponse


class CompatibilityChecker:
    """Analyzes runtime environments and detects breaking framework changes."""

    @classmethod
    def audit_environment(
        cls,
        model_name: str,
        transformers_version: str,
        torch_version: str,
        target_device: str,
        head_dim: int,
        rotary_dim: int
    ) -> CompatibilityAuditResponse:
        """
        Inspects model configurations and library versions to identify runtime issues
        documented in AirLLM user issues (#258, #301, #317).
        """
        issues: List[str] = []
        recommendations: List[str] = []
        patch_lines: List[str] = []

        # Parse transformers version
        tf_match = re.search(r"(\d+)\.(\d+)", transformers_version)
        tf_major, tf_minor = (int(tf_match.group(1)), int(tf_match.group(2))) if tf_match else (4, 40)

        # 1. Check _is_stateful attribute (Issue #258)
        has_stateful = True
        if tf_major > 4 or (tf_major == 4 and tf_minor >= 40):
            has_stateful = False
            issues.append(
                f"Transformers {transformers_version} introduces stateful generation checks; "
                "custom GenerationMixin base classes without '_is_stateful = False' raise AttributeError."
            )
            recommendations.append(
                "Inject '_is_stateful = False' into custom GenerationMixin model subclasses."
            )
            patch_lines.append(
                "# Patch 1: GenerationMixin stateful flag for transformers >= 4.40\n"
                "if not hasattr(model.__class__, '_is_stateful'):\n"
                "    setattr(model.__class__, '_is_stateful', False)"
            )

        # 2. Check DynamicCache subscriptability (Issue #258)
        dynamic_cache_ok = True
        if tf_major > 4 or (tf_major == 4 and tf_minor >= 36):
            dynamic_cache_ok = False
            issues.append(
                f"Transformers {transformers_version} uses DynamicCache instead of tuple-based past_key_values. "
                "Direct integer indexing (cache[i]) will raise TypeError: 'DynamicCache' object is not subscriptable."
            )
            recommendations.append(
                "Access key/value states using 'cache.key_cache[i]' or 'cache.get_usable_length()' instead of subscripting."
            )
            patch_lines.append(
                "\n# Patch 2: DynamicCache adapter wrapper\n"
                "def get_layer_cache(cache, layer_idx):\n"
                "    if hasattr(cache, 'key_cache'):\n"
                "        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]\n"
                "    return cache[layer_idx]"
            )

        # 3. Check RoPE rotary dimension alignment (Issue #258)
        rope_aligned = True
        if head_dim != rotary_dim:
            rope_aligned = False
            issues.append(
                f"Attention head dimension ({head_dim}) does not match rotary embedding dimension ({rotary_dim}). "
                "Partial rotary application requires slice alignment before elementwise operations."
            )
            recommendations.append(
                "Align rotary frequencies to head_dim or apply rotary projection only to q[..., :rotary_dim]."
            )
            patch_lines.append(
                f"\n# Patch 3: Partial RoPE rotary alignment (rotary_dim={rotary_dim})\n"
                "def apply_aligned_rope(q, k, cos, sin):\n"
                f"    q_rot = q[..., :{rotary_dim}]\n"
                f"    q_pass = q[..., {rotary_dim}:]\n"
                "    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)\n"
                "    return torch.cat([q_rot, q_pass], dim=-1)"
            )

        # 4. Check device and ROCm / DirectML configuration (Issue #301)
        device_compatible = True
        dev_lower = target_device.lower()
        if "rocm" in dev_lower or "directml" in dev_lower:
            device_compatible = True
            recommendations.append(
                "For AMD ROCm / DirectML: set 'HSA_OVERRIDE_GFX_VERSION=11.0.0' and ensure Triton ROCm wheels match torch ROCm ABI."
            )
            patch_lines.append(
                "\n# Patch 4: DirectML / ROCm environment tuning\n"
                "import os\n"
                "os.environ.setdefault('TORCH_DIRECTML_EXT_ENABLED', '1')\n"
                "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')"
            )

        if not issues:
            issues.append("No critical version drift issues detected for standard configuration.")
            recommendations.append("Environment matches supported transformer streaming invariants.")
            patch_lines.append("# No patch required: environment is compatible.")

        generated_patch = "\n".join(patch_lines)

        audit_id = f"audit-{int(abs(hash(model_name + transformers_version)) % 1000000)}"

        return CompatibilityAuditResponse(
            id=audit_id,
            target_model=model_name,
            transformers_version=transformers_version,
            torch_version=torch_version,
            has_stateful_attribute=has_stateful,
            dynamic_cache_compatible=dynamic_cache_ok,
            rope_dimension_aligned=rope_aligned,
            device_compatible=device_compatible,
            issues=issues,
            recommendations=recommendations,
            generated_patch=generated_patch,
            created_at=datetime.now(timezone.utc)
        )
