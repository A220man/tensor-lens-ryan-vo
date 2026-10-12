"""Tests for Hugging Face runtime compatibility auditor and patch generator."""
from app.services.compatibility_checker import CompatibilityChecker


def test_audit_detects_stateful_flag_missing_in_transformers_4_45():
    """Verify detection of _is_stateful requirement in transformers >= 4.40 (Issue #258)."""
    res = CompatibilityChecker.audit_environment(
        model_name="Llama-2-70B",
        transformers_version="4.45.2",
        torch_version="2.4.0",
        target_device="cuda",
        head_dim=128,
        rotary_dim=128,
    )

    assert res.has_stateful_attribute is False
    assert any("_is_stateful" in issue for issue in res.issues)
    assert "_is_stateful" in res.generated_patch


def test_audit_detects_dynamic_cache_in_transformers_4_40():
    """Verify detection of DynamicCache subscripting issue (Issue #258)."""
    res = CompatibilityChecker.audit_environment(
        model_name="Llama-2-70B",
        transformers_version="4.40.1",
        torch_version="2.4.0",
        target_device="cuda",
        head_dim=128,
        rotary_dim=128,
    )

    assert res.dynamic_cache_compatible is False
    assert any("DynamicCache" in issue for issue in res.issues)
    assert "DynamicCache adapter wrapper" in res.generated_patch


def test_audit_detects_rope_dimension_mismatch():
    """Verify detection of head_dim != rotary_dim mismatch (Issue #258)."""
    res = CompatibilityChecker.audit_environment(
        model_name="Custom-Llama",
        transformers_version="4.35.0",
        torch_version="2.3.0",
        target_device="cuda",
        head_dim=128,
        rotary_dim=64,  # Partial RoPE dimension
    )

    assert res.rope_dimension_aligned is False
    assert any("does not match rotary embedding dimension" in issue for issue in res.issues)
    assert "apply_aligned_rope" in res.generated_patch


def test_audit_rocm_device_recommendations():
    """Verify ROCm / DirectML environment recommendations (Issue #301)."""
    res = CompatibilityChecker.audit_environment(
        model_name="TinyLlama",
        transformers_version="4.35.0",
        torch_version="2.3.0",
        target_device="rocm",
        head_dim=64,
        rotary_dim=64,
    )

    assert any("HSA_OVERRIDE_GFX_VERSION" in rec for rec in res.recommendations)
    assert "TORCH_DIRECTML_EXT_ENABLED" in res.generated_patch
