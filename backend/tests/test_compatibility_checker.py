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


def test_audit_generates_layer_offloading_hooks():
    """Verify generation of layer offloading hooks (Issue #44)."""
    res = CompatibilityChecker.audit_environment(
        model_name="Llama-3-70B",
        transformers_version="4.45.0",
        torch_version="2.4.0",
        target_device="cuda",
        head_dim=128,
        rotary_dim=128,
    )
    assert "LayerOffloadHook" in res.generated_patch
    assert "pre_forward" in res.generated_patch
    assert "post_forward" in res.generated_patch


def test_run_verification_suite_all_pass():
    """Verify automated verification test suite passes when patches are applied."""
    res = CompatibilityChecker.run_verification_suite(
        model_name="Llama-3-70B",
        transformers_version="4.45.0",
        torch_version="2.4.0",
        target_device="cuda",
        head_dim=128,
        rotary_dim=64,
    )
    assert res.all_passed is True
    assert res.total_tests == 4
    assert res.passed_tests == 4
    assert res.failed_tests == 0
    assert len(res.results) == 4
    assert all(r.passed for r in res.results)


def test_run_verification_suite_detects_failures():
    """Verify verification suite catches missing patches in breaking environments."""
    # Pass empty patch_code for environment needing stateful flag and aligned rope
    res = CompatibilityChecker.run_verification_suite(
        model_name="Llama-3-70B",
        transformers_version="4.45.0",
        torch_version="2.4.0",
        target_device="cuda",
        head_dim=128,
        rotary_dim=64,
        patch_code="# Incomplete dummy patch without required shims"
    )
    assert res.all_passed is False
    assert res.failed_tests > 0
    test_map = {r.name: r for r in res.results}
    assert test_map["GenerationMixin _is_stateful flag"].passed is False
    assert test_map["RoPE rotary embedding dimension alignment"].passed is False
    assert test_map["Layer offload hook lifecycle (GPU/CPU offload)"].passed is False
