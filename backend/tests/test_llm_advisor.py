"""Tests for LLM advisor integration, offline heuristics, and error handling."""
import pytest
from unittest.mock import AsyncMock, patch
from app.services.llm_advisor import LlmAdvisor
from app.models.entities import InferencePlanRecord, BenchmarkRunRecord


@pytest.fixture
def sample_plan_record():
    return InferencePlanRecord(
        id="plan-test-1",
        model_id="model-llama3-70b",
        hardware_id="gpu-gtx-1650-4gb",
        quantization="int4",
        batch_size=1,
        seq_len=512,
        layers_per_chunk=1,
        total_chunks=80,
        param_vram_mb=450.0,
        activation_vram_mb=128.0,
        kv_cache_vram_mb=160.0,
        total_vram_mb=1250.0,
        vram_utilization_percent=30.5,
        est_prefill_sec=32.0,
        est_generation_sec=1.5,
        est_tokens_per_sec=0.67,
    )


@pytest.mark.asyncio
async def test_offline_deterministic_heuristics_when_no_key(sample_plan_record):
    """Verify deterministic grounded heuristic advice is returned when LLM_API_KEY is empty."""
    advice = await LlmAdvisor.request_advisory(
        plan=sample_plan_record,
        benchmark=None,
        user_query=None,
    )

    assert advice["provider"] == "offline-deterministic-fallback"
    assert advice["is_grounded"] is True
    assert "Deterministic Advisory" in advice["advisory_label"]
    assert len(advice["recommendations"]) > 0
    # Because utilization is 30.5% (< 50%), it should suggest increasing chunk size
    assert any("Increase layers per chunk" in r for r in advice["recommendations"])


@pytest.mark.asyncio
async def test_llm_advisor_with_mocked_openai_response(monkeypatch, sample_plan_record):
    """Verify adapter formatting with mocked external provider."""
    from app.core import config
    monkeypatch.setattr(config.settings, "LLM_API_KEY", "test-mock-key")
    monkeypatch.setattr(config.settings, "LLM_PROVIDER", "openai-compatible")
    monkeypatch.setattr(config.settings, "LLM_MODEL", "gpt-4o-mini")

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "Increase batching chunk size to N=2 to improve PCIe transfer efficiency."
                }
            }
        ]
    }

    class MockAsyncResponse:
        def raise_for_status(self): pass
        def json(self): return mock_response

    class MockAsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): return MockAsyncResponse()

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        advice = await LlmAdvisor.request_advisory(
            plan=sample_plan_record,
            benchmark=None,
        )

    assert advice["provider"] == "openai-compatible"
    assert advice["is_grounded"] is True
    assert "Increase batching chunk size" in advice["analysis_text"]


@pytest.mark.asyncio
async def test_llm_advisor_error_handling_and_redaction(monkeypatch, sample_plan_record):
    """Verify provider communication errors are caught and never leak secrets."""
    from app.core import config
    monkeypatch.setattr(config.settings, "LLM_API_KEY", "test-mock-key")
    monkeypatch.setattr(config.settings, "LLM_PROVIDER", "openai-compatible")

    class FailingAsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs):
            raise RuntimeError("Connection timed out to provider endpoint")

    with patch("httpx.AsyncClient", return_value=FailingAsyncClient()):
        advice = await LlmAdvisor.request_advisory(
            plan=sample_plan_record,
            benchmark=None,
        )

    assert advice["is_grounded"] is False
    assert "Advisory Unavailable" in advice["advisory_label"]
    assert "test-mock-key" not in str(advice)
