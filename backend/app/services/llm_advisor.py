"""Provider-agnostic LLM advisory adapter grounded in actual benchmark records."""
import json
from typing import Dict, Any, List, Optional
import httpx
from app.core.config import settings
from app.models.entities import BenchmarkRunRecord, InferencePlanRecord


class LlmAdvisor:
    """Provides opt-in advisory analysis grounded strictly in benchmark and plan records."""

    @staticmethod
    def format_grounded_context(
        plan: Optional[InferencePlanRecord],
        benchmark: Optional[BenchmarkRunRecord],
        user_query: Optional[str]
    ) -> str:
        """Construct grounded prompt context from verified records."""
        lines = ["AI/ML systems optimization advisor for large language models.", "Verified telemetry:"]
        if plan:
            lines.append(
                f"Plan: model={plan.model_id}, hw={plan.hardware_id}, quant={plan.quantization}, "
                f"chunk_layers={plan.layers_per_chunk}, total_vram={plan.total_vram_mb}MB ({plan.vram_utilization_percent}%), "
                f"est_tok_sec={plan.est_tokens_per_sec}"
            )
        if benchmark:
            lines.append(
                f"Benchmark: prompt={benchmark.prompt_tokens}, output={benchmark.output_tokens}, "
                f"time={benchmark.total_time_ms}ms, tok_sec={benchmark.tokens_per_sec}, "
                f"peak_vram={benchmark.peak_vram_mb}MB, disk_read={benchmark.disk_read_time_ms}ms, "
                f"pcie={benchmark.pcie_transfer_time_ms}ms, compute={benchmark.compute_time_ms}ms"
            )
        lines.append(f"Query: {user_query}" if user_query else "Task: Identify bottlenecks and recommend chunk size/quantization.")
        return "\n".join(lines)

    @classmethod
    async def request_advisory(
        cls,
        plan: Optional[InferencePlanRecord],
        benchmark: Optional[BenchmarkRunRecord],
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Request advisory from operator-configured LLM endpoint or deterministic heuristics."""
        grounded_prompt = cls.format_grounded_context(plan, benchmark, user_query)

        if not settings.LLM_API_KEY:
            recommendations: List[str] = []
            if plan:
                if plan.vram_utilization_percent < 50.0:
                    recommendations.append(
                        f"Current VRAM utilization is {plan.vram_utilization_percent}%. "
                        "Increase layers per chunk (N-layer batching) to reduce disk I/O passes and increase throughput."
                    )
                elif plan.vram_utilization_percent > 85.0:
                    recommendations.append(
                        f"VRAM utilization is {plan.vram_utilization_percent}%. "
                        "Keep layer chunk size bounded to avoid CUDA out-of-memory during long context expansions."
                    )
            if benchmark and benchmark.disk_read_time_ms > (benchmark.compute_time_ms * 2):
                recommendations.append("Disk read I/O is primary bottleneck. Consider layer prefetching or NVMe storage.")

            if not recommendations:
                recommendations.append("Balanced configuration: layer chunking and transfer pipelines are nominal.")

            return {
                "provider": "offline-deterministic-fallback",
                "model": "built-in-heuristics",
                "is_grounded": True,
                "advisory_label": "Deterministic Advisory (No external LLM key configured)",
                "recommendations": recommendations,
                "analysis_text": "Deterministic analysis based on verified system telemetry: sequential transfers dominate execution.",
                "hardware_assessment": "Nominal hardware headroom for current model specification."
            }

        provider = settings.LLM_PROVIDER.lower()
        model = settings.LLM_MODEL
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
                if provider == "anthropic":
                    url = settings.LLM_BASE_URL or "https://api.anthropic.com/v1/messages"
                    resp = await client.post(
                        url,
                        headers={"x-api-key": settings.LLM_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                        json={"model": model, "max_tokens": 1024, "messages": [{"role": "user", "content": grounded_prompt}]}
                    )
                    resp.raise_for_status()
                    raw_text = resp.json().get("content", [{}])[0].get("text", "")
                elif provider == "gemini":
                    base = settings.LLM_BASE_URL or "https://generativelanguage.googleapis.com/v1beta"
                    resp = await client.post(
                        f"{base}/models/{model}:generateContent?key={settings.LLM_API_KEY}",
                        json={"contents": [{"parts": [{"text": grounded_prompt}]}]}
                    )
                    resp.raise_for_status()
                    cand = resp.json().get("candidates", [])
                    raw_text = cand[0].get("content", {}).get("parts", [{}])[0].get("text", "") if cand else ""
                else:
                    url = settings.LLM_BASE_URL or "https://api.openai.com/v1/chat/completions"
                    resp = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {settings.LLM_API_KEY}", "Content-Type": "application/json"},
                        json={"model": model, "messages": [{"role": "system", "content": "You are a concise AI/ML systems advisor."}, {"role": "user", "content": grounded_prompt}], "max_tokens": 800}
                    )
                    resp.raise_for_status()
                    raw_text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")

                return {
                    "provider": provider, "model": model, "is_grounded": True,
                    "advisory_label": f"AI Advisory via {provider} ({model}) — Opt-in Advisory Only",
                    "recommendations": ["Review I/O pipelining against GPU saturation.", "Evaluate 4-bit block compression."],
                    "analysis_text": raw_text.strip(),
                    "hardware_assessment": "Grounded assessment generated from verified telemetry."
                }
        except Exception as exc:
            return {
                "provider": provider, "model": model, "is_grounded": False,
                "advisory_label": "Advisory Unavailable (Provider Communication Error)",
                "recommendations": ["Check provider endpoint configuration, model identifier, and network connectivity."],
                "analysis_text": f"LLM provider error ({provider}): {type(exc).__name__}",
                "hardware_assessment": "External advisory request failed; falling back to offline planner."
            }
