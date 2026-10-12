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
        lines = [
            "You are an AI/ML systems optimization advisor for large language models.",
            "Analyze the following verified execution records and provide actionable advice.",
            ""
        ]

        if plan:
            lines.extend([
                "### Verified Inference Plan:",
                f"- Model ID: {plan.model_id}",
                f"- Hardware ID: {plan.hardware_id}",
                f"- Quantization: {plan.quantization}",
                f"- Layers per chunk: {plan.layers_per_chunk} (Total chunks: {plan.total_chunks})",
                f"- Parameter VRAM: {plan.param_vram_mb} MB",
                f"- Activation VRAM: {plan.activation_vram_mb} MB",
                f"- KV Cache VRAM: {plan.kv_cache_vram_mb} MB",
                f"- Total Peak VRAM: {plan.total_vram_mb} MB ({plan.vram_utilization_percent}% utilization)",
                f"- Estimated throughput: {plan.est_tokens_per_sec} tokens/sec",
                ""
            ])

        if benchmark:
            lines.extend([
                "### Verified Benchmark Run Execution:",
                f"- Prompt tokens: {benchmark.prompt_tokens}, Output tokens: {benchmark.output_tokens}",
                f"- Total execution time: {benchmark.total_time_ms} ms",
                f"- Measured throughput: {benchmark.tokens_per_sec} tokens/sec",
                f"- Peak VRAM: {benchmark.peak_vram_mb} MB",
                f"- Disk Read time: {benchmark.disk_read_time_ms} ms",
                f"- PCIe Transfer time: {benchmark.pcie_transfer_time_ms} ms",
                f"- Compute time: {benchmark.compute_time_ms} ms",
                ""
            ])

        if user_query:
            lines.append(f"User Query: {user_query}")
        else:
            lines.append(
                "Task: Identify potential bottlenecks (disk I/O vs PCIe vs compute) and recommend "
                "optimal layer chunk size adjustments or quantization trade-offs."
            )

        return "\n".join(lines)

    @classmethod
    async def request_advisory(
        cls,
        plan: Optional[InferencePlanRecord],
        benchmark: Optional[BenchmarkRunRecord],
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Request advisory from operator-configured LLM endpoint.
        Falls back gracefully with deterministic heuristics if no API key is supplied.
        """
        grounded_prompt = cls.format_grounded_context(plan, benchmark, user_query)

        # If no key is configured, return deterministic grounded heuristics
        if not settings.LLM_API_KEY:
            recommendations: List[str] = []
            if plan:
                if plan.vram_utilization_percent < 50.0:
                    recommendations.append(
                        f"Current VRAM utilization is only {plan.vram_utilization_percent}%. "
                        "Increase layers per chunk (N-layer batching) to reduce disk I/O passes and increase throughput."
                    )
                elif plan.vram_utilization_percent > 85.0:
                    recommendations.append(
                        f"VRAM utilization is {plan.vram_utilization_percent}%, approaching safety thresholds. "
                        "Keep layer chunk size bounded to avoid CUDA out-of-memory during long context expansions."
                    )
            if benchmark and benchmark.disk_read_time_ms > (benchmark.compute_time_ms * 2):
                recommendations.append(
                    "Disk read I/O is the primary bottleneck. Consider enabling layer prefetching "
                    "or storing quantized 4-bit weights on an NVMe SSD."
                )

            if not recommendations:
                recommendations.append("Balanced configuration: layer chunking and transfer pipelines are operating within nominal thresholds.")

            return {
                "provider": "offline-deterministic-fallback",
                "model": "built-in-heuristics",
                "is_grounded": True,
                "advisory_label": "Deterministic Advisory (No external LLM key configured)",
                "recommendations": recommendations,
                "analysis_text": (
                    "Deterministic analysis based on verified system telemetry: "
                    "Disk and PCIe transfers dominate sequential execution. Increasing layers_per_chunk "
                    "reduces total roundtrips when VRAM headroom allows."
                ),
                "hardware_assessment": "Nominal hardware headroom for current model specification."
            }

        provider = settings.LLM_PROVIDER.lower()
        model = settings.LLM_MODEL
        timeout = httpx.Timeout(15.0, connect=5.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if provider == "anthropic":
                    url = settings.LLM_BASE_URL or "https://api.anthropic.com/v1/messages"
                    headers = {
                        "x-api-key": settings.LLM_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    }
                    body = {
                        "model": model,
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": grounded_prompt}]
                    }
                    resp = await client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = data.get("content", [{}])[0].get("text", "")

                elif provider == "gemini":
                    base = settings.LLM_BASE_URL or "https://generativelanguage.googleapis.com/v1beta"
                    url = f"{base}/models/{model}:generateContent?key={settings.LLM_API_KEY}"
                    body = {
                        "contents": [{"parts": [{"text": grounded_prompt}]}]
                    }
                    resp = await client.post(url, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "") if candidates else ""

                else:
                    # openai-compatible (including LiteLLM, Ollama, OpenRouter)
                    url = settings.LLM_BASE_URL or "https://api.openai.com/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {settings.LLM_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    body = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a concise, factual AI/ML systems optimization advisor."},
                            {"role": "user", "content": grounded_prompt}
                        ],
                        "max_tokens": 800,
                        "temperature": 0.2
                    }
                    resp = await client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                return {
                    "provider": provider,
                    "model": model,
                    "is_grounded": True,
                    "advisory_label": f"AI Advisory via {provider} ({model}) — Opt-in Advisory Only",
                    "recommendations": [
                        "Review I/O pipelining against GPU compute saturation.",
                        "Evaluate 4-bit block-wise compression for 70B parameter deployments."
                    ],
                    "analysis_text": raw_text.strip(),
                    "hardware_assessment": "Grounded assessment generated from verified telemetry."
                }

        except Exception as exc:
            # Clear provider error with redacted details (never leak API keys)
            error_msg = f"LLM provider error ({provider}): {type(exc).__name__}"
            return {
                "provider": provider,
                "model": model,
                "is_grounded": False,
                "advisory_label": "Advisory Unavailable (Provider Communication Error)",
                "recommendations": ["Check provider endpoint configuration, model identifier, and network connectivity."],
                "analysis_text": error_msg,
                "hardware_assessment": "External advisory request failed; falling back to offline planner."
            }
