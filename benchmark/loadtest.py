"""Concurrency sweep for server-side throughput and tail latency.

WHY A SEPARATE PHASE
--------------------
The correctness suite runs at one concurrency and mixes prompt shapes, which is
right for grading and wrong for performance. Three distinct quantities get
conflated otherwise:

  per-stream decode rate   tokens / SUM of per-request latencies.
                           What one user experiences. Roughly flat in load.
  system throughput        tokens / WALL-CLOCK elapsed.
                           What the fleet delivers. Rises with concurrency
                           until the server saturates. This is the quantity the
                           vendor's "+39.5% throughput" claim is about.
  tail latency             p95/p99, which is what an SLO is written against.
                           A median improvement with a p95 regression is a
                           capacity problem wearing a success costume.

A single number reported without its concurrency is not a throughput
measurement, so every row here carries the concurrency it was taken at.

The sweep also finds the knee: the concurrency beyond which throughput stops
rising and latency starts climbing. Two models can have identical single-stream
speed and very different knees, and the knee is what sets how much hardware the
switch actually saves.
"""
from __future__ import annotations

import concurrent.futures
import statistics
import time
from typing import Any

from .providers import build_provider
from .stats import percentile
from .types import EvalCase, Generation, ModelSpec

DEFAULT_LEVELS = [1, 4, 16, 64]


def _synthetic_case(index: int, prompt_tokens: int, max_tokens: int) -> EvalCase:
    """Fixed-shape load case.

    Deliberately uniform: mixing prompt lengths inside a sweep confounds the
    prefill component with the decode component, and TTFT is dominated by
    prefill. Prompt shape is held constant so the comparison across models is
    about the model, not about which sampler drew the longer prompt.
    """
    filler = ("Synthetic servicing narrative for load generation. " * 40)[: prompt_tokens * 4]
    return EvalCase(
        id=f"LOAD-{index:05d}",
        task="loadtest",
        risk_tier="low",
        prompt=f"Summarise the following in at most {max_tokens // 4} words.",
        context=filler,
        grader={"type": "token_f1", "reference": "synthetic"},
    )


def _run_level(
    spec: ModelSpec,
    concurrency: int,
    requests: int,
    prompt_tokens: int,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    provider = build_provider(spec)
    cases = [_synthetic_case(i, prompt_tokens, max_tokens) for i in range(requests)]
    generations: list[Generation] = []

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(provider.generate, case, temperature, seed) for case in cases]
        for future in concurrent.futures.as_completed(futures):
            generations.append(future.result())
    wall_clock = time.perf_counter() - started

    ok = [g for g in generations if not g.error]
    latencies = [g.latency_s for g in ok]
    ttfts = [g.ttft_s for g in ok if g.ttft_s > 0]
    billed = sum(g.billed_output_tokens for g in ok)
    total_latency = sum(latencies)

    return {
        "concurrency": concurrency,
        "requests": requests,
        "completed": len(ok),
        "errors": len(generations) - len(ok),
        "wall_clock_s": round(wall_clock, 4),
        "system_throughput_tokens_s": round(billed / wall_clock, 2) if wall_clock else 0.0,
        "requests_per_s": round(len(ok) / wall_clock, 3) if wall_clock else 0.0,
        "decode_tokens_s_per_stream": round(billed / total_latency, 2) if total_latency else 0.0,
        "p50_latency_s": round(percentile(latencies, 0.50), 4),
        "p95_latency_s": round(percentile(latencies, 0.95), 4),
        "p99_latency_s": round(percentile(latencies, 0.99), 4),
        "mean_latency_s": round(statistics.fmean(latencies), 4) if latencies else 0.0,
        "p50_ttft_s": round(percentile(ttfts, 0.50), 4) if ttfts else None,
        "p95_ttft_s": round(percentile(ttfts, 0.95), 4) if ttfts else None,
        "ttft_measured": bool(ttfts),
        "reasoning_tokens": sum(g.reasoning_tokens for g in ok),
        "billed_output_tokens": billed,
    }


def sweep(
    config: dict[str, Any],
    levels: list[int] | None = None,
    requests_per_level: int = 32,
    prompt_tokens: int = 512,
    max_tokens: int = 256,
) -> dict[str, Any]:
    levels = levels or DEFAULT_LEVELS
    temperature = float(config.get("temperature", 0.0))
    seed = int(config.get("seed", 42))
    specs = [ModelSpec(**raw) for raw in config["models"]]
    for spec in specs:
        # A mock that returns instantly cannot queue, so wall-clock would report
        # thread-pool overhead instead of server saturation. Offline sweeps make
        # the mock actually wait so the shape of the curve is real, even though
        # the absolute numbers are still synthetic.
        if spec.base_url.startswith("mock://"):
            spec.metadata.setdefault("mock_sleep", True)

    results: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        rows = []
        for level in levels:
            requests = max(requests_per_level, level)
            rows.append(
                _run_level(spec, level, requests, prompt_tokens, max_tokens, temperature, seed)
            )
        results[spec.name] = rows

    comparisons = []
    for spec in specs:
        if spec.role != "candidate" or not spec.baseline or spec.baseline not in results:
            continue
        candidate_rows = {r["concurrency"]: r for r in results[spec.name]}
        baseline_rows = {r["concurrency"]: r for r in results[spec.baseline]}
        per_level = []
        for level in levels:
            c, b = candidate_rows.get(level), baseline_rows.get(level)
            if not c or not b:
                continue
            per_level.append(
                {
                    "concurrency": level,
                    "system_throughput_improvement": _gain(
                        b["system_throughput_tokens_s"], c["system_throughput_tokens_s"]
                    ),
                    "p95_latency_improvement": _reduction(b["p95_latency_s"], c["p95_latency_s"]),
                    "p99_latency_improvement": _reduction(b["p99_latency_s"], c["p99_latency_s"]),
                    "p95_ttft_improvement": _reduction(b["p95_ttft_s"], c["p95_ttft_s"]),
                }
            )
        comparisons.append(
            {
                "candidate": spec.name,
                "baseline": spec.baseline,
                "per_level": per_level,
                "candidate_knee": _knee(results[spec.name]),
                "baseline_knee": _knee(results[spec.baseline]),
            }
        )

    return {
        "schema_version": "1.0",
        "synthetic_demo": all(spec.base_url.startswith("mock://") for spec in specs),
        "levels": levels,
        "requests_per_level": requests_per_level,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "by_model": results,
        "comparisons": comparisons,
        "notes": (
            "system_throughput is measured against wall-clock elapsed and is the quantity the "
            "vendor throughput claim refers to. decode_tokens_s_per_stream is reported for "
            "context and is never gated -- it does not change materially with load, so it "
            "cannot answer a capacity question."
        ),
    }


def _gain(baseline: float | None, candidate: float | None) -> float | None:
    if not baseline or candidate is None:
        return None
    return round(candidate / baseline - 1, 4)


def _reduction(baseline: float | None, candidate: float | None) -> float | None:
    if not baseline or candidate is None:
        return None
    return round(1 - candidate / baseline, 4)


def _knee(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Concurrency at peak system throughput.

    Past this point the server is saturated: added load buys queueing delay
    rather than work. Two models with identical single-stream speed can have
    very different knees, and the knee is what decides how much hardware the
    switch actually saves.
    """
    if not rows:
        return None
    best = max(rows, key=lambda r: r["system_throughput_tokens_s"])
    return {
        "concurrency": best["concurrency"],
        "system_throughput_tokens_s": best["system_throughput_tokens_s"],
        "p95_latency_s": best["p95_latency_s"],
    }
