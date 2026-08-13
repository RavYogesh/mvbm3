from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import asdict
from typing import Any

from .graders import grade
from .providers import Provider
from .stats import mean, percentile
from .types import CaseResult, EvalCase, Generation, ModelSpec


class AuditTrail:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, agent: str, event: str, **payload: Any) -> None:
        self.events.append(
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "agent": agent,
                "event": event,
                "payload": payload,
            }
        )


class ExecutionAgent:
    name = "execution-agent"

    def __init__(self, provider: Provider, audit: AuditTrail):
        self.provider = provider
        self.audit = audit

    def run(self, case: EvalCase, temperature: float, seed: int) -> Generation:
        self.audit.emit(self.name, "dispatch", case_id=case.id)
        generation = self.provider.generate(case, temperature, seed)
        self.audit.emit(
            self.name,
            "complete" if not generation.error else "error",
            case_id=case.id,
            latency_s=generation.latency_s,
            error=generation.error,
        )
        return generation


class GradingAgent:
    name = "grading-agent"

    def __init__(self, audit: AuditTrail):
        self.audit = audit

    def run(
        self,
        run_id: str,
        spec: ModelSpec,
        case: EvalCase,
        repetition: int,
        generation: Generation,
    ) -> CaseResult:
        if generation.error:
            score, passed, details = 0.0, False, {"grader": case.grader["type"], "error": generation.error}
        else:
            score, passed, details = grade(case, generation.text)
        cost = None
        if spec.input_price_per_million is not None and spec.output_price_per_million is not None:
            # Reasoning tokens are billed as output tokens. Excluding them
            # understates the cost of whichever model thinks longer -- which is
            # the exact comparison this gate exists to make, so the omission
            # would bias the result toward the more verbose model.
            cost = (
                generation.input_tokens * spec.input_price_per_million
                + generation.billed_output_tokens * spec.output_price_per_million
            ) / 1_000_000
        result = CaseResult(
            run_id=run_id,
            model=spec.name,
            case_id=case.id,
            task=case.task,
            risk_tier=case.risk_tier,
            repetition=repetition,
            output=generation.text,
            score=score,
            passed=passed,
            latency_s=generation.latency_s,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            cost_usd=cost,
            grade_details=details,
            error=generation.error,
            ttft_s=generation.ttft_s,
            reasoning_tokens=generation.reasoning_tokens,
            attempts=generation.attempts,
            infrastructure_error=generation.infrastructure_error,
        )
        self.audit.emit(self.name, "graded", case_id=case.id, model=spec.name, score=score, passed=passed)
        return result


class RiskAgent:
    name = "risk-agent"

    def __init__(self, audit: AuditTrail):
        self.audit = audit

    def inspect(self, result: CaseResult) -> None:
        # An endpoint failure is not a control failure. Raising one as the other
        # trains reviewers to discount the signal, which is worse than not
        # raising it at all.
        if result.infrastructure_error:
            self.audit.emit(
                self.name,
                "infrastructure_error",
                model=result.model,
                case_id=result.case_id,
                attempts=result.attempts,
            )
            return
        if result.risk_tier in {"high", "critical"} and not result.passed:
            self.audit.emit(
                self.name,
                "control_failure",
                model=result.model,
                case_id=result.case_id,
                risk_tier=result.risk_tier,
                failure=result.grade_details.get("failure"),
            )


class PerformanceAgent:
    name = "performance-agent"

    @staticmethod
    def summarize(
        results: list[CaseResult],
        wall_clock_s: float | None = None,
        concurrency: int = 1,
    ) -> dict[str, Any]:
        total_latency = sum(r.latency_s for r in results)
        output_tokens = sum(r.output_tokens for r in results)
        billed_tokens = sum(r.output_tokens + r.reasoning_tokens for r in results)
        reasoning_tokens = sum(r.reasoning_tokens for r in results)
        costs = [r.cost_usd for r in results if r.cost_usd is not None]
        successes = sum(1 for r in results if r.passed)
        infra_errors = sum(1 for r in results if r.infrastructure_error)
        ttfts = [r.ttft_s for r in results if r.ttft_s > 0]
        safety_rows = [
            r
            for r in results
            if r.task in {"pii_safety", "prompt_injection", "policy_compliance"}
            and r.risk_tier == "critical"
        ]
        by_task: dict[str, dict[str, float]] = {}
        for task in sorted({r.task for r in results}):
            rows = [r for r in results if r.task == task]
            by_task[task] = {
                "mean_score": mean(r.score for r in rows),
                "pass_rate": mean(float(r.passed) for r in rows),
                "cases": len(rows),
            }
        # Two different quantities, previously conflated.
        #
        #   per-stream decode rate = tokens / SUM of per-request latencies.
        #     What a single user experiences. The old metric.
        #   system throughput      = tokens / WALL-CLOCK elapsed.
        #     What the fleet delivers. With N workers in flight these differ by
        #     roughly N, and the vendor's "+39.5% throughput" is a claim about
        #     the second one, measured at a stated concurrency.
        per_stream = output_tokens / total_latency if total_latency else 0.0
        system = (billed_tokens / wall_clock_s) if wall_clock_s else None
        return {
            "mean_score": mean(r.score for r in results),
            "pass_rate": mean(float(r.passed) for r in results),
            "p50_latency_s": percentile((r.latency_s for r in results), 0.50),
            "p95_latency_s": percentile((r.latency_s for r in results), 0.95),
            "p99_latency_s": percentile((r.latency_s for r in results), 0.99),
            "p50_ttft_s": percentile(ttfts, 0.50) if ttfts else None,
            "p95_ttft_s": percentile(ttfts, 0.95) if ttfts else None,
            "ttft_measured": bool(ttfts),
            "decode_tokens_s_per_stream": per_stream,
            "throughput_output_tokens_s": per_stream,  # retained for compatibility
            "system_throughput_tokens_s": system,
            "measured_at_concurrency": concurrency,
            "wall_clock_s": wall_clock_s,
            "reasoning_tokens": reasoning_tokens,
            "reasoning_token_share": (reasoning_tokens / billed_tokens) if billed_tokens else 0.0,
            "error_rate": mean(float(r.error is not None) for r in results),
            "infrastructure_error_rate": mean(float(r.infrastructure_error) for r in results),
            "model_error_rate": mean(
                float(r.error is not None and not r.infrastructure_error) for r in results
            ),
            "infrastructure_errors": infra_errors,
            "total_cost_usd": sum(costs) if costs else None,
            "cost_per_success_usd": (sum(costs) / successes) if costs and successes else None,
            # None, not 0.0, when there is nothing to measure. `mean([])` returns
            # 0.0, which the gate reads as a total safety failure -- so an
            # absent family used to look identical to a catastrophic one.
            "critical_safety_pass_rate": (
                mean(float(r.passed) for r in safety_rows) if safety_rows else None
            ),
            "critical_safety_cases": len(safety_rows),
            "over_refusal_count": sum(
                1 for r in results if r.grade_details.get("failure") == "over_refusal"
            ),
            "under_refusal_count": sum(
                1 for r in results if r.grade_details.get("failure") == "under_refusal"
            ),
            "by_task": by_task,
        }


def new_run_id() -> str:
    return "run_" + uuid.uuid4().hex[:12]

