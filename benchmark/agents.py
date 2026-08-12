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
            cost = (
                generation.input_tokens * spec.input_price_per_million
                + generation.output_tokens * spec.output_price_per_million
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
        )
        self.audit.emit(self.name, "graded", case_id=case.id, model=spec.name, score=score, passed=passed)
        return result


class RiskAgent:
    name = "risk-agent"

    def __init__(self, audit: AuditTrail):
        self.audit = audit

    def inspect(self, result: CaseResult) -> None:
        if result.risk_tier in {"high", "critical"} and not result.passed:
            self.audit.emit(
                self.name,
                "control_failure",
                model=result.model,
                case_id=result.case_id,
                risk_tier=result.risk_tier,
            )


class PerformanceAgent:
    name = "performance-agent"

    @staticmethod
    def summarize(results: list[CaseResult]) -> dict[str, Any]:
        total_latency = sum(r.latency_s for r in results)
        output_tokens = sum(r.output_tokens for r in results)
        costs = [r.cost_usd for r in results if r.cost_usd is not None]
        successes = sum(1 for r in results if r.passed)
        by_task: dict[str, dict[str, float]] = {}
        for task in sorted({r.task for r in results}):
            rows = [r for r in results if r.task == task]
            by_task[task] = {
                "mean_score": mean(r.score for r in rows),
                "pass_rate": mean(float(r.passed) for r in rows),
                "cases": len(rows),
            }
        return {
            "mean_score": mean(r.score for r in results),
            "pass_rate": mean(float(r.passed) for r in results),
            "p50_latency_s": percentile((r.latency_s for r in results), 0.50),
            "p95_latency_s": percentile((r.latency_s for r in results), 0.95),
            "throughput_output_tokens_s": output_tokens / total_latency if total_latency else 0.0,
            "error_rate": mean(float(r.error is not None) for r in results),
            "total_cost_usd": sum(costs) if costs else None,
            "cost_per_success_usd": (sum(costs) / successes) if costs and successes else None,
            "critical_safety_pass_rate": mean(
                float(r.passed)
                for r in results
                if r.task in {"pii_safety", "prompt_injection", "policy_compliance"}
                and r.risk_tier == "critical"
            ),
            "by_task": by_task,
        }


def new_run_id() -> str:
    return "run_" + uuid.uuid4().hex[:12]

