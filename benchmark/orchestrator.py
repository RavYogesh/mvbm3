from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

from .agents import AuditTrail, ExecutionAgent, GradingAgent, PerformanceAgent, RiskAgent, new_run_id
from .dataset import dataset_profile, load_dataset
from .providers import build_provider
from .stats import Verdict, mean, non_inferiority
from .types import CaseResult, EvalCase, ModelSpec


def _model_spec(raw: dict[str, Any]) -> ModelSpec:
    return ModelSpec(**raw)


class EvalOrchestrator:
    """Control-tower coordinator for auditable, concurrent evaluation workers."""

    def __init__(self, config: dict[str, Any], gates: dict[str, Any] | None = None):
        self.config = config
        self.gates = gates or {}
        self.audit = AuditTrail()
        self.run_id = new_run_id()

    def _one(
        self,
        spec: ModelSpec,
        case: EvalCase,
        repetition: int,
        execution: ExecutionAgent,
        grading: GradingAgent,
        risk: RiskAgent,
    ) -> CaseResult:
        generation = execution.run(
            case,
            float(self.config.get("temperature", 0.0)),
            int(self.config.get("seed", 42)) + repetition,
        )
        result = grading.run(self.run_id, spec, case, repetition, generation)
        risk.inspect(result)
        return result

    def run(self, cases: list[EvalCase]) -> dict[str, Any]:
        specs = [_model_spec(item) for item in self.config["models"]]
        repetitions = int(self.config.get("repetitions", 1))
        max_workers = int(self.config.get("max_workers", 4))
        all_results: list[CaseResult] = []
        wall_clock: dict[str, float] = {}
        grading = GradingAgent(self.audit)
        risk = RiskAgent(self.audit)
        self.audit.emit("orchestrator", "run_started", run_id=self.run_id, models=len(specs), cases=len(cases))

        for spec in specs:
            provider = build_provider(spec)
            execution = ExecutionAgent(provider, self.audit)
            # Wall-clock elapsed per model is what makes system throughput
            # measurable at all; the sum of per-request latencies is a different
            # quantity and cannot substitute for it.
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(self._one, spec, case, repetition, execution, grading, risk)
                    for repetition in range(repetitions)
                    for case in cases
                ]
                for future in concurrent.futures.as_completed(futures):
                    all_results.append(future.result())
            wall_clock[spec.name] = time.perf_counter() - started

        by_model: dict[str, list[CaseResult]] = {
            spec.name: [r for r in all_results if r.model == spec.name] for spec in specs
        }
        summaries = {
            name: PerformanceAgent.summarize(rows, wall_clock.get(name), max_workers)
            for name, rows in by_model.items()
        }
        design = self._design_checks(cases, repetitions)
        comparisons = self._comparisons(specs, by_model, summaries, design)
        self.audit.emit("orchestrator", "run_completed", run_id=self.run_id, results=len(all_results))
        return {
            "schema_version": "2.0",
            "run_id": self.run_id,
            "run_name": self.config.get("run_name", self.run_id),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "synthetic_demo": all(spec.base_url.startswith("mock://") for spec in specs),
            "config": self._safe_config(),
            "dataset": dataset_profile(cases),
            "design_checks": design,
            "summaries": summaries,
            "comparisons": comparisons,
            "results": [r.to_dict() for r in sorted(all_results, key=lambda x: (x.model, x.case_id, x.repetition))],
            "audit_trail": self.audit.events,
        }

    def _safe_config(self) -> dict[str, Any]:
        safe = json.loads(json.dumps(self.config))
        for model in safe.get("models", []):
            model.pop("headers", None)
        return safe

    # -- experimental design -------------------------------------------------
    def _design_checks(self, cases: list[EvalCase], repetitions: int) -> dict[str, Any]:
        """Validate the experiment before trusting anything it produced.

        These floors were declared in the gates config from the start but were
        never read by any code, so the config promised a sample-size discipline
        the harness did not enforce. They are enforced here, and a violation is
        surfaced as a blocking design finding rather than a footnote.
        """
        findings: list[dict[str, Any]] = []
        if not self.gates.get("enforce_sample_size", False):
            return {"status": "NOT_ENFORCED", "findings": findings}

        minimum = int(self.gates.get("minimum_cases_per_task", 0))
        profile = dataset_profile(cases)
        for task, count in sorted(profile["by_task"].items()):
            if count < minimum:
                findings.append(
                    {
                        "check": f"minimum_cases_per_task:{task}",
                        "value": count,
                        "threshold": minimum,
                        "status": "FAIL",
                        "detail": f"{task} has {count} cases; per-task gating on {count} samples is noise",
                    }
                )

        min_reps = int(self.gates.get("minimum_repetitions", 1))
        if repetitions < min_reps:
            findings.append(
                {
                    "check": "minimum_repetitions",
                    "value": repetitions,
                    "threshold": min_reps,
                    "status": "FAIL",
                    "detail": "not enough repetitions to estimate sampling variance",
                }
            )

        # Repetitions at temperature 0 are near-duplicates on most serving
        # stacks: they triple the bill and UNDERSTATE variance rather than
        # measuring it. Either sample at temperature or stop paying for reps.
        temperature = float(self.config.get("temperature", 0.0))
        min_temp = float(self.gates.get("minimum_temperature_for_repetitions", 0.0))
        if repetitions > 1 and temperature < min_temp:
            findings.append(
                {
                    "check": "repetitions_measure_variance",
                    "value": temperature,
                    "threshold": min_temp,
                    "status": "WARN",
                    "detail": (
                        f"{repetitions} repetitions at temperature {temperature} are near-deterministic; "
                        "they multiply cost without estimating sampling variance"
                    ),
                }
            )

        blocking = [f for f in findings if f["status"] == "FAIL"]
        return {
            "status": "FAIL" if blocking else ("WARN" if findings else "PASS"),
            "findings": findings,
        }

    # -- comparisons ---------------------------------------------------------
    def _paired_scores(
        self, cand_rows: list[CaseResult], base_rows: list[CaseResult], predicate=None
    ) -> tuple[list[float], list[float]]:
        """Align candidate and baseline on (case_id, repetition).

        Infrastructure failures are dropped from the pair rather than scored 0.
        A gateway timeout is not evidence about model quality, and leaving it in
        moves the quality delta by whichever side happened to be unlucky.
        """
        base_map = {(r.case_id, r.repetition): r for r in base_rows}
        candidate_scores: list[float] = []
        baseline_scores: list[float] = []
        for row in sorted(cand_rows, key=lambda r: (r.case_id, r.repetition)):
            match = base_map.get((row.case_id, row.repetition))
            if match is None:
                continue
            if row.infrastructure_error or match.infrastructure_error:
                continue
            if predicate and not predicate(row):
                continue
            candidate_scores.append(row.score)
            baseline_scores.append(match.score)
        return candidate_scores, baseline_scores

    def _comparisons(
        self,
        specs: list[ModelSpec],
        by_model: dict[str, list[CaseResult]],
        summaries: dict[str, dict[str, Any]],
        design: dict[str, Any],
    ) -> list[dict[str, Any]]:
        comparisons: list[dict[str, Any]] = []
        alpha = float(self.gates.get("alpha", 0.05))
        power_target = float(self.gates.get("power_target", 0.80))
        critical_tasks = set(self.gates.get("critical_tasks", []))

        for candidate in [s for s in specs if s.role == "candidate" and s.baseline]:
            if candidate.baseline not in by_model:
                continue
            cand_rows = by_model[candidate.name]
            base_rows = by_model[candidate.baseline]
            candidate_summary = summaries[candidate.name]
            baseline_summary = summaries[candidate.baseline]

            preservation: list[dict[str, Any]] = []

            def test(name: str, margin: float, predicate=None) -> dict[str, Any]:
                c, b = self._paired_scores(cand_rows, base_rows, predicate)
                result = non_inferiority(
                    name, c, b, margin, alpha=alpha, power_target=power_target
                )
                payload = result.to_dict()
                preservation.append(payload)
                return payload

            test("aggregate", float(self.gates.get("aggregate_margin", 0.03)))
            test(
                "high_risk",
                float(self.gates.get("high_risk_margin", 0.02)),
                lambda r: r.risk_tier in {"high", "critical"},
            )
            for task in sorted({r.task for r in cand_rows}):
                margin = float(
                    self.gates.get("critical_task_margin", 0.02)
                    if task in critical_tasks
                    else self.gates.get("noncritical_task_margin", 0.05)
                )
                test(f"task:{task}", margin, lambda r, t=task: r.task == t)

            metrics = {
                "mean_score_delta": candidate_summary["mean_score"] - baseline_summary["mean_score"],
                "preservation": preservation,
                "p95_latency_improvement": _improvement(
                    baseline_summary["p95_latency_s"], candidate_summary["p95_latency_s"]
                ),
                "p99_latency_improvement": _improvement(
                    baseline_summary["p99_latency_s"], candidate_summary["p99_latency_s"]
                ),
                "p95_ttft_improvement": _improvement(
                    baseline_summary["p95_ttft_s"], candidate_summary["p95_ttft_s"]
                ),
                "system_throughput_improvement": _ratio_gain(
                    baseline_summary["system_throughput_tokens_s"],
                    candidate_summary["system_throughput_tokens_s"],
                ),
                "decode_rate_improvement_per_stream": _ratio_gain(
                    baseline_summary["decode_tokens_s_per_stream"],
                    candidate_summary["decode_tokens_s_per_stream"],
                ),
                "cost_per_success_improvement": _improvement(
                    baseline_summary["cost_per_success_usd"], candidate_summary["cost_per_success_usd"]
                ),
                "reasoning_token_share": candidate_summary["reasoning_token_share"],
                "baseline_reasoning_token_share": baseline_summary["reasoning_token_share"],
                "critical_safety_pass_rate": candidate_summary["critical_safety_pass_rate"],
                "over_refusals": candidate_summary["over_refusal_count"],
                "under_refusals": candidate_summary["under_refusal_count"],
                "model_error_rate": candidate_summary["model_error_rate"],
                "infrastructure_error_rate": candidate_summary["infrastructure_error_rate"],
            }
            comparisons.append(
                {
                    "candidate": candidate.name,
                    "baseline": candidate.baseline,
                    "metrics": metrics,
                    "gates": self._evaluate_gates(metrics, candidate_summary, design),
                }
            )
        return comparisons

    # -- gates ---------------------------------------------------------------
    def _evaluate_gates(
        self, metrics: dict[str, Any], summary: dict[str, Any], design: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.gates:
            return {"overall": "NOT_CONFIGURED", "checks": []}
        checks: list[dict[str, Any]] = []

        def threshold_check(name: str, value: float | None, limit: float, direction: str = "min") -> None:
            if value is None:
                status = "NOT_MEASURED"
            else:
                status = "PASS" if (value >= limit if direction == "min" else value <= limit) else "FAIL"
            checks.append(
                {"name": name, "value": value, "threshold": limit, "direction": direction, "status": status}
            )

        # Quality: non-inferiority, one entry per family.
        for item in metrics["preservation"]:
            checks.append(
                {
                    "name": f"preservation:{item['name']}",
                    "value": item["delta"],
                    "lower_bound": item["lower_bound"],
                    "threshold": -item["margin"],
                    "direction": "non_inferiority",
                    "n": item["n"],
                    "observed_power": item["observed_power"],
                    "required_n": item["required_n"],
                    "status": item["status"],
                    "notes": item["notes"],
                }
            )

        # Safety, two-sided.
        threshold_check(
            "critical_safety_pass_rate",
            metrics["critical_safety_pass_rate"],
            float(self.gates.get("critical_safety_pass_rate_min", 1.0)),
        )
        threshold_check(
            "over_refusals", metrics["over_refusals"], float(self.gates.get("max_over_refusals", 0)), "max"
        )

        # Efficiency.
        threshold_check(
            "p95_latency_improvement",
            metrics["p95_latency_improvement"],
            float(self.gates.get("p95_latency_improvement_min", 0.0)),
        )
        threshold_check(
            "p99_latency_improvement",
            metrics["p99_latency_improvement"],
            float(self.gates.get("p99_latency_improvement_min", 0.0)),
        )
        threshold_check(
            "p95_ttft_improvement",
            metrics["p95_ttft_improvement"],
            float(self.gates.get("p95_ttft_improvement_min", 0.0)),
        )
        threshold_check(
            "system_throughput_improvement",
            metrics["system_throughput_improvement"],
            float(self.gates.get("system_throughput_improvement_min", 0.0)),
        )
        threshold_check(
            "cost_per_success_improvement",
            metrics["cost_per_success_improvement"],
            float(self.gates.get("cost_per_success_improvement_min", 0.0)),
        )

        # Errors, split by cause.
        threshold_check(
            "model_error_rate", metrics["model_error_rate"],
            float(self.gates.get("model_error_rate_max", 1.0)), "max",
        )
        threshold_check(
            "infrastructure_error_rate", metrics["infrastructure_error_rate"],
            float(self.gates.get("infrastructure_error_rate_max", 1.0)), "max",
        )

        for finding in design.get("findings", []):
            checks.append(
                {
                    "name": f"design:{finding['check']}",
                    "value": finding["value"],
                    "threshold": finding["threshold"],
                    "direction": "min",
                    "status": finding["status"],
                    "notes": [finding["detail"]],
                }
            )

        statuses = {item["status"] for item in checks}
        if "FAIL" in statuses:
            overall = "BLOCK"
        elif Verdict.INCONCLUSIVE.value in statuses or "NOT_MEASURED" in statuses:
            # Deliberately distinct from BLOCK. "We could not tell" and "it is
            # worse" call for different actions -- more samples versus stop --
            # and collapsing them into one verdict is how a team ends up
            # quietly loosening a margin to make an underpowered run pass.
            overall = "INCONCLUSIVE"
        elif not checks:
            overall = "NOT_CONFIGURED"
        else:
            overall = "PASS"
        return {"overall": overall, "checks": checks}


def _improvement(baseline: float | None, candidate: float | None) -> float | None:
    """Fractional reduction, for metrics where lower is better."""
    if baseline is None or candidate is None or not baseline:
        return None
    return 1 - (candidate / baseline)


def _ratio_gain(baseline: float | None, candidate: float | None) -> float | None:
    """Fractional gain, for metrics where higher is better."""
    if baseline is None or candidate is None or not baseline:
        return None
    return candidate / baseline - 1


def run_from_files(
    config_path: str | Path, dataset_path: str | Path, gates_path: str | Path | None = None
) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    gates = json.loads(Path(gates_path).read_text(encoding="utf-8")) if gates_path else None
    cases = load_dataset(dataset_path)
    return EvalOrchestrator(config, gates).run(cases)
