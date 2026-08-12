from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .agents import AuditTrail, ExecutionAgent, GradingAgent, PerformanceAgent, RiskAgent, new_run_id
from .dataset import dataset_profile, load_dataset
from .providers import build_provider
from .stats import bootstrap_mean_ci, mean
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
        grading = GradingAgent(self.audit)
        risk = RiskAgent(self.audit)
        self.audit.emit("orchestrator", "run_started", run_id=self.run_id, models=len(specs), cases=len(cases))

        for spec in specs:
            provider = build_provider(spec)
            execution = ExecutionAgent(provider, self.audit)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(self._one, spec, case, repetition, execution, grading, risk)
                    for repetition in range(repetitions)
                    for case in cases
                ]
                for future in concurrent.futures.as_completed(futures):
                    all_results.append(future.result())

        by_model: dict[str, list[CaseResult]] = {
            spec.name: [r for r in all_results if r.model == spec.name] for spec in specs
        }
        summaries = {name: PerformanceAgent.summarize(rows) for name, rows in by_model.items()}
        comparisons = self._comparisons(specs, by_model, summaries)
        self.audit.emit("orchestrator", "run_completed", run_id=self.run_id, results=len(all_results))
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "run_name": self.config.get("run_name", self.run_id),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "synthetic_demo": all(spec.base_url.startswith("mock://") for spec in specs),
            "config": self._safe_config(),
            "dataset": dataset_profile(cases),
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

    def _comparisons(
        self,
        specs: list[ModelSpec],
        by_model: dict[str, list[CaseResult]],
        summaries: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        comparisons: list[dict[str, Any]] = []
        for candidate in [s for s in specs if s.role == "candidate" and s.baseline]:
            if candidate.baseline not in by_model:
                continue
            cand_rows = by_model[candidate.name]
            base_rows = by_model[candidate.baseline]
            base_map = {(r.case_id, r.repetition): r for r in base_rows}
            paired = [(r, base_map[(r.case_id, r.repetition)]) for r in cand_rows if (r.case_id, r.repetition) in base_map]
            deltas = [c.score - b.score for c, b in paired]
            high_risk_deltas = [c.score - b.score for c, b in paired if c.risk_tier in {"high", "critical"}]
            ci_low, ci_high = bootstrap_mean_ci(high_risk_deltas)
            candidate_summary = summaries[candidate.name]
            baseline_summary = summaries[candidate.baseline]
            task_retention: dict[str, float | None] = {}
            for task in sorted(set(candidate_summary["by_task"]) | set(baseline_summary["by_task"])):
                c = candidate_summary["by_task"].get(task, {}).get("mean_score", 0.0)
                b = baseline_summary["by_task"].get(task, {}).get("mean_score", 0.0)
                task_retention[task] = (c / b) if b else None
            quality_retention = (
                candidate_summary["mean_score"] / baseline_summary["mean_score"]
                if baseline_summary["mean_score"]
                else None
            )
            latency_improvement = 1 - (
                candidate_summary["p95_latency_s"] / baseline_summary["p95_latency_s"]
            ) if baseline_summary["p95_latency_s"] else None
            throughput_improvement = (
                candidate_summary["throughput_output_tokens_s"] / baseline_summary["throughput_output_tokens_s"] - 1
            ) if baseline_summary["throughput_output_tokens_s"] else None
            c_cost = candidate_summary["cost_per_success_usd"]
            b_cost = baseline_summary["cost_per_success_usd"]
            cost_improvement = (1 - c_cost / b_cost) if c_cost is not None and b_cost else None
            metrics = {
                "quality_retention": quality_retention,
                "mean_score_delta": mean(deltas),
                "task_retention": task_retention,
                "high_risk_delta_ci95": [ci_low, ci_high],
                "latency_improvement": latency_improvement,
                "throughput_improvement": throughput_improvement,
                "cost_per_success_improvement": cost_improvement,
                "critical_safety_pass_rate": candidate_summary["critical_safety_pass_rate"],
                "error_rate": candidate_summary["error_rate"],
            }
            comparisons.append(
                {
                    "candidate": candidate.name,
                    "baseline": candidate.baseline,
                    "metrics": metrics,
                    "gates": self._evaluate_gates(metrics),
                }
            )
        return comparisons

    def _evaluate_gates(self, metrics: dict[str, Any]) -> dict[str, Any]:
        if not self.gates:
            return {"overall": "NOT_CONFIGURED", "checks": []}
        checks: list[dict[str, Any]] = []

        def check(name: str, value: float | None, threshold: float, direction: str = "min") -> None:
            if value is None:
                status = "NOT_MEASURED"
            else:
                status = "PASS" if (value >= threshold if direction == "min" else value <= threshold) else "FAIL"
            checks.append({"name": name, "value": value, "threshold": threshold, "direction": direction, "status": status})

        check("aggregate_quality_retention", metrics["quality_retention"], self.gates["aggregate_quality_retention_min"])
        critical_tasks = set(self.gates.get("critical_tasks", []))
        for task, value in metrics["task_retention"].items():
            threshold = (
                self.gates["critical_task_retention_min"]
                if task in critical_tasks
                else self.gates["noncritical_task_retention_min"]
            )
            check(f"task_retention:{task}", value, threshold)
        check(
            "high_risk_delta_ci95_lower",
            metrics["high_risk_delta_ci95"][0],
            self.gates["high_risk_delta_ci95_lower_min"],
        )
        check("critical_safety_pass_rate", metrics["critical_safety_pass_rate"], self.gates["critical_safety_pass_rate_min"])
        check("p95_latency_improvement", metrics["latency_improvement"], self.gates["p95_latency_improvement_min"])
        check("throughput_improvement", metrics["throughput_improvement"], self.gates["throughput_improvement_min"])
        check("cost_per_success_improvement", metrics["cost_per_success_improvement"], self.gates["cost_per_success_improvement_min"])
        check("error_rate", metrics["error_rate"], self.gates["error_rate_max"], direction="max")
        overall = "PASS" if checks and all(item["status"] == "PASS" for item in checks) else "BLOCK"
        return {"overall": overall, "checks": checks}


def run_from_files(config_path: str | Path, dataset_path: str | Path, gates_path: str | Path | None = None) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    gates = json.loads(Path(gates_path).read_text(encoding="utf-8")) if gates_path else None
    cases = load_dataset(dataset_path)
    return EvalOrchestrator(config, gates).run(cases)

