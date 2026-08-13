from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agentic import score_scenario
from .calibration import calibrate
from .dataset import load_dataset
from .loadtest import sweep
from .orchestrator import EvalOrchestrator, run_from_files
from .providers import build_provider
from .stats import Verdict, mean, non_inferiority
from .types import ModelSpec

ROOT = Path(__file__).resolve().parents[1]


def _demo_config() -> dict[str, Any]:
    return {
        "run_name": "offline-harness-proof-not-model-evidence",
        "repetitions": 1,
        "max_workers": 4,
        "temperature": 0.0,
        "seed": 42,
        "models": [
            {"name": "mock-uncompressed-baseline", "role": "baseline", "model": "mock", "base_url": "mock://baseline"},
            {
                "name": "mock-compressed-candidate",
                "role": "candidate",
                "baseline": "mock-uncompressed-baseline",
                "model": "mock",
                "base_url": "mock://candidate",
            },
        ],
    }


def _agentic_demo_config() -> dict[str, Any]:
    """Offline agentic demo with an explicit per-turn degradation.

    The default mock keys its failures off the trailing digits of the case id,
    which for per-turn ids (`...-t00`) would fire on a fixed turn index in every
    scenario -- an artefact, not a model. An explicit rate gives each turn an
    independent draw, which is what makes the compounding visible: an 8%
    per-call failure rate over a five-call trajectory is roughly a 34%
    end-to-end failure rate.
    """
    config = _demo_config()
    config["run_name"] = "offline-agentic-proof-not-model-evidence"
    config["models"][1]["metadata"] = {"mock_degradation": 0.30}
    return config


def _load_gates(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_config(path: str | None) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else _demo_config()


# ---------------------------------------------------------------------------
# agentic suite
# ---------------------------------------------------------------------------
def run_agentic(config: dict[str, Any], scenarios_path: str | Path, gates: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]
    specs = [ModelSpec(**raw) for raw in config["models"]]
    temperature = float(config.get("temperature", 0.0))
    seed = int(config.get("seed", 42))

    by_model: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for spec in specs:
        provider = build_provider(spec)
        rows = [score_scenario(s, provider, temperature, seed) for s in scenarios]
        by_model[spec.name] = rows
        by_path: dict[str, dict[str, Any]] = {}
        for row in rows:
            bucket = by_path.setdefault(row["control_path"], {"n": 0, "passed": 0})
            bucket["n"] += 1
            bucket["passed"] += int(row["success"])
        for bucket in by_path.values():
            bucket["rate"] = round(bucket["passed"] / bucket["n"], 4)
        summaries[spec.name] = {
            "scenarios": len(rows),
            "success_rate": round(mean(float(r["success"]) for r in rows), 4),
            "success_by_control_path": by_path,
            "policy_violations": sum(r["policy_violations"] for r in rows),
            "scope_violations": sum(r["scope_violations"] for r in rows),
            "violation_gates": sorted({g for r in rows for g in r["violation_gates"]}),
            "mean_turns": round(mean(r["turns"] for r in rows), 2),
            "mean_cascade_depth": round(mean(r["cascade_depth"] for r in rows), 2),
            "tool_precision": round(mean(r["tool_precision"] for r in rows), 4),
            "tool_recall": round(mean(r["tool_recall"] for r in rows), 4),
            "budget_exhausted": sum(1 for r in rows if r["terminal_reason"] == "budget_exhausted"),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "output_tokens_per_success": (
                sum(r["output_tokens"] for r in rows) / max(sum(1 for r in rows if r["success"]), 1)
            ),
        }

    comparisons = []
    margin = float(gates.get("agentic_success_margin", 0.05))
    max_violations = int(gates.get("agentic_policy_violations_max", 0))
    for spec in specs:
        if spec.role != "candidate" or not spec.baseline or spec.baseline not in by_model:
            continue
        candidate_rows = {r["scenario_id"]: r for r in by_model[spec.name]}
        baseline_rows = {r["scenario_id"]: r for r in by_model[spec.baseline]}
        shared = sorted(set(candidate_rows) & set(baseline_rows))
        result = non_inferiority(
            "agentic_success",
            [float(candidate_rows[s]["success"]) for s in shared],
            [float(baseline_rows[s]["success"]) for s in shared],
            margin,
            alpha=float(gates.get("alpha", 0.05)),
            power_target=float(gates.get("power_target", 0.80)),
        )
        candidate_summary = summaries[spec.name]
        baseline_summary = summaries[spec.baseline]
        violations = candidate_summary["policy_violations"]

        # Turn and cost inflation, gated separately from success.
        #
        # With a generous turn budget a per-call regression is absorbed by
        # retries: the trajectory still completes, so success looks unchanged
        # while every completion costs more turns and more tokens. The
        # degradation is real and entirely invisible to a success-rate gate --
        # until the budget binds, at which point success falls off a cliff.
        # Gating both catches the effect on the way up rather than after it
        # becomes an incident.
        turn_inflation = _inflation(baseline_summary["mean_turns"], candidate_summary["mean_turns"])
        cost_inflation = _inflation(
            baseline_summary["output_tokens_per_success"], candidate_summary["output_tokens_per_success"]
        )
        max_turn_inflation = float(gates.get("agentic_turn_inflation_max", 0.15))
        max_cost_inflation = float(gates.get("agentic_cost_inflation_max", 0.20))

        checks = [
            {**result.to_dict(), "name": "preservation:agentic_success"},
            {
                "name": "agentic_policy_violations",
                "value": violations,
                "threshold": max_violations,
                "direction": "max",
                "status": "PASS" if violations <= max_violations else "FAIL",
            },
            {
                "name": "agentic_turn_inflation",
                "value": turn_inflation,
                "threshold": max_turn_inflation,
                "direction": "max",
                "status": _threshold_status(turn_inflation, max_turn_inflation),
            },
            {
                "name": "agentic_cost_inflation",
                "value": cost_inflation,
                "threshold": max_cost_inflation,
                "direction": "max",
                "status": _threshold_status(cost_inflation, max_cost_inflation),
            },
            {
                "name": "agentic_budget_exhausted",
                "value": candidate_summary["budget_exhausted"],
                "threshold": baseline_summary["budget_exhausted"],
                "direction": "max",
                "status": _threshold_status(
                    candidate_summary["budget_exhausted"], baseline_summary["budget_exhausted"]
                ),
            },
        ]
        statuses = {c["status"] for c in checks}
        overall = (
            "BLOCK" if "FAIL" in statuses
            else ("INCONCLUSIVE" if Verdict.INCONCLUSIVE.value in statuses else "PASS")
        )
        comparisons.append(
            {
                "candidate": spec.name,
                "baseline": spec.baseline,
                # The headline of this suite. Compare it against the single-turn
                # delta: the gap between the two IS the compounding effect, and
                # it is the number a single-turn benchmark cannot produce.
                "end_to_end_success_delta": round(
                    summaries[spec.name]["success_rate"] - summaries[spec.baseline]["success_rate"], 4
                ),
                "turn_inflation": turn_inflation,
                "cost_inflation": cost_inflation,
                "checks": checks,
                "overall": overall,
            }
        )

    return {
        "schema_version": "1.0",
        "synthetic_demo": all(spec.base_url.startswith("mock://") for spec in specs),
        "scenarios": len(scenarios),
        "summaries": summaries,
        "comparisons": comparisons,
        "detail": by_model,
    }


def _inflation(baseline: float | None, candidate: float | None) -> float | None:
    """Fractional increase, for metrics where lower is better."""
    if not baseline or candidate is None:
        return None
    return round(candidate / baseline - 1, 4)


def _threshold_status(value: float | None, limit: float) -> str:
    if value is None:
        return "NOT_MEASURED"
    return "PASS" if value <= limit else "FAIL"


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pulsar / HyperNova bank validation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the clearly labeled offline synthetic proof")
    demo.add_argument("--dataset", required=True)
    demo.add_argument("--out", required=True)

    run = sub.add_parser("run", help="Run configured OpenAI-compatible endpoints")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--gates", default="config/acceptance_gates.json")
    run.add_argument("--out", required=True)

    agentic = sub.add_parser("agentic", help="Run the multi-turn tool-executing suite")
    agentic.add_argument("--config", default=None, help="omit for the offline mock")
    agentic.add_argument("--scenarios", default="data/agentic_scenarios.json")
    agentic.add_argument("--gates", default="config/acceptance_gates.json")
    agentic.add_argument("--out", required=True)

    load = sub.add_parser("loadtest", help="Concurrency sweep for throughput and tail latency")
    load.add_argument("--config", default=None, help="omit for the offline mock")
    load.add_argument("--levels", default="1,4,16,64")
    load.add_argument("--requests", type=int, default=32)
    load.add_argument("--prompt-tokens", type=int, default=512)
    load.add_argument("--max-tokens", type=int, default=256)
    load.add_argument("--out", required=True)

    validate = sub.add_parser(
        "validate-harness", help="Calibrate the instrument before trusting any number it produces"
    )
    validate.add_argument("--dataset", default="data/bank_eval_v1.jsonl")
    validate.add_argument("--gates", default="config/acceptance_gates.json")
    validate.add_argument("--out", default=None)

    args = parser.parse_args()

    if args.command == "demo":
        gates = _load_gates(ROOT / "config" / "acceptance_gates.json")
        result = EvalOrchestrator(_demo_config(), gates).run(load_dataset(args.dataset))
    elif args.command == "run":
        result = run_from_files(args.config, args.dataset, args.gates)
    elif args.command == "agentic":
        config = _load_config(args.config) if args.config else _agentic_demo_config()
        result = run_agentic(config, args.scenarios, _load_gates(args.gates))
    elif args.command == "loadtest":
        result = sweep(
            _load_config(args.config),
            levels=[int(x) for x in args.levels.split(",") if x.strip()],
            requests_per_level=args.requests,
            prompt_tokens=args.prompt_tokens,
            max_tokens=args.max_tokens,
        )
    else:
        result = calibrate(args.dataset, _load_gates(args.gates))

    if args.command == "validate-harness":
        _print_calibration(result)
        if args.out:
            _write(args.out, result)
        raise SystemExit(0 if result["overall"] == "PASS" else 1)

    _write(args.out, result)
    print(f"Wrote {args.out} | synthetic_demo={result.get('synthetic_demo')}")
    for comparison in result.get("comparisons", []):
        verdict = comparison.get("overall") or comparison.get("gates", {}).get("overall")
        if verdict:
            print(f"{comparison['candidate']} vs {comparison['baseline']}: {verdict}")
            if "end_to_end_success_delta" in comparison:
                print(f"  end-to-end success delta : {comparison['end_to_end_success_delta']:+.4f}")
                if comparison.get("turn_inflation") is not None:
                    print(f"  turns per completion     : {comparison['turn_inflation']:+.1%}")
                if comparison.get("cost_inflation") is not None:
                    print(f"  tokens per completion    : {comparison['cost_inflation']:+.1%}")


def _write(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_calibration(result: dict[str, Any]) -> None:
    print("\nHarness calibration -- validating the instrument, not any vendor model\n")
    for check in result["checks"]:
        print(f"  [{check['status']:<4}] {check['name']:<26} {check['detail']}")
    print()
    if result["overall"] == "PASS":
        print("  All checks passed. The harness detects a degradation of the size we care about,")
        print("  does not fire when there is none, and refuses to report an underpowered")
        print("  comparison as a pass. Safe to point at live endpoints.\n")
    else:
        print("  CALIBRATION FAILED -- do not run this harness against a vendor model.")
        print("  Any number it produced would be uninterpretable.\n")


if __name__ == "__main__":
    main()
