from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_dataset
from .orchestrator import EvalOrchestrator, run_from_files


def _demo_config() -> dict[str, object]:
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
                "base_url": "mock://candidate"
            }
        ],
    }


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
    args = parser.parse_args()

    if args.command == "demo":
        gates_path = Path(__file__).resolve().parents[1] / "config" / "acceptance_gates.json"
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        result = EvalOrchestrator(_demo_config(), gates).run(load_dataset(args.dataset))
    else:
        result = run_from_files(args.config, args.dataset, args.gates)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output} | run_id={result['run_id']} | synthetic_demo={result['synthetic_demo']}")
    for comparison in result["comparisons"]:
        print(f"{comparison['candidate']} vs {comparison['baseline']}: {comparison['gates']['overall']}")


if __name__ == "__main__":
    main()

