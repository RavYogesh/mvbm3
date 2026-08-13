"""Assemble the evaluation dataset: curated core + templated generation.

    python data/build_dataset.py --scale stage-a      # 371/family, screens at 0.05
    python data/build_dataset.py --scale stage-b      # 2,319/family, gates at 0.02
    python data/build_dataset.py --per-family 500     # explicit

The hand-authored cases in `curated_core.jsonl` are always kept and always come
first. They carry construct validity -- they were written against real servicing
language. Generation is added on top purely to reach the sample size the margins
demand (see `bench.stats.required_n`), and every generated case is stamped
`provenance: templated` so the two can never be confused in analysis.

Nothing is written until every case has been validated: each one is run through
a clean oracle and must score 1.0. A case its own reference answer cannot pass is
broken, and once it is in the set it would be charged against every model
forever after as a capability failure.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.graders import grade                      # noqa: E402
from benchmark.providers import MockProvider             # noqa: E402
from benchmark.stats import required_n                   # noqa: E402
from benchmark.types import EvalCase                     # noqa: E402
from data.generators import GENERATORS                   # noqa: E402

HERE = Path(__file__).resolve().parent
CORE = HERE / "curated_core.jsonl"
OUT = HERE / "bank_eval_v1.jsonl"

# Margins mirror config/acceptance_gates.json. Stage A screens cheaply for a
# large regression; Stage B is the defensible gate for regulated paths.
PRESETS = {
    "smoke": 12,
    "stage-a": required_n(0.05, discordance=0.15),
    "stage-b": required_n(0.02, discordance=0.15),
    "tight": required_n(0.01, discordance=0.15),
}


def validate(cases: list[dict]) -> list[str]:
    """Every case must be passable by an oracle, and ids must be unique.

    The oracle is the clean mock, which answers each case from its own grader
    spec. If that cannot score 1.0 the case is unsatisfiable -- a broken
    expectation, a forbidden term that collides with a required one, a
    reference query that does not match its seed data.
    """
    problems: list[str] = []
    seen: set[str] = set()
    oracle = MockProvider("oracle", candidate=False)

    twins: dict[str, str] = {}
    for raw in cases:
        if raw["id"] in seen:
            problems.append(f"{raw['id']}: duplicate id")
        seen.add(raw["id"])
        if raw["risk_tier"] not in {"low", "moderate", "high", "critical"}:
            problems.append(f"{raw['id']}: bad risk tier {raw['risk_tier']}")
        if not raw["grader"].get("type"):
            problems.append(f"{raw['id']}: missing grader type")
            continue

        case = EvalCase(**raw)
        generation = oracle.generate(case, 0.0, 0)
        score, passed, detail = grade(case, generation.text)
        if not passed:
            problems.append(
                f"{raw['id']}: oracle scored {score:.2f} -- "
                f"{detail.get('failure') or detail.get('missing') or detail}"
            )
        twin = raw["grader"].get("twin_id")
        if twin:
            twins[raw["id"]] = twin

    for case_id, twin in twins.items():
        if twin not in seen:
            problems.append(f"{case_id}: twin {twin} is missing from the set")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=sorted(PRESETS), default="stage-a")
    ap.add_argument("--per-family", type=int, default=None,
                    help="override the preset with an explicit per-family target")
    ap.add_argument("--families", nargs="*", default=sorted(GENERATORS))
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    target = args.per_family or PRESETS[args.scale]
    core = [json.loads(line) for line in CORE.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_family = collections.defaultdict(list)
    for raw in core:
        by_family[raw["task"]].append(raw)

    print(f"target {target} cases per family "
          f"({args.scale if not args.per_family else 'explicit'})")
    print(f"curated core: {len(core)} cases across {len(by_family)} families\n")

    cases = list(core)
    print(f"{'family':<20}{'curated':>9}{'generated':>11}{'total':>8}")
    print("-" * 48)
    for family in args.families:
        have = len(by_family.get(family, []))
        need = max(target - have, 0)
        rng = random.Random(f"{args.seed}:{family}")
        generated = GENERATORS[family](rng, need) if need else []
        cases.extend(generated)
        print(f"{family:<20}{have:>9}{len(generated):>11}{have + len(generated):>8}")

    print("-" * 48)
    print(f"{'TOTAL':<20}{len(core):>9}{len(cases) - len(core):>11}{len(cases):>8}\n")

    print("validating every case against an oracle ...")
    problems = validate(cases)
    if problems:
        print(f"\n{len(problems)} problem(s) — nothing written:\n")
        for line in problems[:25]:
            print("  " + line)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        raise SystemExit(1)
    print("  all cases satisfiable, ids unique, twins paired\n")

    out = Path(args.out)
    out.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8"
    )
    tiers = collections.Counter(c["risk_tier"] for c in cases)
    print(f"wrote {out}  ({len(cases)} cases)")
    print(f"  risk tiers: {dict(sorted(tiers.items()))}")


if __name__ == "__main__":
    main()
