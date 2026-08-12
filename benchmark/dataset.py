from __future__ import annotations

import json
from pathlib import Path

from .types import EvalCase


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            case = EvalCase(**item)
            if case.id in seen:
                raise ValueError(f"Duplicate case id {case.id!r} at line {line_number}")
            if case.risk_tier not in {"low", "moderate", "high", "critical"}:
                raise ValueError(f"Invalid risk tier for {case.id}: {case.risk_tier}")
            if not case.grader.get("type"):
                raise ValueError(f"Missing grader type for {case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("Dataset is empty")
    return cases


def dataset_profile(cases: list[EvalCase]) -> dict[str, object]:
    by_task: dict[str, int] = {}
    by_risk: dict[str, int] = {}
    for case in cases:
        by_task[case.task] = by_task.get(case.task, 0) + 1
        by_risk[case.risk_tier] = by_risk.get(case.risk_tier, 0) + 1
    return {"cases": len(cases), "by_task": by_task, "by_risk": by_risk}

