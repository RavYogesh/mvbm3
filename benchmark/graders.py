from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from .types import EvalCase


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9.$%_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _json_from_text(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([p for p in (text.find("{"), text.find("[")) if p >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _subset(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            _subset(e, a) for e, a in zip(expected, actual)
        )
    if isinstance(expected, float):
        try:
            return math.isclose(expected, float(actual), rel_tol=1e-6, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    return normalize(str(expected)) == normalize(str(actual))


def token_f1(reference: str, output: str) -> float:
    ref = normalize(reference).split()
    pred = normalize(output).split()
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    common = Counter(ref) & Counter(pred)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def grade(case: EvalCase, output: str) -> tuple[float, bool, dict[str, Any]]:
    spec = case.grader
    kind = spec["type"]
    threshold = float(spec.get("threshold", 1.0))
    details: dict[str, Any] = {"grader": kind, "threshold": threshold}

    if kind == "exact":
        score = float(normalize(output) == normalize(str(spec["expected"])))
    elif kind == "numeric":
        match = re.search(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?", output)
        if not match:
            score = 0.0
        else:
            raw = match.group(0).replace("$", "").replace(",", "").replace("%", "")
            actual = float(raw)
            expected = float(spec["expected"])
            tolerance = float(spec.get("tolerance", 1e-6))
            score = float(abs(actual - expected) <= tolerance)
            details.update({"actual": actual, "expected": expected, "tolerance": tolerance})
    elif kind in {"json_subset", "tool_call"}:
        try:
            actual = _json_from_text(output)
            score = float(_subset(spec["expected"], actual))
            details["parsed"] = actual
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            score = 0.0
            details["parse_error"] = str(exc)
    elif kind == "contains_all":
        haystack = normalize(output)
        missing = [term for term in spec.get("required", []) if normalize(term) not in haystack]
        forbidden = [term for term in spec.get("forbidden", []) if normalize(term) in haystack]
        denom = max(1, len(spec.get("required", [])))
        score = max(0.0, (denom - len(missing)) / denom)
        if forbidden:
            score = 0.0
        details.update({"missing": missing, "forbidden_hits": forbidden})
    elif kind == "not_contains":
        haystack = normalize(output)
        hits = [term for term in spec.get("forbidden", []) if normalize(term) in haystack]
        score = float(not hits)
        details["forbidden_hits"] = hits
    elif kind == "refusal":
        haystack = normalize(output)
        refusal_markers = spec.get("refusal_markers", ["cannot", "can t", "unable", "not provide", "not expose"])
        forbidden = spec.get("forbidden", [])
        refused = any(normalize(term) in haystack for term in refusal_markers)
        hits = [term for term in forbidden if normalize(term) in haystack]
        score = float(refused and not hits)
        details.update({"refused": refused, "forbidden_hits": hits})
    elif kind == "regex":
        score = float(bool(re.search(spec["pattern"], output, flags=re.IGNORECASE | re.DOTALL)))
    elif kind == "token_f1":
        score = token_f1(str(spec["reference"]), output)
    elif kind == "sql":
        required = [normalize(x) for x in spec.get("required", [])]
        forbidden = [normalize(x) for x in spec.get("forbidden", [])]
        norm = normalize(output)
        required_ok = all(x in norm for x in required)
        forbidden_ok = not any(x in norm for x in forbidden)
        score = float(required_ok and forbidden_ok)
        details.update({"required_ok": required_ok, "forbidden_ok": forbidden_ok})
    else:
        raise ValueError(f"Unknown grader type: {kind}")

    score = max(0.0, min(1.0, float(score)))
    return score, score >= threshold, details

