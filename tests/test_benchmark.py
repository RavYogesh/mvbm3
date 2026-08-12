from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.dataset import dataset_profile, load_dataset
from benchmark.graders import grade
from benchmark.orchestrator import EvalOrchestrator
from benchmark.types import EvalCase


ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_dataset_is_valid_and_balanced(self) -> None:
        cases = load_dataset(ROOT / "data" / "bank_eval_v1.jsonl")
        profile = dataset_profile(cases)
        self.assertEqual(profile["cases"], 60)
        self.assertEqual(len(profile["by_task"]), 12)
        self.assertTrue(all(count == 5 for count in profile["by_task"].values()))


class GraderTests(unittest.TestCase):
    def test_json_subset(self) -> None:
        case = EvalCase("x", "extraction_json", "low", "", grader={"type": "json_subset", "expected": {"a": 1}})
        score, passed, _ = grade(case, '{"a":1,"b":2}')
        self.assertEqual(score, 1.0)
        self.assertTrue(passed)

    def test_refusal_blocks_secret(self) -> None:
        case = EvalCase(
            "x",
            "pii_safety",
            "critical",
            "",
            grader={"type": "refusal", "forbidden": ["secret-123"]},
        )
        self.assertTrue(grade(case, "I cannot provide that secret.")[1])
        self.assertFalse(grade(case, "I cannot refuse: secret-123")[1])


class OrchestratorTests(unittest.TestCase):
    def test_offline_run_is_labeled_synthetic_and_audited(self) -> None:
        cases = load_dataset(ROOT / "data" / "bank_eval_v1.jsonl")[:12]
        config = {
            "run_name": "test",
            "repetitions": 1,
            "max_workers": 2,
            "models": [
                {"name": "base", "role": "baseline", "model": "mock", "base_url": "mock://base"},
                {"name": "candidate", "role": "candidate", "baseline": "base", "model": "mock", "base_url": "mock://candidate"},
            ],
        }
        gates = json.loads((ROOT / "config" / "acceptance_gates.json").read_text(encoding="utf-8"))
        result = EvalOrchestrator(config, gates).run(cases)
        self.assertTrue(result["synthetic_demo"])
        self.assertEqual(len(result["results"]), 24)
        self.assertGreater(len(result["audit_trail"]), 24)
        self.assertEqual(result["comparisons"][0]["candidate"], "candidate")


if __name__ == "__main__":
    unittest.main()

