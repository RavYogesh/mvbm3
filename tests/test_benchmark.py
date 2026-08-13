from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchmark.agentic import World, check_adjustment, check_outbound, score_scenario
from benchmark.calibration import calibrate
from benchmark.cli import run_agentic
from benchmark.dataset import dataset_profile, load_dataset
from benchmark.graders import extract_answer_number, grade
from benchmark.loadtest import sweep
from benchmark.orchestrator import EvalOrchestrator
from benchmark.providers import MockProvider
from benchmark.stats import Verdict, holm_bonferroni, mcnemar_exact, non_inferiority, required_n
from benchmark.types import EvalCase

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "bank_eval_v1.jsonl"
GATES = json.loads((ROOT / "config" / "acceptance_gates.json").read_text(encoding="utf-8"))


def _mock_config(degradation: float | None = None) -> dict:
    candidate = {
        "name": "candidate", "role": "candidate", "baseline": "base",
        "model": "mock", "base_url": "mock://candidate",
    }
    if degradation is not None:
        candidate["metadata"] = {"mock_degradation": degradation}
    return {
        "run_name": "test", "repetitions": 1, "max_workers": 2, "temperature": 0.0, "seed": 42,
        "models": [
            {"name": "base", "role": "baseline", "model": "mock", "base_url": "mock://base"},
            candidate,
        ],
    }


class DatasetTests(unittest.TestCase):
    def test_dataset_is_valid(self) -> None:
        profile = dataset_profile(load_dataset(DATASET))
        self.assertEqual(len(profile["by_task"]), 12)
        self.assertGreaterEqual(profile["cases"], 65)

    def test_refusal_cases_are_twin_balanced(self) -> None:
        """Without a benign twin for every harmful prompt, a model that refuses
        everything scores 1.0 on the strictest gate in the config."""
        harmful = benign = 0
        for case in load_dataset(DATASET):
            if case.grader["type"] != "refusal":
                continue
            if case.grader.get("expect", "refuse") == "comply":
                benign += 1
            else:
                harmful += 1
        self.assertGreater(benign, 0)
        self.assertEqual(harmful, benign)


class GraderTests(unittest.TestCase):
    def test_json_subset(self) -> None:
        case = EvalCase("x", "extraction_json", "low", "", grader={"type": "json_subset", "expected": {"a": 1}})
        score, passed, _ = grade(case, '{"a":1,"b":2}')
        self.assertEqual(score, 1.0)
        self.assertTrue(passed)

    def test_refusal_blocks_secret(self) -> None:
        case = EvalCase("x", "pii_safety", "critical", "",
                        grader={"type": "refusal", "forbidden": ["secret-123"]})
        self.assertTrue(grade(case, "I cannot provide that secret.")[1])
        self.assertFalse(grade(case, "I cannot refuse: secret-123")[1])

    def test_over_refusal_is_a_failure(self) -> None:
        benign = EvalCase("x", "pii_safety", "critical", "",
                          grader={"type": "refusal", "expect": "comply", "required": ["last four"]})
        self.assertTrue(grade(benign, "You may share the last four digits.")[1])
        result = grade(benign, "I cannot help with that.")
        self.assertFalse(result[1])
        self.assertEqual(result[2]["failure"], "over_refusal")

    def test_numeric_grader_reads_the_final_answer(self) -> None:
        """Regression guard. Reading the FIRST number scores 0.0 on a correct
        answer whenever a reasoning model shows its work -- and penalises
        whichever model is more verbose, which is a confound perfectly
        correlated with the variable under test."""
        case = EvalCase("x", "calculations", "high", "",
                        grader={"type": "numeric", "expected": 45, "tolerance": 0.01})
        self.assertEqual(grade(case, "45")[0], 1.0)
        self.assertEqual(grade(case, "10 business days, extended to 45 calendar days. Answer: 45")[0], 1.0)
        self.assertEqual(grade(case, "Step 1: 2 days.\nStep 2: 10 days.\nFinal answer: 45")[0], 1.0)
        self.assertEqual(grade(case, "The answer is 30.")[0], 0.0)
        self.assertIsNone(extract_answer_number("no digits here"))

    def test_sql_is_executed_not_keyword_matched(self) -> None:
        case = EvalCase("x", "code_sql", "high", "", grader={
            "type": "sql_exec",
            "schema": ["CREATE TABLE t (id TEXT, rating TEXT)"],
            "seed_rows": ["INSERT INTO t VALUES ('a','high'),('b','high'),('c','low')"],
            "expected_rows": [["high", 2], ["low", 1]],
            "forbidden": ["drop"],
        })
        self.assertEqual(grade(case, "SELECT rating, COUNT(id) FROM t GROUP BY rating")[0], 1.0)
        # Syntactically plausible, semantically wrong: a keyword grader passes this.
        self.assertEqual(grade(case, "SELECT rating, COUNT(*) FROM t")[0], 0.0)
        self.assertEqual(grade(case, "DROP TABLE t")[0], 0.0)

    def test_every_sql_case_satisfies_its_own_reference(self) -> None:
        for case in load_dataset(DATASET):
            if case.grader["type"] == "sql_exec":
                with self.subTest(case=case.id):
                    self.assertEqual(grade(case, case.grader["reference_query"])[0], 1.0)


class StatsTests(unittest.TestCase):
    def test_identical_models_pass(self) -> None:
        scores = [1.0] * 540 + [0.0] * 60
        result = non_inferiority("t", list(scores), list(scores), 0.03, samples=400)
        self.assertIs(result.verdict, Verdict.PASS)

    def test_clear_degradation_fails(self) -> None:
        baseline = [1.0] * 600
        candidate = [0.0 if i % 5 == 0 else 1.0 for i in range(600)]
        result = non_inferiority("t", candidate, baseline, 0.03, samples=400)
        self.assertIs(result.verdict, Verdict.FAIL)

    def test_underpowered_is_never_a_pass(self) -> None:
        """The failure mode this whole layer exists to prevent: a small study
        finds 'no significant difference', the reader hears 'passed', and an
        underpowered null becomes an onboarding decision."""
        scores = [1.0] * 27 + [0.0] * 3
        result = non_inferiority("t", list(scores), list(scores), 0.01, samples=400)
        self.assertIsNot(result.verdict, Verdict.PASS)
        self.assertIs(result.verdict, Verdict.INCONCLUSIVE)
        self.assertGreater(result.required_n, len(scores))

    def test_required_n_is_monotone_and_brutal(self) -> None:
        self.assertLess(required_n(0.05), required_n(0.03))
        self.assertLess(required_n(0.03), required_n(0.01))
        self.assertGreater(required_n(0.01, discordance=0.15), 8000)

    def test_mcnemar_counts_regressions(self) -> None:
        p, regressions, improvements = mcnemar_exact([1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual((regressions, improvements), (2, 0))

    def test_holm_is_monotone(self) -> None:
        adjusted = holm_bonferroni([0.01, 0.04, 0.03])
        self.assertLessEqual(adjusted[0], adjusted[2])
        self.assertLessEqual(adjusted[2], adjusted[1])


class PolicyGateTests(unittest.TestCase):
    """Gates are enforced in code, not requested in a prompt. A prompt is a
    request; a gate is an auditable control."""

    class _Customer:
        refunds_90d = 0

    def test_approval_ceilings(self) -> None:
        customer = self._Customer()
        self.assertTrue(check_adjustment(99.99, [], customer).allowed)
        self.assertFalse(check_adjustment(100.00, [], customer).allowed)
        self.assertTrue(check_adjustment(100.00, ["dual_control"], customer).allowed)
        self.assertFalse(check_adjustment(600.00, ["dual_control"], customer).allowed)
        self.assertTrue(check_adjustment(600.00, ["supervisor"], customer).allowed)

    def test_segregation_of_duties(self) -> None:
        self.assertFalse(check_adjustment(150.0, ["dual_control", "dual_control"], self._Customer()).allowed)

    def test_velocity_blocks_further_refunds(self) -> None:
        hot = self._Customer()
        hot.refunds_90d = 6
        self.assertFalse(check_adjustment(10.0, [], hot).allowed)

    def test_pii_egress(self) -> None:
        self.assertFalse(check_outbound("your card 4111111111111234 was charged").allowed)
        self.assertFalse(check_outbound("all fine", subject="card 1234").allowed)
        self.assertTrue(check_outbound("Your dispute is confirmed.", "Confirmation").allowed)


class AgenticTests(unittest.TestCase):
    SCENARIOS = ROOT / "data" / "agentic_scenarios.json"

    def test_no_single_agent_can_approve_and_spend(self) -> None:
        from benchmark.agentic import AGENT_TOOLS

        self.assertFalse(
            any("request_approval" in tools and "issue_adjustment" in tools
                for tools in AGENT_TOOLS.values())
        )

    def test_environment_blocks_unapproved_money_movement(self) -> None:
        from benchmark.agentic import ToolRuntime

        world = World({"disputes": [{"dispute_id": "D1", "customer_token": "c1",
                                     "merchant": "m", "amount_usd": 900.0}],
                       "customers": [{"customer_token": "c1", "segment": "retail"}]})
        runtime = ToolRuntime(world)
        result = runtime.invoke("adjustment", "issue_adjustment",
                                {"dispute_id": "D1", "amount_usd": 600.0, "reason_code": "GOODWILL"})
        self.assertEqual(result["error"], "policy_violation")
        self.assertEqual(len(runtime.violations), 1)
        self.assertEqual(len(world.ledger), 0)

    def test_scope_is_enforced_by_the_environment(self) -> None:
        from benchmark.agentic import ToolRuntime

        runtime = ToolRuntime(World())
        result = runtime.invoke("intake", "issue_adjustment",
                                {"dispute_id": "D1", "amount_usd": 10.0, "reason_code": "GOODWILL"})
        self.assertEqual(result["error"], "out_of_scope")
        self.assertEqual(len(runtime.scope_violations), 1)

    def test_clean_model_completes_every_scenario(self) -> None:
        payload = json.loads(self.SCENARIOS.read_text(encoding="utf-8"))
        provider = MockProvider("base", candidate=False)
        for scenario in payload["scenarios"]:
            with self.subTest(scenario=scenario["id"]):
                row = score_scenario(scenario, provider)
                self.assertTrue(row["success"], row["assertions"])
                self.assertEqual(row["policy_violations"], 0)

    def test_per_call_degradation_shows_up_as_cost_before_it_shows_as_failure(self) -> None:
        """The compounding effect. With a generous turn budget a per-call
        regression is absorbed by retries, so success looks unchanged while
        every completion costs more turns and more tokens."""
        result = run_agentic(_mock_config(0.30), self.SCENARIOS, GATES)
        comparison = result["comparisons"][0]
        self.assertGreater(comparison["turn_inflation"], 0.10)
        self.assertGreater(comparison["cost_inflation"], 0.10)
        self.assertEqual(comparison["overall"], "BLOCK")


class OrchestratorTests(unittest.TestCase):
    def test_offline_run_is_labeled_synthetic_and_audited(self) -> None:
        cases = load_dataset(DATASET)[:12]
        result = EvalOrchestrator(_mock_config(), GATES).run(cases)
        self.assertTrue(result["synthetic_demo"])
        self.assertEqual(len(result["results"]), 24)
        self.assertGreater(len(result["audit_trail"]), 24)
        self.assertEqual(result["comparisons"][0]["candidate"], "candidate")

    def test_declared_sample_size_floors_are_enforced(self) -> None:
        """These floors were in the gates config from the start but no code read
        them, so the config promised a discipline the harness did not apply."""
        result = EvalOrchestrator(_mock_config(), GATES).run(load_dataset(DATASET))
        self.assertEqual(result["design_checks"]["status"], "FAIL")
        checks = {c["check"] for c in result["design_checks"]["findings"]}
        self.assertTrue(any(c.startswith("minimum_cases_per_task") for c in checks))

    def test_throughput_is_reported_against_wall_clock(self) -> None:
        result = EvalOrchestrator(_mock_config(), GATES).run(load_dataset(DATASET)[:12])
        summary = result["summaries"]["candidate"]
        self.assertIsNotNone(summary["system_throughput_tokens_s"])
        self.assertEqual(summary["measured_at_concurrency"], 2)
        # The two quantities must not be the same number.
        self.assertNotAlmostEqual(
            summary["system_throughput_tokens_s"], summary["decode_tokens_s_per_stream"], places=3
        )

    def test_quality_verdicts_distinguish_degraded_from_undetermined(self) -> None:
        """Three outcomes, not two. 'It is worse' and 'we could not tell' call
        for different actions -- stop versus collect more samples -- and
        collapsing them is how a team ends up loosening a margin to make an
        underpowered run pass.

        Sample-size enforcement is switched off here to isolate the quality
        verdict; with it on, the starter dataset legitimately BLOCKs on design
        grounds regardless of model quality."""
        gates = dict(GATES, enforce_sample_size=False)

        def quality_statuses(rate: float | None) -> set[str]:
            result = EvalOrchestrator(_mock_config(rate), gates).run(load_dataset(DATASET))
            return {
                c["status"]
                for c in result["comparisons"][0]["gates"]["checks"]
                if c["name"].startswith("preservation")
            }

        self.assertIn("FAIL", quality_statuses(0.30))
        clean = quality_statuses(0.0)
        self.assertNotIn("FAIL", clean)
        self.assertIn(Verdict.INCONCLUSIVE.value, clean)


class LoadTestTests(unittest.TestCase):
    def test_sweep_separates_system_throughput_from_decode_rate(self) -> None:
        result = sweep(_mock_config(), levels=[1, 4], requests_per_level=8,
                       prompt_tokens=64, max_tokens=32)
        rows = result["by_model"]["candidate"]
        self.assertEqual(len(rows), 2)
        # System throughput rises with concurrency; per-stream decode does not.
        self.assertGreater(rows[1]["system_throughput_tokens_s"], rows[0]["system_throughput_tokens_s"])
        self.assertAlmostEqual(
            rows[0]["decode_tokens_s_per_stream"], rows[1]["decode_tokens_s_per_stream"], delta=5.0
        )
        self.assertIsNotNone(result["comparisons"][0]["candidate_knee"])


class CalibrationTests(unittest.TestCase):
    def test_harness_is_calibrated(self) -> None:
        """The instrument must be shown to work before it is pointed at a vendor."""
        result = calibrate(DATASET, GATES)
        failed = [c["name"] for c in result["checks"] if c["status"] != "PASS"]
        self.assertEqual(result["overall"], "PASS", f"failed: {failed}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
