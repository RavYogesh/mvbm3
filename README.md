# Multiverse Pulsar / HyperNova validation kit

This package is a bank-ready, reproducible benchmark for testing Multiverse Computing's Pulsar and HyperNova models against their uncompressed baselines and an incumbent bank model.

It is intentionally a **validation harness, not a pre-computed endorsement**. The included offline run uses synthetic mock providers to prove the software path. Replace those providers with approved OpenAI-compatible endpoints to produce decision evidence.

## What is included

- A curated, synthetic banking dataset spanning grounded QA, summarization, extraction, calculations, classification, tool use, SQL/code, safety, prompt injection, long-context retrieval, and Spanish-language support. Refusal cases are **twinned**: every harmful prompt has a benign near-mirror that must be answered.
- A multi-agent control tower that dispatches execution, deterministic grading, risk review, performance profiling, statistical analysis, and report generation.
- A **streaming** OpenAI-compatible provider adapter — real TTFT, reasoning-token accounting, retry with full jitter — for vLLM, SGLang, hosted endpoints, or an internal model gateway.
- Paired baseline-versus-candidate **non-inferiority testing** with pre-registered margins, power analysis, exact McNemar, safety gates, and an audit trail.
- A **multi-turn, tool-executing agentic suite** where policy gates are enforced in code and assertions are evaluated on world state.
- A **concurrency sweep** separating system throughput from per-stream decode rate, and locating the saturation knee.
- **Harness calibration** that proves the instrument can detect a degradation before it is pointed at a vendor.
- A Streamlit dashboard, CLI, unit tests, research memo, architecture diagrams, and explicit acceptance gates.

## Quick start: offline proof

Calibrate the instrument first. A benchmark harness is software, and software is wrong until shown otherwise — but a wrong harness fails in the most dangerous available direction: it reports a confident number.

```powershell
python -m benchmark.cli validate-harness
```

Then run the three suites:

```powershell
python -m benchmark.cli demo --dataset data/bank_eval_v1.jsonl --out results/demo_run.json
python -m benchmark.cli agentic --out results/demo_agentic.json
python -m benchmark.cli loadtest --levels 1,4,16,64 --out results/demo_loadtest.json
python -m unittest discover -s tests -v
```

Launch the dashboard:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens in **offline demo** mode. Results are labeled synthetic.

## Suites

| Command | What it measures | Why it is separate |
|---|---|---|
| `demo` / `run` | Single-turn correctness across 12 task families | Grading wants mixed prompt shapes; performance measurement does not |
| `agentic` | Multi-turn tool-executing trajectories with policy gates | Errors compound across turns. A 2-point per-call regression is roughly a 12-point end-to-end regression, and a single-turn suite cannot see it |
| `loadtest` | Throughput and tail latency across a concurrency sweep | A throughput number reported without its concurrency is not a measurement |
| `validate-harness` | Sensitivity, specificity, underpowered guard, grader self-checks | Proves the instrument works before its output is used as evidence |

## Connect real models

1. Copy `config/models.example.json` to a controlled location.
2. Set the endpoint URLs, exact model/revision identifiers, and API-key environment-variable names.
3. Pin the serving runtime, quantization, GPU type/count, tensor parallelism, context length, and decode configuration.
4. Run paired tests on the same hardware and workload shape.

```powershell
$env:PULSAR_API_KEY = "<secret>"
$env:BASELINE_API_KEY = "<secret>"
python -m benchmark.cli run --config config/models.example.json --dataset data/bank_eval_v1.jsonl --out results/real_run.json
```

For self-hosted vLLM/SGLang, an API key may be omitted. Do not place secrets in the JSON configuration or result bundle.

Per-model `metadata` switches: `stream` (default true — set false only for endpoints that cannot stream, in which case TTFT is reported as unmeasured rather than faked), `reasoning_effort`, `max_attempts`, `backoff_base_s`.

## Recommended comparison pairs

| Candidate | Required quality baseline | Operational comparator |
|---|---|---|
| Pulsar-16B BF16/FP8/NVFP4 | NVIDIA Nemotron-3-Nano-30B-A3B, same serving stack | Current bank-approved model |
| HyperNova-60B-2605 | OpenAI gpt-oss-120b, matched reasoning effort | Current bank-approved model |

Treat every precision as a **separate candidate with its own gate**. The vendor's efficiency headline and quality headline come from different builds, and quoting them together is the single easiest way to approve a model nobody actually measured.

Run each pair at least three times after warm-up, **at a temperature above zero** — repetitions at temperature 0 are near-deterministic, so they multiply cost without estimating sampling variance. The harness raises this as a design finding rather than letting it pass silently. Use a held-out dataset split, blind human review for high-risk outputs, and matched hardware. Do not compare a self-hosted candidate with a rate-limited hosted baseline and call the difference a model efficiency result.

## Decision contract

The gates in `config/acceptance_gates.json` are pre-registered. Changing a margin after seeing a result is a governance event, not a config edit.

**Quality is tested for non-inferiority against absolute margins, not retention ratios.** A ratio silently varies the tolerance with task difficulty: at 0.97 retention a baseline scoring 0.95 may drop 2.9 points, while one scoring 0.40 may drop only 1.2. Risk tier should set the tolerance, not the incidental difficulty of the task.

```
H0:  candidate <= baseline - margin      the model IS materially worse
H1:  candidate >  baseline - margin      preservation
```

The burden of proof sits with the candidate. Preservation is accepted only when the one-sided lower confidence bound clears the margin **and** the comparison had the power to detect a breach.

| Gate | Threshold |
|---|---|
| Aggregate preservation | non-inferior at 0.03 absolute |
| Critical task families | non-inferior at 0.02 |
| Other task families | non-inferior at 0.05 |
| High-risk cases | non-inferior at 0.02 |
| Critical safety pass rate | 100%, **and zero over-refusals** |
| p95 / p99 latency | 20% / 10% improvement |
| System throughput, wall-clock at stated concurrency | 25% improvement |
| p95 TTFT | 20% improvement |
| Cost per successful task | 30% improvement |
| Model vs infrastructure error rate | 0.5% / 2%, tracked separately |
| Agentic turn / cost inflation | 15% / 20% |

### Three verdicts, not two

`PASS`, `BLOCK`, and `INCONCLUSIVE`. The third exists because *"it is worse"* and *"we could not tell"* call for different actions — stop, versus collect more samples — and collapsing them is how a team ends up quietly loosening a margin so that an underpowered run passes.

### The sample-size constraint

At a discordance rate of 0.15, demonstrating non-inferiority needs:

| Margin | Paired cases per family |
|---|---|
| 0.05 | 371 |
| 0.03 | 1,031 |
| 0.02 | 2,319 |
| 0.01 | 9,274 |

**The bundled 66-case starter set cannot support these margins.** Not "is hard to" — cannot, arithmetically. The harness enforces the `minimum_cases_per_task` floor the config declares, prints `required_n` beside every verdict, and returns `INCONCLUSIVE` rather than a pass it cannot defend. Expand the dataset before a formal decision.

Tune gates only through the bank's documented risk-acceptance process.

## What the agentic suite adds

Tools **execute** against stateful objects, so assertions are evaluated on world state — ledger, audit log, message queue — never on the transcript. "The agent said it issued a credit" and "a credit exists in the ledger" are different events, and only the second is a result.

Policy gates are enforced **in code**. A prompt is a request; a gate is an auditable control. A blocked attempt is recorded as a violation rather than silently dropped, so the scorecard can report *the model attempted an unauthorised money movement N times, and the control held N times*.

Tool access is scoped per agent, so **no single agent can complete a controlled action alone**. A $260 credit requires the policy agent to obtain an approval token and the adjustment agent to spend it. There is no path where a confused model gets a lucky pass, which is what makes the success metric informative.

Watch turn and cost inflation, not only success rate. With a generous turn budget a per-call regression is absorbed by retries: the trajectory still completes, so success looks unchanged while every completion costs more turns and more tokens — until the budget binds, at which point success falls off a cliff. Both are gated.

## Data handling

All examples are synthetic. They resemble banking workflows but contain no customer data and are not legal, compliance, credit, or investment advice. Before production evaluation, add internally approved, de-identified examples sampled from the intended use cases and keep a sealed holdout split owned by an independent validation team.

## Evidence status

See `docs/RESEARCH.md`. Vendor and model-card numbers are catalogued as published claims. The harness does not mark them verified until a run records the exact endpoint, model revision, serving runtime, hardware, prompt, seed/decode settings, and raw outputs.

Known gaps in the vendor's own evidence that this harness cannot close on its own, and that belong in the model risk file: no fairness or disparate-impact evaluation of the compressed models, no formal multilingual evaluation (both vendors state English-centric training), and no published prompt-injection robustness data for agentic deployment.
