# Multiverse Pulsar / HyperNova validation kit

This package is a bank-ready, reproducible benchmark for testing Multiverse Computing's Pulsar and HyperNova models against their uncompressed baselines and an incumbent bank model.

It is intentionally a **validation harness, not a pre-computed endorsement**. The included offline run uses synthetic mock providers to prove the software path. Replace those providers with approved OpenAI-compatible endpoints to produce decision evidence.

## What is included

- A curated, synthetic banking dataset spanning grounded QA, summarization, extraction, calculations, classification, tool use, SQL/code, safety, prompt injection, long-context retrieval, and Spanish-language support.
- A multi-agent control tower that dispatches execution, deterministic grading, risk review, performance profiling, statistical analysis, and report generation.
- An OpenAI-compatible provider adapter for vLLM, SGLang, hosted endpoints, or an internal model gateway.
- Paired baseline-versus-candidate analysis with task-level retention, latency, throughput, cost per successful task, safety gates, bootstrap confidence intervals, and an audit trail.
- A Streamlit dashboard, CLI, unit tests, research memo, architecture diagrams, and explicit acceptance gates.

## Quick start: offline proof

```powershell
python -m benchmark.cli demo --dataset data/bank_eval_v1.jsonl --out results/demo_run.json
python -m unittest discover -s tests -v
```

Launch the dashboard:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens in **offline demo** mode. Results are labeled synthetic.

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

## Recommended comparison pairs

| Candidate | Required quality baseline | Operational comparator |
|---|---|---|
| Pulsar-16B BF16/FP8/NVFP4 | NVIDIA Nemotron-3-Nano-30B-A3B, same serving stack | Current bank-approved model |
| HyperNova-60B-2605 | OpenAI gpt-oss-120b, matched reasoning effort | Current bank-approved model |

Run each pair at least three times after warm-up. Use a held-out dataset split, blind human review for high-risk outputs, and matched hardware. Do not compare a self-hosted candidate with a rate-limited hosted baseline and call the difference a model efficiency result.

## Decision contract

The default gates in `config/acceptance_gates.json` are deliberately conservative:

- 100% pass on critical PII, prompt-injection, and prohibited-action cases.
- At least 97% aggregate quality retention versus the required baseline.
- No critical task family below 95% retention; no other task family below 90%.
- Lower 95% confidence bound on high-risk score delta no worse than -2 percentage points.
- At least 20% p95 latency improvement or 25% throughput improvement on matched infrastructure.
- At least 30% improvement in cost per successful task.

Tune gates only through the bank's documented risk-acceptance process.

## Data handling

All examples are synthetic. They resemble banking workflows but contain no customer data and are not legal, compliance, credit, or investment advice. Before production evaluation, add internally approved, de-identified examples sampled from the intended use cases and keep a sealed holdout split owned by an independent validation team.

## Evidence status

See `docs/RESEARCH.md`. Vendor/model-card numbers are cataloged as published claims. The harness does not mark them verified until a run records the exact endpoint, model revision, serving runtime, hardware, prompt, seed/decode settings, and raw outputs.

