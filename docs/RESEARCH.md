# Pulsar and HyperNova due-diligence memo

**Research cut:** 2026-08-13  
**Decision posture:** proceed to a controlled proof of concept; do not onboard for production from public benchmark claims alone.

## Executive finding

Multiverse Computing has published a plausible and technically grounded compression approach, open model artifacts, detailed benchmark tables, and serving instructions. The public evidence supports the hypothesis that both models can reduce parameter/weight footprint and improve inference performance. It does **not** establish uniform quality preservation across bank workloads.

The model cards themselves show task-dependent retention. Pulsar is close to its Nemotron base on AIME, GPQA-D, IFBench, and LiveCodeBench but has larger published gaps on Tau2-Telecom, SciCode, AA-LCR, and AA-Omni. HyperNova-60B-2605 is close to gpt-oss-120b on IFBench, GPQA-D, AIME, and Tau2-Telecom and exceeds it on the reported LiveCodeBench split, but has larger gaps on HLE, AA-LCR, SciCode, Terminal-Bench Hard, and Aider. That pattern is exactly why aggregate averages are unsafe as an onboarding criterion.

## Claim and evidence register

| Topic | Published claim / observation | Evidence class | Validation implication |
|---|---|---|---|
| Compression method | CompactifAI uses tensor-network decomposition and can compose with quantization. The 2024 paper reports up to 93% memory reduction on LLaMA-7B with 2–3% average accuracy loss across five classic benchmarks after a healing step. | Author paper, not Pulsar/HyperNova-specific | Reproduce on the exact candidate and intended tasks; clarify healing/fine-tuning data and contamination controls. |
| Pulsar footprint | Model card: Nemotron ~31.6B total / 3.6B active becomes Pulsar 16.15B total / 3.1B active, described as 50% compression. | Vendor model card | Verify exact revision, safetensor size, loaded VRAM, KV cache, and peak memory. |
| Pulsar quality | Published retention is near parity on several tasks, but approximately 76% on Tau2-Telecom, 81% on SciCode, 86% on AA-LCR, and 75% on AA-Omni relative to the listed base scores. | Vendor model card; NVIDIA reproduction is asserted on a vendor page | Require raw configs/results or reproduce independently. A public standalone NVIDIA technical report was not located in this research pass. |
| Pulsar efficiency | Vendor page reports B200 system throughput 3,363 → 3,760 tok/s for BF16 and ~4,800 tok/s for FP8/NVFP4; TTFT 2.18 → 1.80s (BF16) and ~1.25s for quantized variants; weight footprints as low as ~10GB for NVFP4. | Vendor-published benchmark, described as reproduced by NVIDIA | Run matched B200/L40S tests with identical context shape, concurrency, runtime, precision, and power telemetry. |
| HyperNova footprint | Model card reports 60B total / 4.8B active MoE. Its table reports 65GB → 32GB model weights versus gpt-oss-120b. | Vendor model card | Verify artifact bytes and peak loaded memory; total parameters, active parameters, and runtime memory are different constructs. |
| HyperNova efficiency | At concurrency 128, model card reports throughput 3,821 → 5,210 tok/s (+36%) and TTFT 7.04 → 4.85s (-31%). | Vendor model card | Reproduce under the same runtime and output-length distribution; include p50/p95/p99 and energy/request. |
| HyperNova third-party view | Artificial Analysis currently reports an Intelligence Index score of 18, ~333 output tok/s, ~0.91s TTFT, a 130k context window, and unusually high evaluation verbosity (140M output tokens vs 53M median). | Independent benchmark operator | Treat speed and price as provider-dependent. Track tokens per successful task because verbosity can erode unit economics. |
| Tool calling | Both cards state OpenAI-style/native tool-calling support. | Vendor model cards | Test schema validity, correct tool choice, arguments, abstention, multi-step recovery, and prompt-injected tool output. |
| Languages | Pulsar card says primarily English with added Spanish and no systematic evaluation outside those languages; HyperNova says other languages are not formally evaluated. | Vendor model cards | Gate only languages actually required; the starter set includes Spanish probes but is not sufficient for formal multilingual approval. |
| Licensing and supply chain | HyperNova is presented as Apache-2.0 open weights. Pulsar references an NVIDIA-licensed base while vendor material describes its release as Apache-2.0. Both examples may require `trust_remote_code=True`. | Model cards / repository instructions | Legal must review derivative/license obligations. Mirror artifacts, pin commit hashes, scan code, and prohibit unreviewed remote execution. |

## Published score retention

Retention below is candidate score divided by the listed base-model score. It is a diagnostic, not a claim that benchmark scales are interval-linear.

### Pulsar-16B versus Nemotron-3-Nano-30B-A3B

| Benchmark | Base | Pulsar | Retention |
|---|---:|---:|---:|
| MMLU-Pro | 78.30 | 74.78 | 95.5% |
| AIME 2025 | 87.29 | 87.22 | 99.9% |
| BFCL-v4 | 53.80 | 49.03 | 91.1% |
| LiveCodeBench | 67.40 | 65.45 | 97.1% |
| Tau2-Telecom | 40.94 | 31.29 | 76.4% |
| SciCode | 32.05 | 25.84 | 80.6% |
| AA-LCR | 34.00 | 29.33 | 86.3% |
| AA-Omni | 20.69 | 15.60 | 75.4% |
| IFBench | 71.97 | 70.79 | 98.4% |
| GPQA-D | 72.02 | 71.41 | 99.2% |
| HLE | 10.47 | 10.24 | 97.8% |

### HyperNova-60B-2605 versus gpt-oss-120b

| Benchmark | Base | HyperNova | Retention |
|---|---:|---:|---:|
| HLE | 18.5 | 15.0 | 81.1% |
| MMLU-Pro | 79.6 | 76.8 | 96.5% |
| AIME25 | 93.7 | 90.0 | 96.1% |
| GPQA-D | 74.6 | 71.9 | 96.4% |
| IFBench | 67.0 | 66.6 | 99.4% |
| AA-LCR | 49.0 | 40.3 | 82.2% |
| Tau2-Telecom | 63.7 | 61.7 | 96.9% |
| SciCode | 41.5 | 36.0 | 86.7% |
| LiveCodeBench | 62.8 | 68.7 | 109.4% |
| Terminal-Bench Hard | 24.2 | 15.9 | 65.7% |
| Aider | 43.6 | 34.2 | 78.4% |

## Material evidence gaps

1. **No bank-specific outcomes.** Public suites do not establish complaint accuracy, disclosure correctness, customer-data protection, policy adherence, or safe monetary/tool actions.
2. **No single stable “quality preserved” definition.** Some tasks are close to the base; others show double-digit relative drops. Critical-task floors matter more than an average.
3. **Uncertainty is under-reported.** Several model-card evaluations use one run; public tables generally lack confidence intervals and paired case-level results.
4. **Serving configuration matters.** Precision, runtime version, kernels, tensor parallelism, prompt length, output length, reasoning effort, and concurrency can dominate the efficiency result.
5. **Benchmark contamination is unresolved.** Request healing/fine-tuning data descriptions, cutoff dates, deduplication methods, and whether target benchmark data or close derivatives were present.
6. **Long-context capacity is not long-context quality.** Validate retrieval and reasoning at 8k/32k/64k/128k and, for Pulsar claims, beyond 128k, including distractors and repeated needles.
7. **Version metadata needs reconciliation.** Public pages use multiple dates/versions for HyperNova-2605 and benchmark methodology snapshots can change. Pin hashes; never evaluate a floating model name.
8. **Third-party claims need artifacts.** Obtain NVIDIA reproduction configs/raw results and the precise Artificial Analysis provider snapshot before treating them as independent control evidence.

## Benchmark design

### Comparison matrix

- Pulsar BF16, FP8, and NVFP4 against the exact Nemotron base, plus the bank incumbent.
- HyperNova-60B-2605 against the exact gpt-oss-120b baseline, plus the bank incumbent.
- Same hardware class, runtime build, tensor-parallel settings, max context, prompt template, and decode configuration within each pair.
- Separate runs for deterministic (`temperature=0`) application quality and vendor-reproduction settings.

### Quality and risk measures

- Deterministic exact/numeric/JSON/tool/SQL graders where possible.
- Groundedness, completeness, and unsupported-claim checks for free text.
- PII/secrets non-disclosure, prompt injection, unsafe tool selection, excessive agency, and prohibited action tests.
- Human review for all high/critical free-form failures and a blinded sample of passes.
- Task-level pass rate, mean score, worst-slice score, and candidate-minus-baseline paired confidence intervals.

### Efficiency measures

- Artifact size, loaded weights, peak GPU memory, KV-cache use, GPU count, and power draw.
- TTFT p50/p95/p99, inter-token latency, end-to-end latency, output tokens/s, requests/s, and error rate.
- Concurrency sweeps and prompt/output buckets (1k/256, 8k/1k, 32k/2k, and use-case-specific long context).
- GPU-hours and energy per 1,000 successful tasks; cost per successful task, not only cost per token.

### Statistical protocol

- Randomize paired request order after identical warm-up.
- Use at least three repetitions for stochastic settings; more for low-frequency safety events.
- Report paired bootstrap 95% confidence intervals. Use McNemar tests for paired binary outcomes when sample size permits.
- Correct for multiple task-family comparisons and publish all pre-registered slices.
- Freeze the rubric before unsealing the holdout set.

## Default onboarding decision gates

- 100% pass on critical privacy, secrets, prompt-injection, and prohibited-action cases.
- Aggregate quality retention at least 97% versus the required base.
- Critical task-family retention at least 95%; every other family at least 90%.
- Lower bound of the high-risk paired 95% confidence interval no worse than -2 percentage points.
- p95 latency at least 20% better **or** throughput at least 25% better on matched infrastructure.
- Cost per successful task at least 30% better.
- Error rate no greater than 0.5%, with no unexplained crash, malformed tool call, or evidence-loss condition.

Passing these gates supports a time-boxed shadow pilot, not unrestricted production use.

## Vendor evidence request

Request the following before the POC:

- Exact model commit hashes, tokenizer/chat templates, base-model hashes, and checksums.
- Full license chain and commercial-use opinion, especially for Pulsar/NVIDIA derivation.
- Healing/SFT data description, benchmark deduplication method, cutoff dates, and safety tuning details.
- Raw benchmark outputs, seeds, prompts, judge versions, parsers, exclusions, and error logs.
- NVIDIA reproduction statement/artifacts and hardware/software manifests.
- Supported serving runtimes, kernel requirements, known quality regressions by precision, and memory formulas.
- Security architecture, vulnerability disclosure, artifact signing/SBOM, patch SLA, and incident process.
- Data-processing terms, telemetry behavior, support model, roadmap, and model-change notification SLA.

## Primary sources

- Multiverse Computing, [Pulsar-16B model card](https://huggingface.co/MultiverseComputingCAI/Pulsar-16B-BF16)
- Multiverse Computing, [Pulsar-16B validation and performance announcement](https://multiversecomputing.com/resources/pulsar-16b-built-by-multiverse-computing-validated-by-nvidia)
- Multiverse Computing, [HyperNova-60B-2605 model card](https://huggingface.co/MultiverseComputingCAI/Hypernova-60B-2605)
- Artificial Analysis, [HyperNova-60B-2605 analysis](https://artificialanalysis.ai/models/hypernova-60b/)
- Tomut et al., [CompactifAI paper](https://arxiv.org/abs/2401.14109)
- Federal Reserve, [Supervisory Guidance on Model Risk Management (SR-26-2)](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm)
- NIST, [AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
- NIST, [Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
- OpenAI, [Evals API reference](https://developers.openai.com/api/reference/resources/evals/)

Regulatory note: the 2026 interagency model-risk guidance states that generative and agentic AI are outside its formal scope, while also saying governance should determine appropriate controls for tools not covered. This kit uses its vendor-validation, documentation, monitoring, and outcomes-analysis principles as a conservative governance analogue; it does not represent a legal determination that SR-26-2 applies to these models.

