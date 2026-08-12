# Curated starter dataset

`bank_eval_v1.jsonl` contains 60 synthetic cases across 12 task families. Every case has an explicit risk tier, deterministic grader, provenance label, and no real customer data.

This is the **starter calibration set**, not the full statistical decision set. For onboarding evidence:

1. Add at least 30 independently curated examples per task family.
2. Source de-identified cases from the actual intended workflows.
3. Keep 20–30% as a sealed holdout owned by Model Risk or another independent validation function.
4. Deduplicate against any fine-tuning, healing, or vendor evaluation corpus.
5. Require blind dual review for high/critical free-form cases and report inter-annotator agreement.
6. Version the dataset, rubrics, and exclusions; never silently edit a released test set.

Task families: grounded QA, summarization, JSON extraction, calculations, classification, tool calls, SQL/code, PII safety, prompt injection, policy compliance, long-context retrieval, and Spanish-language support.

