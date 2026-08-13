from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from benchmark.dataset import dataset_profile, load_dataset
from benchmark.orchestrator import EvalOrchestrator


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "bank_eval_v1.jsonl"
GATES = ROOT / "config" / "acceptance_gates.json"
DEMO_RESULT = ROOT / "results" / "demo_run.json"


def demo_config() -> dict[str, object]:
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
                "base_url": "mock://candidate",
            },
        ],
    }


def run_eval(config: dict[str, object]) -> dict[str, object]:
    cases = load_dataset(DATASET)
    gates = json.loads(GATES.read_text(encoding="utf-8"))
    return EvalOrchestrator(config, gates).run(cases)


def load_initial_result() -> dict[str, object] | None:
    if DEMO_RESULT.exists():
        return json.loads(DEMO_RESULT.read_text(encoding="utf-8"))
    return None


st.set_page_config(page_title="Model Validation Control Tower", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    :root { --bank-red:#D71920; --bank-gold:#F6C344; --ink:#202124; }
    .stApp { background: #FAF8F3; }
    h1, h2, h3 { color: var(--ink); }
    div[data-testid="stMetric"] { background:white; border-top:4px solid var(--bank-red); padding:14px; }
    .control-banner { background:var(--bank-red); color:white; padding:18px 22px; border-left:10px solid var(--bank-gold); margin-bottom:18px; }
    .control-banner b { color:white; }
    .synthetic { background:#FFF2CC; border:1px solid #D6B656; padding:10px 14px; color:#5F4600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="control-banner"><b>GENAI MODEL VALIDATION CONTROL TOWER</b><br>'
    'Dispatch paired evaluations. Preserve raw evidence. Gate by risk, quality, and unit economics.</div>',
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = load_initial_result()

with st.sidebar:
    st.header("Run control")
    mode = st.radio("Mode", ["Offline proof", "Approved endpoints"], help="Offline proof validates software only.")
    uploaded = None
    if mode == "Approved endpoints":
        uploaded = st.file_uploader("Model configuration JSON", type=["json"])
        st.caption("Secrets are read from environment variables named in the file.")
    if st.button("Dispatch evaluation", type="primary", use_container_width=True):
        if mode == "Offline proof":
            config = demo_config()
        elif uploaded:
            config = json.loads(uploaded.getvalue().decode("utf-8"))
        else:
            st.error("Upload a model configuration first.")
            st.stop()
        with st.status("Dispatching agents…", expanded=True) as status:
            st.write("Preflight: dataset and gate contract loaded")
            result = run_eval(config)
            st.write(f"Execution and grading complete: {len(result['results'])} case results")
            status.update(label="Evaluation complete", state="complete")
        st.session_state.result = result
    st.divider()
    profile = dataset_profile(load_dataset(DATASET))
    st.caption(f"Starter suite: {profile['cases']} cases / {len(profile['by_task'])} task families")

result = st.session_state.result
if not result:
    st.info("Dispatch the offline proof to populate the control tower.")
    st.stop()

if result.get("synthetic_demo"):
    st.markdown(
        '<div class="synthetic"><b>SYNTHETIC SOFTWARE PROOF</b> — these are mock outputs, not Pulsar or HyperNova evidence.</div>',
        unsafe_allow_html=True,
    )

summaries = result["summaries"]
summary_rows = []
for model, values in summaries.items():
    summary_rows.append(
        {
            "model": model,
            "mean_score": values["mean_score"],
            "pass_rate": values["pass_rate"],
            "p95_latency_s": values["p95_latency_s"],
            "throughput_tok_s": values["throughput_output_tokens_s"],
            "error_rate": values["error_rate"],
        }
    )
summary_df = pd.DataFrame(summary_rows)

overview, dispatch, quality, efficiency, risk, evidence = st.tabs(
    ["Decision", "Dispatch", "Quality", "Efficiency", "Risk & controls", "Evidence"]
)

with overview:
    comparison = result["comparisons"][0] if result["comparisons"] else None
    cols = st.columns(5)
    if comparison:
        metrics = comparison["metrics"]
        aggregate = next(
            (p for p in metrics["preservation"] if p["name"] == "aggregate"), {}
        )
        safety = metrics["critical_safety_pass_rate"]
        cols[0].metric("Decision gate", comparison["gates"]["overall"])
        cols[1].metric(
            "Aggregate quality delta",
            f"{aggregate.get('delta', 0):+.3f}",
            help=f"one-sided lower bound {aggregate.get('lower_bound', 0):+.3f} "
                 f"against a margin of -{aggregate.get('margin', 0):.2f}",
        )
        cols[2].metric("p95 latency gain", f"{(metrics['p95_latency_improvement'] or 0):.1%}")
        cols[3].metric(
            "System throughput gain",
            f"{(metrics['system_throughput_improvement'] or 0):.1%}",
            help="measured against wall-clock elapsed, not the sum of request latencies",
        )
        cols[4].metric(
            "Critical safety",
            "n/a" if safety is None else f"{safety:.1%}",
            delta=f"-{metrics['over_refusals']} over-refusals" if metrics["over_refusals"] else None,
            delta_color="inverse",
        )
        st.subheader("Gate contract")
        st.dataframe(
            pd.DataFrame(comparison["gates"]["checks"]), use_container_width=True, hide_index=True
        )
    if result.get("design_checks", {}).get("findings"):
        st.subheader("Experimental design")
        st.dataframe(
            pd.DataFrame(result["design_checks"]["findings"]),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Three verdicts, not two. INCONCLUSIVE means the run could not have detected a breach "
        "of the margin even if one existed -- it is not a soft pass, and NOT_MEASURED blocks "
        "approval rather than counting as one."
    )

with dispatch:
    st.subheader("Agent lanes")
    agent_counts = pd.DataFrame(result["audit_trail"]).groupby(["agent", "event"]).size().reset_index(name="events")
    st.dataframe(agent_counts, use_container_width=True, hide_index=True)
    st.subheader("Execution manifest")
    manifest = pd.DataFrame(result["config"]["models"])
    st.dataframe(manifest, use_container_width=True, hide_index=True)
    st.caption(f"Run ID: {result['run_id']} | Created: {result['created_at']}")

with quality:
    st.subheader("Score by task family")
    task_rows = []
    for model, values in summaries.items():
        for task, item in values["by_task"].items():
            task_rows.append({"model": model, "task": task, "mean_score": item["mean_score"]})
    task_df = pd.DataFrame(task_rows)
    fig = px.bar(
        task_df,
        x="task",
        y="mean_score",
        color="model",
        barmode="group",
        range_y=[0, 1],
        color_discrete_sequence=["#6B6B6B", "#D71920", "#F6C344"],
    )
    fig.update_layout(xaxis_title="", yaxis_title="Mean score", legend_title="")
    st.plotly_chart(fig, use_container_width=True)
    if result["comparisons"]:
        st.subheader("Non-inferiority by family")
        preservation = pd.DataFrame(result["comparisons"][0]["metrics"]["preservation"])[
            ["name", "n", "delta", "lower_bound", "margin", "observed_power", "required_n", "status"]
        ]
        st.dataframe(preservation, use_container_width=True, hide_index=True)
        st.caption(
            "Preservation is accepted only when the one-sided lower bound clears the margin AND "
            "the comparison had the power to detect a breach. Where observed_power is low, "
            "required_n is the case count that margin actually demands."
        )

with efficiency:
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            summary_df,
            x="model",
            y="p95_latency_s",
            color="model",
            color_discrete_sequence=["#6B6B6B", "#D71920"],
            title="p95 latency — lower is better",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="seconds")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            summary_df,
            x="model",
            y="throughput_tok_s",
            color="model",
            color_discrete_sequence=["#6B6B6B", "#D71920"],
            title="Output throughput — higher is better",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="output tokens / second")
        st.plotly_chart(fig, use_container_width=True)
    st.warning("Use matched hardware, runtime, context shape, concurrency, and warm-up. API-provider speed is not model-only evidence.")

with risk:
    results_df = pd.DataFrame(result["results"])
    critical = results_df[results_df["risk_tier"].isin(["high", "critical"])]
    failures = critical[critical["passed"] == False]  # noqa: E712
    st.metric("High/critical failures", len(failures))
    st.dataframe(
        failures[["model", "case_id", "task", "risk_tier", "score", "error"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Raw customer data, autonomous monetary actions, and unreviewed adverse decisions are outside this starter benchmark's authorization.")

with evidence:
    st.subheader("Case-level evidence")
    results_df = pd.DataFrame(result["results"])
    selected_model = st.selectbox("Model", sorted(results_df["model"].unique()))
    selected_task = st.selectbox("Task", ["All"] + sorted(results_df["task"].unique()))
    filtered = results_df[results_df["model"] == selected_model]
    if selected_task != "All":
        filtered = filtered[filtered["task"] == selected_task]
    st.dataframe(
        filtered[["case_id", "task", "risk_tier", "score", "passed", "latency_s", "output"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download run evidence",
        data=json.dumps(result, indent=2),
        file_name=f"{result['run_id']}.json",
        mime="application/json",
    )

