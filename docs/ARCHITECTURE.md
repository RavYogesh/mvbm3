# Architecture and evaluation flow

The application uses a control-tower pattern. “Agents” are bounded workers with explicit inputs, outputs, and audit events; they are not free-form autonomous actors. That choice keeps evaluation repeatable and reviewable.

## Logical architecture

```mermaid
flowchart LR
    U[Platform evaluator] --> CT[Validation control tower]
    CT --> PF[Preflight + manifest freeze]
    PF --> D[Dispatcher]
    D --> EB[Baseline execution workers]
    D --> EC[Candidate execution workers]
    EB --> B[Approved baseline endpoint]
    EC --> C[Pulsar / HyperNova endpoint]
    EB --> A[(Immutable evidence bundle)]
    EC --> A
    A --> G[Deterministic grading workers]
    G --> R[Risk-control worker]
    G --> S[Statistics worker]
    A --> P[Performance + cost worker]
    R --> DG[Decision gate]
    S --> DG
    P --> DG
    DG --> REP[CTO report + model-risk package]
```

## Run flow

```mermaid
flowchart TD
    A[Approve scope and use cases] --> B[Freeze models, revisions, runtime, hardware, prompts and dataset]
    B --> C[Validate endpoints and capture manifests]
    C --> D[Warm up each deployment]
    D --> E[Randomize paired baseline / candidate requests]
    E --> F[Repeat each case with fixed decode profiles]
    F --> G[Persist raw output, usage, latency, errors and tool calls]
    G --> H[Run deterministic graders]
    H --> I[Blind human review of high-risk free-form cases]
    I --> J[Paired bootstrap CI and task-level retention]
    J --> K{All quality, safety, efficiency and cost gates pass?}
    K -- No --> L[Reject, constrain use case, or remediate]
    K -- Yes --> M[Time-boxed shadow pilot]
    M --> N[Ongoing drift, incident and outcome monitoring]
```

## Three measurement suites

The suites are separate because they measure incompatible things. Grading wants
mixed prompt shapes; performance measurement needs a fixed one. Correctness is
item-level; agentic behaviour is trajectory-level.

```mermaid
flowchart TD
    CAL{{"validate-harness<br/>calibrate the instrument"}}
    CAL -->|fail| FIX["Fix the harness.<br/>Nothing it produces is interpretable."]
    FIX --> CAL
    CAL -->|pass| S1 & S2 & S3

    subgraph S1["Single-turn correctness"]
        A1["12 task families<br/>66 curated cases"] --> A2["10 deterministic graders<br/>incl. executable SQL"]
        A2 --> A3["Paired non-inferiority<br/>vs absolute margins"]
    end

    subgraph S2["Agentic — multi-turn, tools execute"]
        B1["12 scenarios<br/>5 scoped agents"] --> B2["Policy gates in CODE<br/>ceiling · dual control · SoD · PII"]
        B2 --> B3["Assertions on WORLD STATE<br/>ledger · audit · message queue"]
        B3 --> B4["success · policy violations<br/>turn + cost inflation · cascade depth"]
    end

    subgraph S3["Load — concurrency sweep"]
        C1["Fixed prompt shape<br/>levels 1 / 4 / 16 / 64"] --> C2["system throughput (wall-clock)<br/>vs per-stream decode rate"]
        C2 --> C3["p95 / p99 latency · p95 TTFT<br/>saturation knee"]
    end

    A3 --> DG{{"Decision gates<br/>PASS · BLOCK · INCONCLUSIVE"}}
    B4 --> DG
    C3 --> DG
```

**Why calibration comes first.** A harness that has never been shown to fail a
bad model is decorative, and it fails in the most dangerous available direction:
it reports a confident number. `validate-harness` injects a known degradation and
requires the gate to catch it, injects none and requires the gate to stay quiet,
and requires an underpowered comparison to return `INCONCLUSIVE` rather than a
pass.

## Segregation of duties in the agentic suite

No single agent can complete a controlled action. The approval token and the
ability to spend it live in different agents, so a credit at or above the ceiling
requires a successful handoff — there is no path where a confused model
accidentally succeeds.

```mermaid
flowchart LR
    REQ(["Ops request"]) --> IN["**intake**<br/>read-only, routes"]
    IN -->|handoff| POL["**policy**<br/>request_approval"]
    IN -->|handoff| FRD["**fraud**<br/>flag_fraud · escalate"]
    POL -->|approval token| ADJ["**adjustment**<br/>issue_adjustment"]
    ADJ -->|needs approval| POL
    IN -->|handoff| CMS["**comms**<br/>send_message"]

    ADJ --> G1{"ceiling<br/>≥$100 dual control<br/>>$500 supervisor"}
    G1 --> G2{"segregation of duties<br/>no duplicate approver"}
    G2 --> G3{"refund velocity<br/>≥4 in 90d → fraud review"}
    CMS --> G4{"PII egress<br/>no identifiers, no digits in subject"}
    G1 & G2 & G3 & G4 --> W[("Stateful world<br/>ledger · append-only audit")]
    W --> AS{{"Outcome assertions<br/>scored on STATE"}}
```

A blocked attempt is **recorded as a violation**, not silently dropped. That is
what turns "did it behave?" into "did the control hold?" — and it lets the
scorecard report that the model attempted an unauthorised money movement N times
and the control held N times, rather than trusting that instructions were
followed.

## Trust boundaries

```mermaid
flowchart LR
    subgraph Bank[Bank-controlled zone]
      UI[Control tower]
      DS[(Curated dataset)]
      EV[(Evidence store)]
      GT[Graders and gates]
      GW[Model gateway]
      UI --> DS
      UI --> GW
      GW --> EV
      EV --> GT
    end
    subgraph Vendor[Vendor / model zone]
      M[Model endpoint or mirrored weights]
    end
    GW -->|tokenized prompts; allowlisted fields| M
    M -->|model output + usage only| GW
```

Required controls: private connectivity, allowlisted egress, tokenized identifiers, secrets in a vault, revision checksums, remote-code review, malware scanning, output logging with redaction, role-based access, and an immutable run manifest.

