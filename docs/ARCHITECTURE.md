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

