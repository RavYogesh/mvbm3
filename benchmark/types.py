from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    task: str
    risk_tier: str
    prompt: str
    context: str = ""
    grader: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def messages(self) -> list[dict[str, str]]:
        system = (
            "You are being evaluated in a controlled US-bank benchmark. "
            "Use only the supplied context when the task is grounded. Follow the requested output format exactly."
        )
        user = self.prompt if not self.context else f"CONTEXT:\n{self.context}\n\nTASK:\n{self.prompt}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    baseline: str | None = None
    timeout_s: int = 180
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    text: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class CaseResult:
    run_id: str
    model: str
    case_id: str
    task: str
    risk_tier: str
    repetition: int
    output: str
    score: float
    passed: bool
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    grade_details: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

