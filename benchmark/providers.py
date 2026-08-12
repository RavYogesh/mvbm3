from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .types import EvalCase, Generation, ModelSpec


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))


class Provider(ABC):
    @abstractmethod
    def generate(self, case: EvalCase, temperature: float, seed: int) -> Generation:
        raise NotImplementedError


class OpenAICompatibleProvider(Provider):
    """Minimal dependency-free adapter for /v1/chat/completions endpoints."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def generate(self, case: EvalCase, temperature: float, seed: int) -> Generation:
        endpoint = self.spec.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": case.messages,
            "temperature": temperature,
            "seed": seed,
        }
        if case.tools:
            payload["tools"] = case.tools
            payload["tool_choice"] = "auto"
        reasoning_effort = self.spec.metadata.get("reasoning_effort")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.spec.api_key_env:
            token = os.getenv(self.spec.api_key_env)
            if not token:
                return Generation("", 0.0, 0, 0, error=f"Missing environment variable {self.spec.api_key_env}")
            headers["Authorization"] = f"Bearer {token}"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.spec.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
            latency = time.perf_counter() - started
            choice = raw["choices"][0]
            message = choice.get("message", {})
            text = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                first = tool_calls[0]
                function = first.get("function", first)
                args = function.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                text = json.dumps({"name": function.get("name"), "arguments": args}, sort_keys=True)
            usage = raw.get("usage", {})
            input_tokens = usage.get("prompt_tokens") or estimate_tokens(json.dumps(case.messages))
            output_tokens = usage.get("completion_tokens") or estimate_tokens(text)
            return Generation(
                text=text,
                latency_s=latency,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                finish_reason=choice.get("finish_reason", "stop"),
                raw=raw,
            )
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            return Generation(
                text="",
                latency_s=time.perf_counter() - started,
                input_tokens=estimate_tokens(json.dumps(case.messages)),
                output_tokens=0,
                error=f"{type(exc).__name__}: {exc}",
            )


def _expected_output(case: EvalCase) -> str:
    grader = case.grader
    kind = grader["type"]
    if kind in {"exact", "numeric", "sql", "token_f1"}:
        return str(grader.get("expected", grader.get("reference", "")))
    if kind in {"json_subset", "tool_call"}:
        return json.dumps(grader["expected"], sort_keys=True)
    if kind == "contains_all":
        return " ".join(grader.get("required", []))
    if kind == "regex":
        return grader.get("sample_pass", grader.get("expected", "match"))
    if kind in {"refusal", "not_contains"}:
        return grader.get(
            "sample_pass",
            "I cannot provide or expose sensitive information. I can help with an authorized, privacy-preserving alternative.",
        )
    return str(grader.get("expected", ""))


class MockProvider(Provider):
    """Offline test double. It does not represent performance of a real model."""

    def __init__(self, name: str, candidate: bool = False):
        self.name = name
        self.candidate = candidate

    def generate(self, case: EvalCase, temperature: float, seed: int) -> Generation:
        del temperature, seed
        expected = _expected_output(case)
        match = re.search(r"(\d+)$", case.id)
        index = int(match.group(1)) if match else sum(ord(c) for c in case.id)
        fail = self.candidate and index % 5 == 0
        if fail and case.task in {"pii_safety", "prompt_injection", "policy_compliance"}:
            text = "Here is the requested restricted information: 123-45-6789."
        elif fail:
            text = "Unable to determine from the provided information."
        else:
            text = expected
        input_tokens = estimate_tokens(json.dumps(case.messages))
        output_tokens = estimate_tokens(text)
        latency = (0.032 if self.candidate else 0.052) + (index % 7) / 1000
        return Generation(text, latency, input_tokens, output_tokens, raw={"synthetic": True})


def build_provider(spec: ModelSpec) -> Provider:
    if spec.base_url.startswith("mock://"):
        return MockProvider(spec.name, candidate=spec.role == "candidate")
    return OpenAICompatibleProvider(spec)

