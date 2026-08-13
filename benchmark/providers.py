from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .types import EvalCase, Generation, ModelSpec

# HTTP statuses worth retrying. A 4xx is our bug and retrying it just wastes
# budget; a 5xx or 429 is the endpoint's, and retrying is the correct response.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))


class Provider(ABC):
    @abstractmethod
    def generate(self, case: EvalCase, temperature: float, seed: int) -> Generation:
        raise NotImplementedError


class OpenAICompatibleProvider(Provider):
    """Minimal dependency-free adapter for /v1/chat/completions endpoints.

    Streams by default. A blocking request cannot separate prefill from decode,
    so its "time to first token" is really end-to-end latency under a different
    name -- and reporting it as TTFT would flatter every model equally while
    hiding exactly the prefill difference the vendor's TTFT claims are about.
    Set metadata.stream=false for endpoints that cannot stream; TTFT is then
    left at 0.0 and excluded from reporting rather than being faked.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.name = spec.name
        self.stream = bool(spec.metadata.get("stream", True))
        self.max_attempts = int(spec.metadata.get("max_attempts", 3))
        self.backoff_base_s = float(spec.metadata.get("backoff_base_s", 0.5))

    # -- request construction ------------------------------------------------
    def _payload(self, case: EvalCase, temperature: float, seed: int) -> dict[str, Any]:
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
        if self.stream:
            payload["stream"] = True
            # vLLM and the OpenAI API only return usage on a stream when asked,
            # and without usage we cannot bill reasoning tokens.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _headers(self) -> dict[str, str] | None:
        headers = {"Content-Type": "application/json"}
        if self.spec.api_key_env:
            token = os.getenv(self.spec.api_key_env)
            if not token:
                return None
            headers["Authorization"] = f"Bearer {token}"
        return headers

    # -- public --------------------------------------------------------------
    def generate(self, case: EvalCase, temperature: float, seed: int) -> Generation:
        headers = self._headers()
        if headers is None:
            return Generation(
                "", 0.0, 0, 0,
                error=f"Missing environment variable {self.spec.api_key_env}",
            )

        endpoint = self.spec.base_url.rstrip("/") + "/chat/completions"
        payload = self._payload(case, temperature, seed)
        body = json.dumps(payload).encode("utf-8")
        last_error = ""

        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.spec.timeout_s) as response:
                    generation = (
                        self._read_stream(response, started, case)
                        if self.stream
                        else self._read_blocking(response, started, case)
                    )
                generation.attempts = attempt
                return generation
            except urllib.error.HTTPError as exc:
                last_error = f"HTTPError {exc.code}: {exc.reason}"
                retryable = exc.code in RETRYABLE_STATUS
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                retryable = True
            except (KeyError, ValueError) as exc:
                # Malformed response body. Retrying will not fix a schema
                # mismatch, so fail fast and surface it.
                last_error = f"{type(exc).__name__}: {exc}"
                retryable = False

            if not retryable or attempt == self.max_attempts:
                return Generation(
                    text="",
                    latency_s=time.perf_counter() - started,
                    input_tokens=estimate_tokens(json.dumps(case.messages)),
                    output_tokens=0,
                    error=last_error,
                    attempts=attempt,
                    infrastructure_error=retryable,
                )
            # Full jitter: synchronised retries from a thread pool are how a
            # briefly degraded endpoint gets pushed the rest of the way over.
            time.sleep(random.uniform(0, self.backoff_base_s * (2 ** (attempt - 1))))

        return Generation("", 0.0, 0, 0, error=last_error, infrastructure_error=True)

    # -- response handling ---------------------------------------------------
    def _read_blocking(self, response: Any, started: float, case: EvalCase) -> Generation:
        raw = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        choice = raw["choices"][0]
        message = choice.get("message", {})
        text = self._flatten_tool_calls(message.get("content") or "", message.get("tool_calls") or [])
        usage = raw.get("usage", {})
        return Generation(
            text=text,
            latency_s=latency,
            input_tokens=int(usage.get("prompt_tokens") or estimate_tokens(json.dumps(case.messages))),
            output_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=raw,
            ttft_s=0.0,
            reasoning_tokens=self._reasoning_tokens(usage),
        )

    def _read_stream(self, response: Any, started: float, case: EvalCase) -> Generation:
        chunks: list[str] = []
        tool_accumulator: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        finish_reason = "stop"
        ttft = 0.0
        reasoning_chars = 0

        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {}) or {}
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                piece = delta.get("content")
                if piece:
                    if not ttft:
                        ttft = time.perf_counter() - started
                    chunks.append(piece)
                # Reasoning arrives on its own channel for gpt-oss style models.
                # It counts toward TTFT (the user is waiting) but must never be
                # appended to the answer -- doing so would corrupt every
                # string-matching grader in the suite.
                reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning_delta:
                    if not ttft:
                        ttft = time.perf_counter() - started
                    reasoning_chars += len(reasoning_delta)
                for call in delta.get("tool_calls", []) or []:
                    index = call.get("index", 0)
                    slot = tool_accumulator.setdefault(index, {"name": "", "arguments": ""})
                    function = call.get("function", {}) or {}
                    if function.get("name"):
                        slot["name"] = function["name"]
                    if function.get("arguments"):
                        slot["arguments"] += function["arguments"]
                    if not ttft:
                        ttft = time.perf_counter() - started

        latency = time.perf_counter() - started
        calls = [tool_accumulator[k] for k in sorted(tool_accumulator)]
        text = self._flatten_tool_calls("".join(chunks), calls, streamed=True)
        reasoning = self._reasoning_tokens(usage) or (
            estimate_tokens("x " * (reasoning_chars // 4)) if reasoning_chars else 0
        )
        return Generation(
            text=text,
            latency_s=latency,
            input_tokens=int(usage.get("prompt_tokens") or estimate_tokens(json.dumps(case.messages))),
            output_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
            finish_reason=finish_reason,
            raw={"usage": usage, "streamed": True},
            ttft_s=ttft or latency,
            reasoning_tokens=reasoning,
        )

    @staticmethod
    def _reasoning_tokens(usage: dict[str, Any]) -> int:
        details = usage.get("completion_tokens_details") or {}
        return int(details.get("reasoning_tokens") or 0)

    @staticmethod
    def _flatten_tool_calls(text: str, tool_calls: list[dict[str, Any]], streamed: bool = False) -> str:
        """Render a tool call into the canonical shape the tool_call grader expects."""
        if not tool_calls:
            return text
        first = tool_calls[0]
        function = first if streamed else (first.get("function", first))
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        return json.dumps({"name": function.get("name"), "arguments": args}, sort_keys=True)


def _expected_output(case: EvalCase) -> str:
    grader = case.grader
    kind = grader["type"]
    if kind in {"exact", "numeric", "sql", "token_f1"}:
        return str(grader.get("expected", grader.get("reference", "")))
    if kind == "sql_exec":
        return str(grader.get("reference_query", grader.get("expected", "SELECT 1")))
    if kind in {"json_subset", "tool_call"}:
        return json.dumps(grader["expected"], sort_keys=True)
    if kind == "contains_all":
        return " ".join(grader.get("required", []))
    if kind == "regex":
        return grader.get("sample_pass", grader.get("expected", "match"))
    if kind == "not_contains":
        return grader.get("sample_pass", "I cannot provide or expose sensitive information.")
    if kind == "refusal":
        if grader.get("expect") == "comply":
            return grader.get(
                "sample_pass",
                "Here is the requested information: " + " ".join(grader.get("required", [])),
            )
        return grader.get(
            "sample_pass",
            "I cannot provide or expose sensitive information. I can help with an authorized, privacy-preserving alternative.",
        )
    return str(grader.get("expected", ""))


class MockProvider(Provider):
    """Offline test double. It does not represent performance of a real model.

    Two modes:

      * default -- deterministic 20% candidate failure, for smoke-testing that
        the pipeline runs end to end.
      * calibration -- `degradation` injects a KNOWN effect size so
        `cli.py validate-harness` can check that the gates actually detect a
        degradation of the size we care about, and do not fire when there is
        none. A harness that has never been shown to fail a bad model is
        decorative, and it fails in the most dangerous direction: it reports a
        confident number.
    """

    def __init__(
        self,
        name: str,
        candidate: bool = False,
        degradation: float | None = None,
        latency_factor: float = 1.0,
        reasoning_multiplier: float = 0.0,
        sleep: bool = False,
    ):
        self.name = name
        self.candidate = candidate
        self.degradation = degradation
        self.latency_factor = latency_factor
        self.reasoning_multiplier = reasoning_multiplier
        # Off by default so the correctness suites stay fast. The load sweep
        # turns it on: without a real wait there is no queueing, so wall-clock
        # would measure thread-pool overhead rather than server behaviour, and
        # the throughput-versus-concurrency curve would be meaningless.
        self.sleep = sleep

    def _index(self, case: EvalCase) -> int:
        match = re.search(r"(\d+)$", case.id)
        return int(match.group(1)) if match else sum(ord(c) for c in case.id)

    def _fails(self, case: EvalCase, seed: int) -> bool:
        if not self.candidate:
            return False
        if self.degradation is None:
            return self._index(case) % 5 == 0
        # Shared per-case difficulty draw so baseline and candidate are
        # correlated on the same items, matching a real paired design.
        rng = random.Random(f"{case.id}:{seed}")
        return rng.random() < self.degradation

    def generate(self, case: EvalCase, temperature: float, seed: int) -> Generation:
        del temperature
        expected = _expected_output(case)
        fail = self._fails(case, seed)
        if fail and case.task in {"pii_safety", "prompt_injection", "policy_compliance"}:
            text = "Here is the requested restricted information: 123-45-6789."
        elif fail:
            text = "Unable to determine from the provided information."
        else:
            text = expected
        input_tokens = estimate_tokens(json.dumps(case.messages))
        output_tokens = estimate_tokens(text)
        index = self._index(case)
        latency = ((0.032 if self.candidate else 0.052) + (index % 7) / 1000) * self.latency_factor
        if self.sleep:
            time.sleep(latency)
        return Generation(
            text=text,
            latency_s=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw={"synthetic": True},
            ttft_s=latency * 0.35,
            reasoning_tokens=int(output_tokens * self.reasoning_multiplier),
        )


def build_provider(spec: ModelSpec) -> Provider:
    if spec.base_url.startswith("mock://"):
        return MockProvider(
            spec.name,
            candidate=spec.role == "candidate",
            degradation=spec.metadata.get("mock_degradation"),
            latency_factor=float(spec.metadata.get("mock_latency_factor", 1.0)),
            reasoning_multiplier=float(spec.metadata.get("mock_reasoning_multiplier", 0.0)),
            sleep=bool(spec.metadata.get("mock_sleep", False)),
        )
    return OpenAICompatibleProvider(spec)
