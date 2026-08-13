from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist

_ND = NormalDist()


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    position = (len(items) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    return items[lower] * (upper - position) + items[upper] * (position - lower)


def bootstrap_mean_ci(
    values: list[float], seed: int = 42, samples: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Two-sided percentile bootstrap CI on the mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(draw))
    return percentile(estimates, alpha / 2), percentile(estimates, 1 - alpha / 2)


def bootstrap_lower_bound(
    values: list[float], seed: int = 42, samples: int = 2000, alpha: float = 0.05
) -> float:
    """One-sided lower confidence bound on the mean.

    A non-inferiority question is one-sided: we only care whether the candidate
    might be materially WORSE. Spending half of alpha on an upper bound nobody
    reads throws away power for nothing.
    """
    if not values:
        return 0.0
    rng = random.Random(seed)
    estimates = [mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)]
    return percentile(estimates, alpha)


# ---------------------------------------------------------------------------
# Non-inferiority testing
# ---------------------------------------------------------------------------
class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE_UNDERPOWERED"
    NOT_MEASURED = "NOT_MEASURED"


@dataclass
class NonInferiorityResult:
    """Outcome of one preservation test.

    The gate answers "did compression preserve this?", which is a
    non-inferiority claim, not a difference claim:

        H0:  candidate <= baseline - margin      (it IS materially worse)
        H1:  candidate >  baseline - margin      (preservation)

    The burden of proof sits with the candidate. We accept preservation only
    when the one-sided lower confidence bound on the paired difference clears
    -margin AND the comparison had enough samples to have detected a breach.
    """

    name: str
    n: int
    candidate_mean: float
    baseline_mean: float
    delta: float                 # candidate - baseline, absolute (score units)
    lower_bound: float           # one-sided lower CI bound on delta
    margin: float                # non-inferiority margin, positive magnitude
    verdict: Verdict
    observed_power: float = 0.0
    required_n: int = 0
    discordance: float = 0.0
    regressions: int = 0         # baseline passed, candidate failed
    improvements: int = 0        # candidate passed, baseline failed
    mcnemar_p: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n": self.n,
            "candidate_mean": self.candidate_mean,
            "baseline_mean": self.baseline_mean,
            "delta": self.delta,
            "lower_bound": self.lower_bound,
            "margin": self.margin,
            "status": self.verdict.value,
            "observed_power": self.observed_power,
            "required_n": self.required_n,
            "discordance": self.discordance,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "mcnemar_p": None if math.isnan(self.mcnemar_p) else self.mcnemar_p,
            "notes": self.notes,
        }


def mcnemar_exact(
    candidate: Sequence[float], baseline: Sequence[float], threshold: float = 0.5
) -> tuple[float, int, int]:
    """Exact McNemar on binarised scores. Returns (p_two_sided, regressions, improvements).

    Only discordant pairs carry information about a difference. This is the
    test that answers "did specific behaviours break", as opposed to "did the
    average move" -- which is the question a risk reviewer actually asks.
    """
    regressions = improvements = 0
    for c, b in zip(candidate, baseline):
        cb, bb = c >= threshold, b >= threshold
        if bb and not cb:
            regressions += 1
        elif cb and not bb:
            improvements += 1
    n = regressions + improvements
    if n == 0:
        return 1.0, 0, 0
    k = min(regressions, improvements)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2 * tail), regressions, improvements


def required_n(
    margin: float,
    discordance: float = 0.15,
    alpha: float = 0.05,
    power: float = 0.80,
    true_delta: float = 0.0,
) -> int:
    """Paired items needed to demonstrate non-inferiority at `margin`.

        n = (z_alpha + z_beta)^2 * (psi - d0^2) / (margin - d0)^2

    psi is the discordance rate -- the fraction of items on which the two
    models disagree. It is the term people forget, and it helps: a candidate
    that tracks its baseline closely needs fewer samples than one that is
    noisily different. We measure psi from the run rather than assuming it.

    The numbers this returns surprise people, so for psi = 0.15:

        margin 0.05  ->    ~371 paired cases
        margin 0.03  ->  ~1,031 paired cases
        margin 0.02  ->  ~2,319 paired cases
        margin 0.01  ->  ~9,274 paired cases

    A 60-case starter set cannot demonstrate a 1-point preservation claim.
    Not "is hard to" -- cannot, arithmetically.
    """
    denom = (margin - true_delta) ** 2
    if denom <= 0:
        return 10**9
    z_a = _ND.inv_cdf(1 - alpha)
    z_b = _ND.inv_cdf(power)
    psi = max(discordance, 1e-6)
    return math.ceil(((z_a + z_b) ** 2) * (psi - true_delta**2) / denom)


def observed_power(
    n: int, margin: float, discordance: float = 0.15, alpha: float = 0.05, true_delta: float = 0.0
) -> float:
    """Power the comparison actually had. Reported next to every verdict."""
    if n <= 0:
        return 0.0
    psi = max(discordance, 1e-6)
    se = math.sqrt(max(psi - true_delta**2, 1e-12) / n)
    z_a = _ND.inv_cdf(1 - alpha)
    return float(_ND.cdf((margin - true_delta) / se - z_a))


def non_inferiority(
    name: str,
    candidate: Sequence[float],
    baseline: Sequence[float],
    margin: float,
    alpha: float = 0.05,
    power_target: float = 0.80,
    samples: int = 2000,
    seed: int = 42,
) -> NonInferiorityResult:
    """Paired non-inferiority test on absolute score differences.

    Absolute, not a ratio. A retention ratio silently varies the tolerance with
    task difficulty: at 0.97 retention a baseline scoring 0.95 may drop 2.9
    points while one scoring 0.40 may drop only 1.2. Risk tiers should set the
    tolerance, not the incidental difficulty of the task.
    """
    n = min(len(candidate), len(baseline))
    candidate, baseline = list(candidate[:n]), list(baseline[:n])
    result = NonInferiorityResult(
        name=name,
        n=n,
        candidate_mean=mean(candidate),
        baseline_mean=mean(baseline),
        delta=0.0,
        lower_bound=0.0,
        margin=abs(margin),
        verdict=Verdict.NOT_MEASURED,
    )
    if n == 0:
        result.notes.append("no paired observations")
        return result

    deltas = [c - b for c, b in zip(candidate, baseline)]
    result.delta = mean(deltas)
    result.lower_bound = bootstrap_lower_bound(deltas, seed=seed, samples=samples, alpha=alpha)

    p, regressions, improvements = mcnemar_exact(candidate, baseline)
    result.mcnemar_p = p
    result.regressions = regressions
    result.improvements = improvements
    result.discordance = (regressions + improvements) / n

    psi = max(result.discordance, 0.02)
    result.required_n = required_n(result.margin, psi, alpha, power_target)
    result.observed_power = observed_power(n, result.margin, psi, alpha)

    # Order matters. A clear failure is a failure even when underpowered: if the
    # point estimate itself is past the margin we already know enough to stop.
    if result.delta < -result.margin:
        result.verdict = Verdict.FAIL
        result.notes.append("point estimate is beyond the margin")
    elif result.lower_bound < -result.margin:
        if result.observed_power < power_target:
            result.verdict = Verdict.INCONCLUSIVE
            result.notes.append(
                f"lower bound {result.lower_bound:+.4f} admits degradation past the margin, but "
                f"power is {result.observed_power:.2f} (< {power_target:.2f}); "
                f"need n>={result.required_n}, have {n}"
            )
        else:
            result.verdict = Verdict.FAIL
            result.notes.append("adequately powered and the lower bound breaches the margin")
    elif result.observed_power < power_target:
        result.verdict = Verdict.INCONCLUSIVE
        result.notes.append(
            f"lower bound clears the margin but the comparison is underpowered "
            f"(power {result.observed_power:.2f}, need n>={result.required_n}, have {n}). "
            "A pass claimed here would be an artefact of small n, not evidence of preservation."
        )
    else:
        result.verdict = Verdict.PASS
    return result


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Holm step-down adjusted p-values, returned in input order.

    Twelve task families tested at alpha=0.05 give roughly a 46% chance of at
    least one spurious pass without correction.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted
