from __future__ import annotations

import math
import random
from collections.abc import Iterable


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


def bootstrap_mean_ci(values: list[float], seed: int = 42, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(draw))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)

