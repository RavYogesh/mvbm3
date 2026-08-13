"""Multiverse model validation harness.

Suites:
    benchmark.cli demo|run       single-turn correctness across 12 task families
    benchmark.cli agentic        multi-turn, tool-executing trajectories
    benchmark.cli loadtest       concurrency sweep for throughput and tail latency
    benchmark.cli validate-harness   instrument calibration -- run this first

Quality is tested for NON-INFERIORITY against pre-registered absolute margins,
and the harness returns three verdicts, not two: an underpowered comparison
reports INCONCLUSIVE rather than a pass it cannot defend.
"""

__version__ = "2.0.0"
