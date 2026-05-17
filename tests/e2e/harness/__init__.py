"""End-to-end test harness — drives the live Dendrite backend over REST + LSL.

Adapted from the proven `dendrite-docs/paper/benchmarks/` system-eval scaffold,
trimmed to the swarm + MOABB paths and given pass/fail assertions. The harness
resolves a dataset (local in-house via env var, else MOABB) and replays it
through the real pipeline.

Run standalone:  uv run python -m tests.e2e.harness.runner
"""
