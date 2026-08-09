"""Recall benchmarks for willow-substrate backends.

Closes ChatGPT audit finding #4 ('no retrieval benchmark'). Runs a fixed set
of ground-truth-labelled queries against the same corpus through each
available backend and reports recall@k, mean reciprocal rank, and per-query
wall-clock latency.

Entry point: `python -m benchmarks.recall`.
"""
