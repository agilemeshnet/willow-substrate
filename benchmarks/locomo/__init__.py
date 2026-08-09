"""LoCoMo evidence-retrieval benchmark for willow-substrate.

LoCoMo (snap-research/locomo) provides ten substantial timestamped
conversations with gold evidence turn IDs, so this benchmark measures
objectively whether Willow placed the right memories into the retrieval
result AND into the final context packet, not whether an LLM happened
to guess correctly.

The split (raw retrieval vs final context) is deliberate: our own
testing has found cases where hybrid retrieval located the correct
event but context assembly discarded it behind recent-and-foveated
memories. Reporting both surfaces makes that class of regression
visible instead of silent.

See docs/BENCHMARK_LOCOMO.md for the methodology and how to fetch the
LoCoMo dataset (not vendored; open-licensed and expected at
benchmarks/locomo/data/ when the runner fires).
"""
