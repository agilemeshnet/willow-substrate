#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if command -v willow >/dev/null 2>&1; then
  WILLOW=(willow)
else
  export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  WILLOW=(python3 -m willow.cli)
fi

DEMO_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/willow-five-process.XXXXXX")"
export WILLOW_HOME="$DEMO_ROOT/store"
mkdir -p "$WILLOW_HOME"

for number in 1 2 3 4; do
  transcript="$DEMO_ROOT/terminal-$number.jsonl"
  printf '%s\n' \
    "{\"type\":\"user\",\"uuid\":\"user-$number\",\"message\":{\"role\":\"user\",\"content\":\"Terminal $number is comparing Drosophila connectome recurrent motifs with foveation\"}}" \
    "{\"type\":\"assistant\",\"uuid\":\"assistant-$number\",\"message\":{\"role\":\"assistant\",\"content\":\"Terminal $number has preserved its current connectome finding.\"}}" \
    > "$transcript"

  printf '%s\n' \
    "{\"session_id\":\"terminal-$number\",\"transcript_path\":\"$transcript\"}" \
    | "${WILLOW[@]}" hook stop &
done

wait

printf '%s\n' \
  '{"session_id":"terminal-5","prompt":"What are the other terminals doing with the connectome?"}' \
  | "${WILLOW[@]}" hook prompt

"${WILLOW[@]}" verify
printf 'Demo store retained at %s\n' "$WILLOW_HOME"
