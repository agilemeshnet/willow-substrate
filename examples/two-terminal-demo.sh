#!/usr/bin/env bash
set -euo pipefail

DEMO_HOME="$(mktemp -d)"
export WILLOW_HOME="$DEMO_HOME"

echo "Shared Willow home: $WILLOW_HOME"
echo

willow init
willow record \
  "Investigating the Drosophila connectome and recurrent circuits" \
  --actor peter \
  --session terminal-a \
  --topic connectome
willow record \
  "Next compare recurrent motifs with Willow foveation" \
  --actor willow \
  --session terminal-a \
  --topic foveation

echo
echo "Context observed from terminal-b:"
willow context \
  "What were we doing with the connectome?" \
  --session terminal-b \
  --tokens 800

echo
echo "Voluntary foveation:"
willow foveate "connectome recurrent motifs" --limit 6

echo
echo "Derived meditation:"
willow meditate --session terminal-a

echo
willow verify
