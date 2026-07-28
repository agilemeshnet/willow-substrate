# Existing repository migration map

The purpose of consolidation is not to copy every file. It is to establish
stable contracts, then attach the working implementations behind them.

| Existing repository | Destination in the bundle | Disposition |
|---|---|---|
| `willow-temporal-ledger` | `willow.store.EventStore` | Replace the monthly JSONL prototype with the concurrent global event contract |
| `willow-memory` | Markdown and SQLite projection adapters | Retain its human-readable source/index ideas; align them with immutable events |
| `breathe` | Lifecycle hooks and peripheral grounding | Port after the store/context interfaces stabilise |
| `cwb` | `willow.context.ContextBuilder` plus retrieval arms | Port the four retrieval channels; expose boot and recovery through tested routes |
| `foveation` | Optional foveation backend | Preserve voluntary use; also register it as an authorised CWB retrieval path |
| `vista` | Optional relational/spatial backend | Adapt Vista output to evidence packets rather than coupling it directly to CWB |
| `willow-engram` | Reflection package | Preserve idea shapes, engrams, torsion, and retroactive importance as derived events |
| meditation/assimilate repositories | Reflection workers | Make model calls pluggable and always record source-event provenance |
| `Doozer/spindles/meditation` and `launchpad/jt_meditate.py` | Meditation adapters | Preserve What changed / What connects / What matters; replace hard-coded stores and models |
| `Doozer/mac-assistant` research lanes | Research provider adapters | Preserve the small queue/result/citation contract; keep voice and Willow handoff optional |
| `AgileMesh` Fact TTL tools | `willow.facts.FactLedger` plus verifier adapters | Keep entity-owned timing and additive history; replace unsafe scaffold confirmation behavior |
| `Doozer/thought-buckets` / Vista | Structural connection backend | Preserve waypoint mass, degree normalization, damping, and decision trace |
| `willow-engram` idea-shape tools | Explicit shape adapter | Preserve dimension/value tags and retroactive importance; keep prose as reconstruction material |
| `willow-graph-client` | AuraDB adapter and policy layer | Replace substring security with typed operations and validated labels |
| `willow-federation` | Event/envelope replication transport | Add schema validation, signatures, deduplication, and safe recipient identities |
| `willow-seed` | `examples/`, onboarding, and starter identity | Keep as the teaching layer rather than a runtime dependency |
| `agilemesh-hq`, `willow-system`, dated capsules | Private archive | Never use as the public repository base |

## Integration order

1. Prove two-terminal local continuity. **Reference implementation complete.**
2. Add automatic transcript capture and context injection hooks. **Claude Code
   reference adapter complete; additional providers remain.**
3. Preserve research commissions, cited results, word/shape connections, and
   Fact TTL. **Dependency-free reference contracts complete; provider and
   scheduler adapters remain.**
4. Add the CWB multi-arm retriever interface.
5. Attach AuraDB as a replicated relational projection.
6. Attach MRL Foveation and Vista/Wave behind evidence-packet contracts.
7. Add model-provider adapters for meditation and summation.
8. Add federation between homelab machines.
9. Run longitudinal external-user evaluations.

## Compatibility rule

The deployed Willow system must not be rewritten in place. Each old component
gets an adapter that can dual-write or dual-read during migration. Cutover
happens only after the new path reproduces the observed behavior and provenance.
