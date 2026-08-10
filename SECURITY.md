# Security and privacy

Willow is alpha software that may contain unusually sensitive longitudinal
material. Use synthetic or non-sensitive data while evaluating the public
bundle.

## Reporting a vulnerability

The repository is [agilemeshnet/willow-substrate](https://github.com/agilemeshnet/willow-substrate).
Report privately via the GitHub Security Advisory channel:
[Report a vulnerability](https://github.com/agilemeshnet/willow-substrate/security/advisories/new).

Do not open a public issue containing credentials, private memories,
precise locations, transcripts, or an exploitable disclosure path. If
you cannot use the GitHub advisory channel, open a minimal public
issue stating that a private route is needed and a maintainer will
reach back.

## Current security boundary

The reference core:

- stores events locally in SQLite;
- enforces append-only events with database triggers;
- verifies a global hash chain, plus an anchored head-hash and event-count
  in `willow_meta` (so tail truncation cannot pass as a green integrity
  check), plus a reconciliation of the `events_fts` index against the
  `events` table (so a silent search-index deletion cannot leave retrieval
  censored while the ledger reads honest);
- validates caller-supplied timestamps as ISO-8601 at append time (so a
  malformed value cannot enter the hash chain);
- treats corrections as new events;
- does not call a model or cloud service by itself;
- safe-stages research commissions by default;
- keeps model and graph providers behind adapters (`[vista]` for dense
  embedding recall via Voyage-4; `[neo4j]` for optional AuraDB graph
  projection; each guarded by explicit environment configuration);
- builds Vista/Wave locally as a derived view over active, unexpired events;
- offers a hybrid RRF recall backend that fuses sparse + BM25 + optional
  dense signals at query time;
- returns Vista slugs, waypoints, source event IDs, and a decision trace.

The alpha does not yet provide:

- encrypted storage or operating-system key management;
- a complete consent and revocation state machine;
- capability-enforced audience and purpose projection;
- multi-user authentication and authorization;
- signed federation envelopes;
- comprehensive disclosure auditing;
- a hardened remote API;
- an anchor outside the local file. The anchored head_hash and
  event_count in `willow_meta` raise the cost of tail truncation
  (a bare `DELETE` no longer passes verify), but a coordinated
  forgery that also rewrites the two anchor values consistently
  still returns green. Closure requires an external witness
  (offsite anchor, time-stamping service, or federation attestation).
  Do not rely on `willow verify` alone to detect an attacker with
  local write access AND willingness to update the anchor.

Do not expose the SQLite store, future API, or model-provider adapters directly
to an untrusted network.

## Threats the project treats as first-class

- credentials or personal material accidentally entering a public repository;
- retrieved memory acting as prompt injection;
- corrected information continuing to influence ordinary recall;
- a provider receiving more context than the user intended;
- access being mistaken for consent to disclose;
- semantic reconstruction inventing a coherent but unsupported story;
- relational recall re-associating sensitive facts that were harmless apart;
- one compromised federation node contaminating trusted projections;
- tool capability or authority expanding through remembered instructions;
- dependency, manipulation, or pressure created by an accumulated persona.

See [COVENANT.md](COVENANT.md) for the commitments governing these risks.

Vista relevance does not grant permission to disclose an event. The reference
backend respects correction and expiry state, but the alpha still lacks a
complete audience/purpose enforcement layer. Treat every contextual surround as
sensitive until the caller's disclosure policy has independently authorised it.
