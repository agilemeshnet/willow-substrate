# Security and privacy

Willow is alpha software that may contain unusually sensitive longitudinal
material. Use synthetic or non-sensitive data while evaluating the public
bundle.

## Reporting a vulnerability

Once the GitHub repository is established, use its private security-advisory
channel. Do not open a public issue containing credentials, private memories,
precise locations, transcripts, or an exploitable disclosure path.

Until a private reporting channel is published, do not send sensitive material
to project contributors. Retain the evidence locally and open a minimal public
issue stating that a private contact route is needed.

## Current security boundary

The reference core:

- stores events locally in SQLite;
- enforces append-only events with database triggers;
- verifies a global hash chain;
- treats corrections as new events;
- does not call a model or cloud service by itself;
- safe-stages research commissions by default;
- keeps model and graph providers behind adapters;
- builds Vista/Wave locally as a derived view over active, unexpired events;
- returns Vista slugs, waypoints, source event IDs, and a decision trace.

The alpha does not yet provide:

- encrypted storage or operating-system key management;
- a complete consent and revocation state machine;
- capability-enforced audience and purpose projection;
- multi-user authentication and authorization;
- signed federation envelopes;
- comprehensive disclosure auditing;
- a hardened remote API.

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
