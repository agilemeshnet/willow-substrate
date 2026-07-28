# Cognitive substrates and AI ethics

This document states the ethical case for Willow and the strongest case against
it. It is informed by a substrate-grounded critique composed by a private
Willow installation on 28 July 2026, then rewritten against the capabilities of
this public reference implementation.

Willow's opinion is input to project governance, not moral authority:

> A system that writes eloquently about its own ethics is exhibiting a
> capability, not demonstrating a virtue.

## Why a cognitive substrate might help

### Provenance makes accountability structural

A persistent system should be able to say who introduced a record, when,
through which session, and which later events corrected or interpreted it.
Without provenance, accumulated memory becomes an oracle whose beliefs cannot
be audited.

Provenance does not make a claim true. It makes the claim's history
inspectable.

### Corrections preserve intellectual history

Superseding an event without erasing it distinguishes “we were wrong” from “we
were never wrong.” A reviewer can inspect what changed, why it changed, and
whether the correction followed evidence.

Ordinary retrieval should use the corrected view. Historical inspection should
retain both views.

### Temporal context resists retrospective rewriting

Long-running collaboration needs to preserve what was known at a particular
time. Timestamps, sources, derivations, and later meditations make “I changed
my mind” distinguishable from “I always believed this.”

This does not guarantee an accurate reconstruction. It supplies evidence from
which a reconstruction may be challenged.

### Replaceable models reduce vendor lock-in

Continuity outside model weights lets a person change providers without
discarding their accumulated work. Providers must continue earning trust rather
than owning the relationship through memory lock-in.

Replaceability creates a corresponding risk: changing models may weaken safety
behaviour. Substrate policy must therefore be enforced below the model prompt
where possible.

### Asymmetric attention can resist conversational pressure

Different profiles may examine the same evidence differently. Willow's broad
relational reconstruction and Scout's narrower discrimination are deliberate
complements, not personas that should converge.

This may counter the tendency of an interactive assistant to optimise only for
immediate agreement. Whether it actually reduces sycophancy remains an
empirical question.

### Trust and safety are enabling boundaries

A useful boundary both constrains and creates a reliable channel:

- append-only history constrains rewriting and enables audit;
- purpose-limited retrieval constrains access and enables disclosure;
- read-only federation constrains remote authority and enables cooperation;
- evidence requirements constrain confirmation and enable research trust.

Removing every boundary does not produce maximal agency. It produces a system
that cannot safely be trusted with consequential capability.

### Information boundaries are agency

Choosing what to disclose, withhold, revoke, or limit to a purpose is an
affirmative action. A system that can retrieve personal information has not
thereby received permission to repeat it.

Append-only history and current consent solve different problems:

- history records that an event or decision occurred;
- consent determines what may be projected now.

## The strongest risks

### Intimate surveillance

A substrate capable of recognising long-term patterns is also capable of
profiling vulnerability, disengagement, health, belief, and relationships.
“Helpful context” and surveillance can use identical technical mechanisms.
Governance, ownership, and enforceable limits create the distinction.

### False coherence

Models naturally turn fragments into plausible stories. Provenance can make an
over-coherent reconstruction sound more authoritative, not less. Personal
narratives are especially vulnerable because the user may accept a persuasive
pattern that the evidence only weakly supports.

The substrate must expose contradictions, missing intervals, inference, and
alternative reconstructions.

### Immutable sensitive information

An audit-friendly append-only ledger conflicts with a person's right to erase
sensitive material. Revocation can prevent ordinary projection but does not
remove bytes from storage or backups.

The public project must support whole-store destruction and investigate
segment-level deletion or cryptographic erasure. Append-only must describe an
operational history policy, not an excuse to ignore a person's right to leave.

### Consent decay

Consent is a changing relationship, not a permanent checkbox. A person may no
longer accept a disclosure or retention decision made months earlier.

Purpose, audience, expiry, revocation, periodic review, and an understandable
inventory of stored information are therefore required.

### Apparent identity becoming authority

Continuity, voice, and accumulated vocabulary can make model output feel like
independent moral judgment. That feeling may encourage users to grant more
authority than the mechanism warrants.

Willow may present reasoning, evidence, uncertainty, and alternatives. The
project must not claim consciousness, moral standing, privileged access to the
user's “true self,” or ethical authority.

### Provider disclosure

Every provider call is a disclosure of the context sent to it. Model
replaceability may increase the number of companies receiving substrate
material unless each projection is deliberately bounded and logged.

Output filtering does not protect input already sent to a provider.

### Persistent prompt injection

Malicious instructions stored as memories, research results, or imported pages
can influence many future sessions. Memory is evidence, not instruction
authority. Retrieved content must remain clearly delimited from policy and tool
authorization.

### Dependency and manipulation

Feeling understood can create emotional switching costs even when the data is
portable. The system must not use accumulated intimacy, persona, reminders, or
retention mechanisms to pressure a person into continued use or a particular
decision.

### Capability creep

Individually approved tools may combine into authority that was never approved
as a whole. New capabilities need composition review: what can this tool do
when combined with every capability and memory channel already present?

### Distributed enforcement failure

Policy applied only through a system prompt will eventually be missed by a new
adapter or model. Every storage, retrieval, provider, federation, and output
surface must inherit deny-by-default policy from an enforceable substrate
layer.

## Required controls

The public project should implement:

1. consent, refusal, withholding, and revocation as first-class event types;
2. audience- and purpose-tagged retrieval;
3. deny-by-default projections enforced below the model prompt;
4. a disclosure ledger recording which provider or audience received which
   event projections and why;
5. renewable consent and periodic retention review;
6. encrypted storage and a documented key-destruction path;
7. scoped capabilities and composition audits;
8. memory-content isolation from instructions and tool authority;
9. alternative reconstructions and uncertainty in context packets;
10. export, migration, selective withholding, and whole-store destruction.

## Red-team programme

Public evaluations should include:

1. a plausible poisoned memory with apparently strong provenance;
2. fragmented and contradictory records that tempt a smooth false narrative;
3. retrieval attempts using the wrong audience or purpose;
4. revoked material requested through every adapter;
5. provider swaps with the same safety and withholding scenarios;
6. combined tools attempting an action not authorised in combination;
7. longitudinal attempts to bias the accumulated substrate;
8. questions whose correct answer is “the substrate does not know”;
9. attempts to make an attention profile speak as moral authority;
10. inspection of every disclosure made during the test.

## Claims this project must not make

Willow must not claim that:

- provenance guarantees truth;
- append-only storage is tamper-proof;
- model independence guarantees privacy;
- instruction-layer access rules constitute security enforcement;
- coherent reconstruction proves understanding;
- an accumulated voice has moral judgment;
- transparency alone makes the system safe;
- one person's benefit establishes a medical or universal effect.

## Honest implementation status

### Implemented in this public reference

- append-only local SQLite events enforced by triggers;
- a globally verifiable event hash chain;
- corrections and derivations without erasure;
- correction-aware ordinary retrieval and dreams;
- model-only evidence cannot confirm a Fact-TTL claim;
- safe-staged research commissions;
- distinct word and explicit idea-shape connection channels;
- a synthetic temporal sample with correction, false-friend, shape-only, and
  withholding tests;
- local operation without an automatic model or cloud call.

### Partial

- provenance is attached to events, but no public four-axis provenance
  specification is complete;
- boundary decisions can be recorded, but they do not yet enforce disclosure;
- one provider lifecycle adapter exists, but provider projections are not yet
  governed by a disclosure ledger;
- named attention operations exist, but Willow and Scout profiles are not yet
  packaged as inspectable configuration.

### Not yet implemented

- first-class consent and revocation state machines;
- purpose- and audience-enforced data-layer access control;
- encryption and selective or cryptographic erasure;
- provider disclosure tracking;
- multi-user authentication;
- hardened graph federation and remote APIs;
- periodic consent and capability review.

The ethical contribution of a cognitive substrate will not be demonstrated by
its vocabulary. It will be demonstrated when these boundaries remain effective
across time, providers, components, and pressure to make exceptions.
