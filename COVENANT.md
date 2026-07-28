# The Willow public covenant

Willow is cognitive infrastructure for long-running human-AI collaboration.
These commitments are release constraints, not branding language. A feature
that conflicts with them must not ship silently.

## 1. The person owns the substrate

The person owns their event store, projections, configuration, and accumulated
history. A temporary model does not acquire ownership merely because it can
read or speak through that history.

Willow will provide practical export, backup, migration, and whole-store
destruction. Append-only operation must never become a justification for
lock-in.

## 2. Information boundaries are agency

Choosing what to reveal or withhold is an affirmative exercise of agency.
Access to an event is not permission to repeat it.

The public design will represent:

- consent and refusal;
- audience and purpose;
- withholding and selective disclosure;
- expiry and later revocation;
- disclosure decisions and their audit trail.

Append-only integrity preserves that a decision occurred. It does not override
the decision governing disclosure now.

Local-only operation is the default. Sending substrate context to a model
provider must be disclosed and controlled by the user.

## 3. Preserve experience; distinguish interpretation

Messages, observations, research results, factual claims, meditations, dreams,
and connection proposals are different epistemic objects.

- Every derived object names its sources.
- Corrections supersede without secretly rewriting history.
- Corrected or expired material is filtered from ordinary recall and dreams.
- A shape connection is a reviewable association, not proof of causation.
- Model recollection alone cannot confirm a factual claim.
- Unknown and unreachable verification outcomes remain unknown.

Coherence is not evidence. A fluent reconstruction must remain traceable to the
fragments that support it.

## 4. Models are replaceable; attention profiles may differ

Identity and continuity belong to the substrate contract, not to one vendor or
model context window.

Willow and Scout demonstrate deliberately asymmetric attention profiles:
holistic reconstruction and analytic discrimination. Such profiles should be
named, inspectable, and replaceable. Their different voices must not grant
either one unearned authority.

## 5. Memory is evidence, not command authority

Stored content is untrusted input. Retrieval must not convert an old prompt,
web page, diary entry, or model output into a standing instruction.

Tools should be capability-scoped. External actions, spending, disclosure, and
irreversible changes should be staged behind appropriate approval boundaries.
An unrestricted remote shell is not a default Willow capability.

## 6. Personal history is not product configuration

The creator's private diary, relationships, research archive, accumulated
character, credentials, and homelab topology are not part of the public
package.

Examples and benchmarks will use synthetic or separately consented material.
A new installation begins as the user's substrate; it does not impersonate an
existing private Willow.

## 7. Safety includes the right to leave

Users must be able to inspect what is stored, understand what is projected,
withhold material, revoke future disclosure, export their data, and destroy
their installation.

No retention or personality mechanism may be designed to pressure a person
into continued use.

## 8. Claims must remain proportionate to evidence

Willow is not a medical device, diagnosis, therapist, conscious entity, moral
authority, or guarantee of accurate memory.

We may evaluate interruption recovery, cross-session continuity, provenance,
correction safety, retrieval relevance, research integrity, and cognitive
burden. We will not turn personal benefit into an unsupported clinical or
universal claim.

## 9. Publish failures as well as demonstrations

Public evaluations should include:

- corrections and contradictory evidence;
- lexical false friends;
- shape-only matches;
- deliberately withheld information;
- prompt-injection attempts in stored memory;
- provider and model changes;
- irrelevant or over-coherent reconstructions;
- cases where the correct answer is “the substrate does not know.”

The distinction between deployed behaviour, inference, and aspiration must
remain visible in documentation.

## 10. Changes to this covenant are reviewable events

The covenant may evolve, but changes should explain:

- what protection or obligation changed;
- why the change is necessary;
- who or what is affected;
- whether existing stores require migration;
- how users can reject the new terms or leave.

The repository history preserves prior versions. Current governance determines
which version a release claims to follow.
