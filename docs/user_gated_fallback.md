# User-Gated Fallback

## Purpose

**User-Gated Fallback** is a runtime policy concept for cases where the system may need to leave a strict low-latency path and perform a more expensive memory recall step.

The core idea is that some fallback behaviors should require explicit user approval rather than happening silently.

## Runtime states

### `PROPOSE_FALLBACK`

The system determines that a useful answer may require a slower or colder memory path and prepares a user-visible proposal.

### `WAIT_USER_APPROVAL`

The system has presented the fallback proposal and is waiting for the user's decision before continuing.

## Required fields

The proposal object should expose at least:

- `approval_required`
- `approval_reason`
- `proposed_fallback_mode`
- `user_visible_message`
- `fallback_if_denied`
- `estimated_extra_latency_ms`
- `estimated_ram_read_bytes`
- `estimated_ssd_read_bytes`

## Policy modes

### `LIVE_LOW_LATENCY`

In this policy:

- SSD fallback requires approval
- FullKV tiered fallback requires approval
- the default bias is to preserve interactive responsiveness

This is the most relevant mode for AI Vtuber or live assistant UX.

### `OFFLINE_CREATIVE`

In this policy:

- RAM recall may be automatic within budget
- SSD recall may be automatic within budget
- the default bias is to preserve continuity and quality when latency is less critical

This is relevant for creative writing or offline assistant tasks.

### `BENCHMARK`

In this policy:

- behavior should be deterministic
- approval is disabled
- fallback behavior should be explicitly configured rather than negotiated at runtime

This is intended for reproducible experiments and evaluation.

## AI Vtuber UX example

For an AI Vtuber or AI character, a fallback prompt might be phrased as:

> May I search older memory? It may take a little longer.

The important property is that the system communicates the tradeoff clearly instead of silently switching into a much slower recall path during live interaction.

## Status

User-Gated Fallback is a **design-level runtime policy**, not a claimed end-to-end implementation in the current repository.
