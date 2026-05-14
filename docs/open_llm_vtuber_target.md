# Open-LLM-VTuber Target

## Why this target

**Open-LLM-VTuber** is a practical target because it is a concrete local runtime scenario where long interaction history, character consistency, and strict VRAM limits all matter at once.

It is also a realistic place to compare:

- ordinary context-limited local inference
- a RelayMEM + RelayKV assisted mode under the same hardware budget

This document describes a target direction, not a production-readiness claim.

## Integration boundary

The intended boundary is:

- **RelayMEM** in the memory / agent layer
- **RelayKV** in the local LLM backend / KV budget layer
- **RelayStack runtime policy** across memory recall, VRAM reservation, and fallback decisions

In other words:

- RelayMEM chooses what older summaries, profile facts, episodes, retrieved knowledge, or structured memory entries should enter active context
- RelayKV manages how the resulting active-context KV working set fits inside the live VRAM budget
- RelayStack plans when to stay in low-latency mode, when to retrieve deeper memory, and when to propose user-gated fallback

## 12GB VRAM target

A representative target is a **12GB VRAM local setup** running:

- `LLM-jp-4 8B 4bit`
- TTS
- ASR
- avatar / animation stack

The point of this target is not that every configuration fits identically, but that the available VRAM for live LLM KV is meaningfully constrained by the rest of the interactive stack.

## Reserved VRAM model

The intended budgeting idea is:

```text
total VRAM
- model weights
- TTS
- ASR
- avatar
- safety margin
= LLM working KV budget
```

RelayKV is then responsible for treating that remaining KV budget as a managed working-set budget rather than assuming the entire active context can stay fully resident on GPU.

As a schema/log-only step, `relaykv/vram_reservation.py` provides residual VRAM budget calculation for local multimodal stacks such as LLM + TTS + ASR + avatar. It does not inspect hardware or change runtime behavior.

RelayStack dry-run now combines RelayMEM context assembly, User-Gated Fallback fields, and VRAM reservation accounting into one no-model/no-GPU planning JSON for local AI Vtuber-style scenarios.

A representative planning artifact should make the residual budget explicit:

```text
runtime mode
model weight reservation
ASR reservation
TTS reservation
avatar / animation reservation
safety margin
residual KV budget
RelayMEM retrieval mode
RelayKV activation state
fallback proposal
```

This makes RelayStack useful as a low-VRAM runtime planner even before real model or engine integration.

## Demo comparison

The practical demo comparison should stay cautious:

1. **Ordinary mode**: context-limited local inference with no RelayMEM or RelayKV assistance
2. **RelayStack mode**: RelayMEM + RelayKV under the same hardware budget

The comparison is useful if it can show one or more of the following without overclaiming:

- longer sustained interaction before useful context is lost
- better recall of prior profile or episode details
- more stable runtime behavior under the same VRAM constraints
- clearer operator control over when slower fallback behavior is allowed
- explicit planning for whether the remaining KV budget is sufficient for the requested mode

## Near-term path

The near-term Open-LLM-VTuber path should remain documentation, schema, and dry-run oriented:

```text
RelayMEM Fast Recall backend
  ↓
prompt preview / CLI memory assistant smoke
  ↓
RelayStack practical runtime planning JSON
  ↓
HF max-context / FullKV baseline quality smoke
  ↓
pressure-triggered RelayKV shadow policy
  ↓
actual runtime adapter selection
```

This avoids tying the project too early to a specific UI or backend while still keeping the product target concrete.

## Caution

This target should be described as an integration direction and demo objective. The current repository does not yet claim a finished Open-LLM-VTuber integration, production runtime support, or validated real-time performance guarantees.
