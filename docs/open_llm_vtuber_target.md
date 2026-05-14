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

In other words:

- RelayMEM chooses what older summaries, profile facts, episodes, retrieved knowledge, or structured memory entries should enter active context
- RelayKV manages how the resulting active-context KV working set fits inside the live VRAM budget

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

## Demo comparison

The practical demo comparison should stay cautious:

1. **Ordinary mode**: context-limited local inference with no RelayMEM or RelayKV assistance
2. **RelayStack mode**: RelayMEM + RelayKV under the same hardware budget

The comparison is useful if it can show one or more of the following without overclaiming:

- longer sustained interaction before useful context is lost
- better recall of prior profile or episode details
- more stable runtime behavior under the same VRAM constraints
- clearer operator control over when slower fallback behavior is allowed

## Caution

This target should be described as an integration direction and demo objective. The current repository does not yet claim a finished Open-LLM-VTuber integration, production runtime support, or validated real-time performance guarantees.
