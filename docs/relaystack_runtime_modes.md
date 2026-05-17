# RelayStack Runtime Modes

Design status: 2026-05-17 JST.

This document records the design-level runtime mode contract for RelayStack. It is a design/contract document only. It does not claim that the described runtime integration is implemented.

Runtime modes here are contract states, not current runtime implementation.

## Core framing

RelayStack Core handles decision, metadata, budget, lineage, and trace.

RelayKV is the fixed-VRAM-budget decode-time KV working-set controller.

RelayMEM and RelayCTX remain context-planning layers.

Development and evaluation order is:

```text
MEM → CTX → KV
```

VRAM-pressure runtime activation order is different:

```text
KV first
```

This difference is intentional.

## Two independent runtime axes

RelayStack runtime should be modeled with two independent axes rather than one linear state machine.

### VRAM / KV axis

- `NORMAL_FULL`
- `RELAYKV_SHADOW`
- `RELAYKV_APPLY`
- `RELAYKV_SAFE_DEGRADE`
- `BLOCKED_NO_SAFE_KV_PATH`
- `REQUEST_CONTEXT_REDUCTION`

### Context / memory axis

- `CONTEXT_NORMAL`
- `CTX_BUDGETED`
- `MEM_RECALL`
- `POST_STREAM_INDEX`

Important:

- VRAM pressure can activate RelayKV even when RelayMEM and RelayCTX are not active.
- Memory recall can happen without RelayKV if active context and VRAM budgets remain safe.

## VRAM / KV mode semantics

### NORMAL_FULL

Meaning:

- FullKV path is available.
- RelayKV is not the active applied path.

Entry condition:

- budget is safe enough for current FullKV operation
- no RelayKV apply gate is active

FullKV availability:

- available

Allowed fallback / degrade behavior:

- Core may emit pressure warnings
- Core may emit a shadow recommendation
- fallback is trivial because FullKV is already the active safe path

Core may emit:

- pressure warning events
- shadow recommendation events
- budget snapshot events

Core must not assume:

- that RelayKV apply has already happened
- that degrade or block semantics are active

### RELAYKV_SHADOW

Meaning:

- FullKV output path remains active.
- RelayKV decisions are evaluated in shadow.

Entry condition:

- VRAM pressure or policy indicates that RelayKV should be evaluated
- shadow capability and gating allow shadow evaluation

FullKV availability:

- available

Allowed fallback / degrade behavior:

- fallback to FullKV is valid because FullKV is already the active path
- no KV materialization is required by this contract state

Core may emit:

- shadow decision events
- quality recommendation events
- apply readiness events

Core must not assume:

- that shadow results imply apply is safe
- that engine-side KV mutation has already occurred

### RELAYKV_APPLY

Meaning:

- RelayKV working set is the active KV path.

Entry condition:

- apply capability is present
- shadow or prior evidence is sufficient
- budget and safety gates pass

FullKV availability:

- not assumed after apply under VRAM pressure

Allowed fallback / degrade behavior:

- transition to `RELAYKV_SAFE_DEGRADE`
- transition to `BLOCKED_NO_SAFE_KV_PATH`
- transition to `REQUEST_CONTEXT_REDUCTION`

Core may emit:

- active apply state events
- capability snapshot events
- degrade recommendation events
- block or context-reduction recommendation events

Core must not assume:

- automatic fallback to FullKV
- scheduler-owned recovery behavior as a Core guarantee
- that apply is reversible without explicit adapter support

### RELAYKV_SAFE_DEGRADE

Meaning:

- FullKV is unavailable or unsafe.
- RelayKV shrinks to a safer or smaller mode.

Entry condition:

- applied RelayKV path encounters higher risk
- budget pressure, quality risk, or capability constraints require a smaller safe mode

FullKV availability:

- unavailable or unsafe

Allowed fallback / degrade behavior:

- reduce retrieval budget
- preserve RECENT / ANCHOR preferentially
- drop optional retrieved blocks
- continue only while a safe degraded path still exists

Core may emit:

- degrade reason
- quality risk signal
- budget risk signal

Core must not assume:

- that degraded RelayKV is equivalent to FullKV
- that optional blocks can always be preserved

### BLOCKED_NO_SAFE_KV_PATH

Meaning:

- no safe KV path exists within the current budget

Entry condition:

- degrade is insufficient
- apply path cannot continue safely
- no valid fallback path remains

FullKV availability:

- unavailable or unsafe

Allowed fallback / degrade behavior:

- stop or deny continuation
- emit block reason to external layers

Core may emit:

- blocked state
- required user or app action signal
- trace event with failure or safety reason

Core must not assume:

- that generation should continue
- that App / Agent handling has already occurred

### REQUEST_CONTEXT_REDUCTION

Meaning:

- active context must be reduced before continuing

Entry condition:

- current active context cannot be served safely
- a reduced or repacked request may restore a safe path

FullKV availability:

- not assumed

Allowed fallback / degrade behavior:

- emit request for reduced or repacked context
- hand off UX and proposal generation to App / Agent and RelayCTX planning paths

Core may emit:

- request-context-reduction events
- target budget hints
- trace event with `user_action_required=true`

Core must not assume:

- that context reduction UX is executed inside RelayStack Core
- that the current request can continue without a new reduced prefill

## Term definitions

### Fallback

Return to the safest still-available path.

Fallback is valid only if that path is actually available.

### RelayKV Degrade

FullKV is unavailable, so RelayKV shrinks to a safer mode.

### Block

No safe path exists, so runtime continuation should stop or be denied.

### Request Context Reduction

Application or user must reduce active context, or RelayCTX must produce a smaller packing plan, before continuing.

## Transition matrix

| From | To | Allowed? | Reason | Notes |
| --- | --- | --- | --- | --- |
| `NORMAL_FULL` | `RELAYKV_SHADOW` | yes | VRAM pressure warning or policy trigger | shadow can start while FullKV remains active |
| `RELAYKV_SHADOW` | `NORMAL_FULL` | yes | shadow ended or no apply needed | FullKV is already the active path |
| `RELAYKV_SHADOW` | `RELAYKV_APPLY` | yes | gates and capabilities pass | apply requires explicit readiness, not shadow alone |
| `RELAYKV_APPLY` | `NORMAL_FULL` | generally no | FullKV fallback after apply is not assumed under pressure | only an explicitly available external reset path could change this |
| `RELAYKV_APPLY` | `RELAYKV_SAFE_DEGRADE` | yes | higher budget or quality risk | preferred safety transition after apply |
| `RELAYKV_SAFE_DEGRADE` | `BLOCKED_NO_SAFE_KV_PATH` | yes | no safe degraded path remains | runtime should not continue |
| `RELAYKV_SAFE_DEGRADE` | `REQUEST_CONTEXT_REDUCTION` | yes | reduced context may restore safety | App / Agent owns the user-facing flow |
| `REQUEST_CONTEXT_REDUCTION` | `NORMAL_FULL` | yes, after new request | reduced or repacked prefill restored safe path | requires a new reduced request or prefill |

## Trace and event implications

Runtime mode transitions should produce RelayStackTraceEvent-compatible events carrying at least:

- `previous_mode`
- `next_mode`
- `trigger`
- `budget_snapshot`
- `quality_signal` optional
- `fallback_or_degrade_reason`
- `adapter_capability_snapshot`
- `user_action_required`

These events should remain append-only and JSON-safe.

## Relation to Phase 13 and Phase 14

### Phase 13

Safe materialization and shadow attention compare.

FullKV fallback is valid in this phase because FullKV remains active.

### Phase 14

Gated apply and safe degrade / block / context-reduction integration.

After RelayKV apply under VRAM pressure, use:

- `RELAYKV_SAFE_DEGRADE`
- `BLOCKED_NO_SAFE_KV_PATH`
- `REQUEST_CONTEXT_REDUCTION`

Do not assume automatic FullKV fallback.

## Near-term usage

Phase 11.5-C is contract consolidation only.

This document does not add:

- runtime adapter behavior
- KV materialization
- attention connection
- scheduler changes
- tool execution inside RelayStack Core
