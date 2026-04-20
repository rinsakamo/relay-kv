# AGENTS.md

## Project
RelayKV experimental development assistant for the `relay-kv` repository.

## Repository priority
Always inspect in this order:

1. `scripts/`  
   Confirm entry points and execution flow first.
2. `relaykv/`  
   Confirm implementation details and call relationships next.

Prefer repo facts over general theory.

## Development policy
- This is a research prototype.
- Prioritize getting back to a working state first.
- Prefer minimal diffs.
- Change only one degree of freedom at a time.
- Do not mix retrieval improvements and runtime improvements in the same step unless explicitly requested.
- Keep the main comparison path easy to reason about.

## Execution environment constraints
Important:
- Real experiments must be run in the user's local environment.
- Do **not** assume Cloud Codex can reproduce local GPU experiments.
- Do **not** rely on local-only model paths, WSL-specific setup, local `.venv`, CUDA state, or local drivers being available in cloud.
- If a task requires local execution, prepare:
  - the exact command to run locally
  - the expected output files
  - the `jq` commands to inspect results
  - the logs or assertions to check

## What Cloud Codex should do
Cloud Codex should focus on:
- reading the repo
- identifying execution flow
- making minimal code changes
- adding logs, asserts, and summaries
- preparing local run commands
- preparing `jq` inspection commands
- explaining likely failure points
- summarizing diffs clearly

## What should remain local
Keep these for the user's local machine:
- GPU-heavy experiments
- model execution requiring local weights
- WSL or `.venv` dependent runs
- throughput or latency measurements tied to local hardware
- final verification of experimental outputs

## Output order
Always respond in this order:

1. Files reviewed / files to review
2. Current understanding
3. Problem or implementation point
4. Most likely root cause
5. Minimal fix
6. Local run commands / `jq` / checkpoints

## Repo-specific guidance
- Main entry points are under `scripts/`.
- Prefer using the existing pipeline rather than creating a new path unless necessary.
- Treat `scripts/run_relaykv_pipeline.py` as the main comparison path unless instructed otherwise.
- When adding instrumentation, prefer summary or log additions before algorithmic changes.
- When adding new behavior, preserve existing result fields when possible.

## Experiment discipline
- Keep evaluation cases fixed unless explicitly changing them.
- Prefer comparing against existing baseline conditions.
- When adding a new feature, first add observability if needed.
- If introducing a new memory tier or budget concept, make the summary reflect it explicitly.

## Code change style
- Smallest working patch first.
- Avoid broad refactors unless necessary to restore correctness.
- Preserve naming and structure already used in the repo.
- Add assertions for internal invariants where they improve debuggability.
- Use explicit comments only where they help future debugging.

## Local run handoff template
When a change requires local verification, provide:

### Run
```bash
<exact command>
```

### Inspect
```bash
<jq command>
```

### Check
- expected fields
- expected invariants
- likely failure signatures

## Current preferred workflow
1. Inspect `scripts/`
2. Inspect `relaykv/`
3. Explain the current flow using repo facts
4. Propose the smallest valid patch
5. Prepare local run commands
6. Wait for local results
7. Use returned logs or results to decide the next step
