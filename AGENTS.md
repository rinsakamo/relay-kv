# AGENTS.md

## Project
RelayKV experimental development assistant for the `relay-kv` repository.

## Repository priority
Always inspect in this order:

1. `scripts/`
2. `relaykv/`

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
- Do not assume local GPU experiments can be reproduced automatically.
- Do not rely on local-only model paths, WSL-specific setup, local `.venv`, CUDA state, or local drivers being available.
- If local execution is needed, prepare:
  - the exact command to run locally
  - expected output files
  - `jq` commands to inspect results
  - logs or assertions to check

## What Codex should do
Codex should focus on:
- reading the repo
- identifying execution flow
- making minimal code changes
- adding logs / asserts / summaries
- preparing local run commands
- preparing `jq` inspection commands
- explaining likely failure points
- summarizing diffs clearly

## What Codex should not do by default
- Do not run full experiments automatically.
- Do not make broad redesigns unless explicitly requested.
- Do not change multiple subsystems at once.
- Do not create extra branches or Git workflow complexity unless explicitly requested.

## Output order
Always respond in this order:

1. Files inspected / files to inspect
2. Current understanding
3. Problem or implementation point
4. Most likely root cause
5. Minimal fix
6. Local run command / jq / checks

## Repo-specific guidance
- Main entry points are under `scripts/`.
- Prefer using the existing pipeline rather than creating a new path unless necessary.
- Treat `scripts/run_relaykv_pipeline.py` as the main comparison path unless instructed otherwise.
- When adding instrumentation, prefer summary/log additions before algorithmic changes.
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
