---
name: fix-ci
description: Make ./run_ci_checks.sh pass (autoformat, mypy, pylint, pytest) so a branch can be PR'd to main. Use when the user says "fix CI", "make CI pass", "run the CI checks", "clean up the repo for a PR", or before merging. Token-efficient by design — drive everything through scripts/ci_digest.sh digests, never raw tool output.
allowed-tools: Bash(*), Read, Edit, Write, Grep, Glob, AskUserQuestion
---

# fix-ci — make `run_ci_checks.sh` pass, token-efficiently

Goal: every CI gate green (`black --check`, `isort --check-only`, `mypy .`,
`pytest . --pylint -m pylint`, `pytest tests/`), fixes committed per stage,
nothing unrelated touched. This is a **shared lab monorepo**: files not changed
on the current branch belong to other projects.

## Hard rules (read first)

- **All tool output goes through `scripts/ci_digest.sh` (skill-relative path).**
  Never run `mypy .`/`pytest`/`pylint` bare and read their output; never `cat`
  a log. For extra context on one finding: `grep -n -A5 '<pattern>' <log path
  from the digest>`.
- **Scope rule:** errors in files NOT in the branch-changed set (from the
  `scope` digest) → show the digest lines and **AskUserQuestion before
  touching them**. In-scope errors: fix autonomously.
- **Commit per stage with explicit paths** (`git add <files>` /
  `git commit -- <files>`), never `git add -A`. Pre-existing dirty files from
  the `scope` snapshot are never staged.
- **Behavior-changing fixes** (anything beyond formatting / typing / mechanical
  lint appeasement — e.g. changing what a test asserts, changing runtime
  logic) → AskUserQuestion first, even in-scope.
- **Convergence guard:** max 3 fix iterations per stage; if the error count
  does not strictly decrease between iterations, STOP, report the remaining
  digest, and ask. Never thrash.
- Fix only what gates CI (non-zero exit codes). Ignore warnings, mypy notes,
  and style nits no gate complains about.

## Repo gotchas (these cost real time/tokens if forgotten)

- `run_ci_checks.sh` has **no `set -e`** — its exit code only reflects
  `pytest tests/`. Never use it as the gate; use `ci_digest.sh confirm`,
  which reports one exit code per stage.
- **Never run mypy on a single file** — it emits phantom
  `Cannot find implementation ... "conftest"` errors. Always `mypy .`
  (via the digest; it's fast at this repo's size).
- The pylint gate (`pytest . --pylint`) is the slow stage (serial over ~126
  files). Iterate with `pylint-fast <files>`; run `pylint-full` only once to
  discover and once to confirm.
- Line limit is **89 characters (pylint), counted in characters** — `awk
  length` counts bytes and false-positives on multibyte chars like `§`.
- `-m pylint` bypasses conftest's default `-m "not slow"`; `pytest tests/`
  keeps slow tests deselected — do not "fix" slow tests CI never runs.
- docformatter runs in `run_autoformat.sh` but is **not enforced by GitHub
  CI** — docformatter-only churn in out-of-scope files should be
  `git restore`d, not committed.

## Procedure

All commands from the repo root. `DIGEST=<this skill dir>/scripts/ci_digest.sh`.

### 0. Preflight
Run `bash $DIGEST scope`. Record: (a) the in-scope file set, (b) pre-existing
dirty files (never staged). If the working tree has unrelated modifications,
mention them in the final report.

### 1. Format
Run `bash $DIGEST format` (runs the three formatters, attributing changes per
formatter).
- In-scope changed files → commit `autoformat`.
- Out-of-scope files changed by **black/isort** (CI-required): ask, then
  separate commit `autoformat (files outside this branch's changes)`.
- Out-of-scope files changed by **docformatter only**: `git restore` them
  (attribution is by first formatter that touched the file, so after
  restoring, sanity-check with `.venv/bin/black --check <f> &&
  .venv/bin/isort --check-only <f>` — if that fails, the file was
  CI-required after all; treat as the previous bullet).

### 2. Mypy (loop ≤ 3)
`bash $DIGEST mypy` → group errors by file → fix ALL errors in a file in one
edit pass (Read only the error-line regions with offset/limit) → re-run the
digest. Commit `fix mypy errors`.

### 3. Pylint
`bash $DIGEST pylint-full` once to find offenders → iterate with
`bash $DIGEST pylint-fast <offending files…>` (≤ 3 iterations) → one final
`pylint-full` as the authoritative gate (the pytest plugin can differ
marginally from the CLI). Commit `fix pylint findings`.

### 4. Tests
`bash $DIGEST tests` → for each failing node-id: `bash $DIGEST test-one
<nodeid>` for the focused traceback → fix (remember the behavior-change ask
rule) → re-run only the previously-failing node-ids via `test-one` → one full
`bash $DIGEST tests`. Commit `fix failing tests`.

If a test mutates tracked files (check `git status` after the tests stage —
e.g. fixtures regenerated under `tests/datasets/`), that is a test bug:
surface it, don't commit the churn.

### 5. Final gate + report
`bash $DIGEST confirm` — all five lines must show `exit=0`. Then verify
`git status --porcelain` shows nothing beyond the preflight's pre-existing
dirty files. Report:

| gate | exit |
|---|---|
| black-check / isort-check / mypy / pylint / tests | … |

plus: commits made (hashes + messages), out-of-scope findings the user
declined to fix, and anything deliberately left alone.

If `confirm` still fails after the stage loops, do NOT start over — report
the failing digest lines and ask how to proceed.
