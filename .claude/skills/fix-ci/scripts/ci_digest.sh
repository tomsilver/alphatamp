#!/bin/bash
# ci_digest.sh — run one CI stage of run_ci_checks.sh, save the FULL output to
# a log under /tmp, and print only a compact digest (~30 lines max) to stdout.
#
# This is the token-efficiency core of the fix-ci skill: the model should read
# digests, never raw logs. For more context on one specific finding, grep the
# log path printed in the digest (grep -n -A5 'pattern' <log>) — never cat it.
#
# Usage: ci_digest.sh <stage> [args]
#   scope                  branch-changed files + pre-existing dirty files
#   format                 run ./run_autoformat.sh; report what changed, attributed
#                          per formatter (black/isort = CI-required, docformatter = not)
#   mypy                   mypy . (NEVER run mypy on single files — phantom
#                          'import conftest' errors)
#   pylint-fast <files…>   direct pylint on listed files (fast iteration only;
#                          NOT the gate)
#   pylint-full            pytest . --pylint -m pylint (authoritative, slow ~minutes)
#   tests                  pytest tests/ -q --tb=no (failure ids only)
#   test-one <nodeid>      one test with --tb=short (focused traceback)
#   confirm                all four CI gates, one exit code per stage
#                          (run_ci_checks.sh itself has no set -e and masks
#                          non-final failures — use this instead)
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 2
VENV=.venv/bin
LOGDIR="${TMPDIR:-/tmp}"
MAX_LINES=20

stage="${1:-}"
shift || true

per_file_counts() {  # stdin: lines starting with path:line...
    cut -d: -f1 | sort | uniq -c | sort -rn | head -10
}

case "$stage" in

scope)
    mb=$(git merge-base main HEAD) || exit 2
    echo "merge-base(main, HEAD) = $mb"
    echo "--- in-scope: files changed on this branch ---"
    git diff --name-only "$mb" HEAD
    echo "--- working-tree dirty files (pre-existing ones must NEVER be staged) ---"
    git status --porcelain
    ;;

format)
    # Run the three formatters separately so changes are attributable:
    # black/isort changes are CI-required (GitHub runs black --check + isort
    # --check-only); docformatter changes are NOT enforced in CI.
    log="$LOGDIR/ci_format.log"
    : > "$log"
    snap() { git status --porcelain | sort; }
    s0=$(snap)
    "$VENV/python" -m black . >>"$log" 2>&1
    s1=$(snap)
    "$VENV/docformatter" -i -r . --exclude venv >>"$log" 2>&1
    s2=$(snap)
    "$VENV/isort" . >>"$log" 2>&1
    s3=$(snap)
    blacked=$(comm -13 <(echo "$s0") <(echo "$s1") | awk '{print $NF}')
    docfmt=$(comm -13 <(echo "$s1") <(echo "$s2") | awk '{print $NF}')
    isorted=$(comm -13 <(echo "$s2") <(echo "$s3") | awk '{print $NF}')
    echo "log=$log"
    echo "--- changed by black (CI-required) ---"; echo "${blacked:-<none>}"
    echo "--- changed by isort (CI-required) ---"; echo "${isorted:-<none>}"
    echo "--- changed by docformatter ONLY if not listed above (NOT CI-enforced; out-of-scope ones may be git-restored) ---"
    echo "${docfmt:-<none>}"
    ;;

mypy)
    log="$LOGDIR/ci_mypy.log"
    "$VENV/mypy" . >"$log" 2>&1
    rc=$?
    total=$(grep -c ' error: ' "$log")
    echo "mypy exit=$rc errors=$total log=$log"
    if [ "$total" -gt 0 ]; then
        echo "--- errors per file ---"
        grep ' error: ' "$log" | per_file_counts
        echo "--- first $MAX_LINES errors ---"
        grep ' error: ' "$log" | head -"$MAX_LINES"
    fi
    exit "$rc"
    ;;

pylint-fast)
    [ $# -ge 1 ] || { echo "usage: ci_digest.sh pylint-fast <files...>"; exit 2; }
    log="$LOGDIR/ci_pylint_fast.log"
    "$VENV/pylint" -j 0 --rcfile=.pylintrc "$@" >"$log" 2>&1
    rc=$?
    # The installed pylint (4.x) is newer than the .pylintrc targets and emits
    # E0015/R0022 config-deprecation noise attributed to ".pylintrc:" — the
    # pytest-pylint gate does not surface those, so exclude them here.
    pl_msgs() { grep -E '^[^ ].*:[0-9]+:[0-9]+: [CRWEF][0-9]{4}' "$log" | grep -v '^\.pylintrc:'; }
    msgs=$(pl_msgs | wc -l | tr -d ' ')
    echo "pylint(direct, iteration only — pylint-full is the gate) exit=$rc messages=$msgs log=$log"
    if [ "$msgs" -gt 0 ]; then
        echo "--- messages per file ---"
        pl_msgs | per_file_counts
        echo "--- first $MAX_LINES messages ---"
        pl_msgs | head -"$MAX_LINES"
    fi
    exit "$rc"
    ;;

pylint-full)
    log="$LOGDIR/ci_pylint_full.log"
    "$VENV/pytest" . --pylint -m pylint --pylint-rcfile=.pylintrc >"$log" 2>&1
    rc=$?
    echo "pylint-full(pytest --pylint, authoritative) exit=$rc log=$log"
    tail -1 "$log"
    fails=$(grep -c '^FAILED ' "$log")
    if [ "$fails" -gt 0 ]; then
        echo "--- failing files ---"
        grep '^FAILED ' "$log" | head -"$MAX_LINES"
        # Failure blocks look like:  ___ [pylint] path/to/file.py ___
        # followed by message lines:  C:680, 0: Line too long (91/89) (line-too-long)
        echo "--- first $((MAX_LINES * 2)) message lines (with file headers) ---"
        grep -E '^_+ \[pylint\] |^[CRWEF]: *[0-9]+,' "$log" | head -$((MAX_LINES * 2))
    fi
    exit "$rc"
    ;;

tests)
    log="$LOGDIR/ci_tests.log"
    "$VENV/pytest" tests/ -q --tb=no >"$log" 2>&1
    rc=$?
    echo "pytest tests/ exit=$rc log=$log"
    tail -1 "$log"
    grep -E '^(FAILED|ERROR) ' "$log" | head -"$MAX_LINES"
    exit "$rc"
    ;;

test-one)
    [ $# -ge 1 ] || { echo "usage: ci_digest.sh test-one <nodeid>"; exit 2; }
    log="$LOGDIR/ci_test_one.log"
    "$VENV/pytest" "$@" -x --tb=short -q >"$log" 2>&1
    rc=$?
    echo "pytest $* exit=$rc log=$log"
    tail -40 "$log"
    exit "$rc"
    ;;

confirm)
    # The four CI gates with per-stage exit codes. Formatting is checked
    # (non-mutating, matching GitHub CI), not re-applied.
    overall=0
    run_gate() {  # name log cmd...
        name="$1"; log="$LOGDIR/$2"; shift 2
        "$@" >"$log" 2>&1
        rc=$?
        [ "$rc" -ne 0 ] && overall=1
        printf '%-14s exit=%-3s log=%s\n' "$name" "$rc" "$log"
    }
    run_gate "black-check" ci_confirm_black.log "$VENV/python" -m black --check .
    run_gate "isort-check" ci_confirm_isort.log "$VENV/isort" --check-only .
    run_gate "mypy" ci_confirm_mypy.log "$VENV/mypy" .
    run_gate "pylint" ci_confirm_pylint.log "$VENV/pytest" . --pylint -m pylint --pylint-rcfile=.pylintrc
    run_gate "tests" ci_confirm_tests.log "$VENV/pytest" tests/ -q --tb=no
    if [ "$overall" -eq 0 ]; then echo "VERDICT: PASS (all gates green)"; else echo "VERDICT: FAIL (see non-zero stages above)"; fi
    exit "$overall"
    ;;

*)
    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
