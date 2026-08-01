"""Tests for the chaptered decision / notebook logs.

The load-bearing test is `test_archive_content_is_preserved`: it is what makes
the split non-lossy as an enforced property rather than a claim.  The rest pin
the parser and the structural invariants the logs rely on.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from alphatamp.approaches.spectre import doclog
from alphatamp.approaches.spectre.doclog import (
    DECISIONS,
    LOGS,
    NOTEBOOK,
    LogEntry,
    check_exact,
    check_links,
    check_preserved,
    check_structure,
    normalize_body,
    parse_chapter,
    render_chapter,
    slugify,
    split_monolith,
)

SPECS = [DECISIONS, NOTEBOOK]
IDS = [spec.name for spec in SPECS]


#: The spectre package directory, which every log path is relative to.
SPECTRE_DIR = Path(__file__).resolve().parents[3] / "src/alphatamp/approaches/spectre"


# --------------------------------------------------------------------------- #
# non-lossiness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_archive_content_is_preserved(spec: doclog.LogSpec) -> None:
    """Every entry of the pre-split log survives, byte-identical.

    Historical entries are append-only: to change what one says, supersede it.
    A silent edit to a body fails here.
    """
    assert not check_preserved(SPECTRE_DIR, spec)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_archive_order_and_bytes(spec: doclog.LogSpec) -> None:
    """Stronger form: content preserved *and* in the archive's order."""
    assert not check_exact(SPECTRE_DIR, spec)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_every_archive_entry_has_a_home(spec: doclog.LogSpec) -> None:
    """No entry was dropped on the floor by chapter assignment."""
    _, archived, _ = split_monolith((SPECTRE_DIR / spec.archive).read_text())
    live = {e.heading for e in doclog.parse_log(SPECTRE_DIR, spec)}
    assert {e.heading for e in archived} <= live


def test_editing_a_historical_body_is_caught(tmp_path: Path) -> None:
    """The append-only rule is enforced, not merely documented."""
    shutil.copytree(SPECTRE_DIR / "docs", tmp_path / "docs")
    chapter = tmp_path / "docs/decisions/01-foundations.md"
    text = chapter.read_text()
    marker = "**Context.**"
    assert marker in text
    chapter.write_text(text.replace(marker, "**Contextual.**", 1))

    problems = check_preserved(tmp_path, DECISIONS)
    assert problems, "a silent edit to a historical entry must fail the check"
    assert "append-only" in problems[0]


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_structure_is_valid(spec: doclog.LogSpec) -> None:
    """Ids, statuses, tracks, ordering, supersession symmetry, links."""
    errors, _ = check_structure(SPECTRE_DIR, spec)
    assert not errors


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_links_resolve(spec: doclog.LogSpec) -> None:
    """Splitting moved every entry down a directory; nothing may dangle."""
    assert not check_links(SPECTRE_DIR, spec)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_ids_are_unique_and_dated(spec: doclog.LogSpec) -> None:
    """An id is a permanent citation key, so it must be unique and stable."""
    entries = doclog.parse_log(SPECTRE_DIR, spec)
    ids = [e.entry_id for e in entries]
    assert len(ids) == len(set(ids))
    assert all(e.entry_id.startswith(e.date) for e in entries)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_non_active_entries_carry_a_banner(spec: doclog.LogSpec) -> None:
    """A retrieved chunk must be able to tell it is quoting a dead number."""
    for entry in doclog.parse_log(SPECTRE_DIR, spec):
        if entry.status != "active":
            assert entry.banner, f"{entry.entry_id} is {entry.status} without a banner"


def test_readme_index_is_current() -> None:
    """The generated tables must match the chapters they are built from."""
    repo = SPECTRE_DIR.parents[3]
    before = {
        spec.name: (SPECTRE_DIR / spec.directory / "README.md").read_text()
        for spec in SPECS
    }
    subprocess.run(
        [sys.executable, "experiments/spectre/decisions_index.py", "index"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    for spec in SPECS:
        after = (SPECTRE_DIR / spec.directory / "README.md").read_text()
        assert after == before[spec.name], (
            f"{spec.directory}/README.md is stale -- run "
            "`python experiments/spectre/decisions_index.py index`"
        )


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def test_strip_round_trips_through_the_parser() -> None:
    """render_strip and parse_chapter must be inverses."""
    entry = LogEntry(
        entry_id="2026-07-27-a-decision",
        date="2026-07-27",
        heading="## 2026-07-27 — A decision",
        body="\nBody text.\n\n---\n\n",
        status="superseded",
        tracks=("method", "evaluation"),
        defines=("G8",),
        supersedes=("2026-07-26-older",),
        banner="⚠️ **SUPERSEDED** — see the newer entry.",
    )
    chapter = DECISIONS.chapters[-1]
    _, parsed = parse_chapter(render_chapter(DECISIONS, chapter, [entry]))
    assert len(parsed) == 1
    got = parsed[0]
    assert got.entry_id == entry.entry_id
    assert got.status == "superseded"
    assert got.tracks == ("method", "evaluation")
    assert got.defines == ("G8",)
    assert got.supersedes == ("2026-07-26-older",)
    assert got.body == entry.body
    assert "SUPERSEDED" in got.banner


def test_banner_does_not_leak_into_the_last_field() -> None:
    """Regression: the banner shares the strip's blockquote with the fields.

    Parsing them together let the banner be swallowed by whichever field came
    last, which silently corrupted the ids that followed it.
    """
    entry = LogEntry(
        entry_id="2026-07-27-x",
        date="2026-07-27",
        heading="## 2026-07-27 — X",
        body="\nb\n\n---\n\n",
        status="retracted",
        superseded_by=("2026-07-28-y",),
        banner="⚠️ **RETRACTED** — superseded by something else entirely.",
    )
    _, parsed = parse_chapter(
        render_chapter(DECISIONS, DECISIONS.chapters[-1], [entry])
    )
    assert parsed[0].superseded_by == ("2026-07-28-y",)


def test_split_monolith_ignores_non_dated_headings() -> None:
    """`notebook.md` keeps a `## YYYY-MM-DD` template in its header block."""
    text = (
        "# Log\n\nPreamble.\n\n## YYYY-MM-DD — short title\n\ntemplate\n\n"
        "## 2026-07-27 — Real entry\n\nbody\n\n---\n\n"
        "## Trailer section\n\nbullets\n"
    )
    preamble, entries, trailer = split_monolith(text)
    assert "template" in preamble
    assert [e.date for e in entries] == ["2026-07-27"]
    assert "Trailer section" in trailer


def test_normalize_body_collapses_links_to_their_text() -> None:
    """In-log citations are wrapped, not replaced, so this restores them."""
    wrapped = "numbers in [`notebook.md` 2026-07-29](../notebook/06-x.md#an-id)."
    assert normalize_body(wrapped) == "numbers in `notebook.md` 2026-07-29."


def test_slugify_drops_code_spans_and_stopwords() -> None:
    """Slugs are minted once and then hand-editable; keep them readable."""
    assert slugify("The `dead` flag is a proxy for the plan length") == (
        "flag-proxy-plan-length"
    )


def test_chapters_tile_the_timeline() -> None:
    """Every date must land in exactly one chapter, so nothing can be orphaned."""
    for spec in SPECS:
        for date in ("2026-01-01", "2026-06-25", "2026-07-19", "2030-01-01"):
            matches = [c for c in spec.chapters if c.contains(date)]
            assert len(matches) == 1, f"{date} matched {len(matches)} chapters"


def test_exactly_one_chapter_is_open() -> None:
    """New entries need an unambiguous home."""
    for spec in SPECS:
        assert sum(c.is_open for c in spec.chapters) == 1


def test_both_logs_share_chapter_boundaries() -> None:
    """One mental model covers both logs; the READMEs rely on this."""
    assert [c.slug for c in DECISIONS.chapters] == [c.slug for c in NOTEBOOK.chapters]


def test_log_registry_is_complete() -> None:
    """`LOGS` is what the CLI dispatches on."""
    assert set(LOGS) == {"decisions", "notebook"}
