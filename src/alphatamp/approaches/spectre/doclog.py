"""Parse, emit and validate SPECTRE's chaptered decision / notebook logs.

Both living logs (`docs/decisions.md`, `docs/notebook.md`) were single files that
grew past 2000 lines.  They are now split into *era chapters* under
`docs/decisions/` and `docs/notebook/`, one file per phase of the project, newest
entry first within each chapter.

The split is required to be **non-lossy**, and that is enforced rather than
asserted.  The pre-split files are frozen byte-for-byte in `docs/archive/` and
this module can reconstruct the monolith from the chapters and diff the two:

* `check_exact` -- the one-time migration gate.  Concatenating the chapters must
  reproduce the frozen monolith byte-for-byte.  Run before any hand-editing.
* `check_preserved` -- the permanent invariant.  After metadata strips are added
  and citations are wrapped in links, exact equality is impossible, so bodies are
  *normalized* first (strip the generated regions, collapse ``[text](url)`` to
  ``text``) and compared for every entry id that exists in the archive.  Entries
  added after the migration are unconstrained.  This catches a silent edit to a
  historical entry forever -- the exact discipline violation the supersession
  protocol forbids.

Each entry carries a generated metadata strip fenced in HTML comments::

    <a id="2026-07-27-necessity-observed"></a>
    ## 2026-07-27 -- Necessity is observed, not predicted

    <!--strip-->
    > **id** `2026-07-27-necessity-observed` * **status** active *
    > **tracks** method, evaluation
    <!--/strip-->

    <original body, byte-identical>

The fence matters: entry bodies already contain banner blockquotes of their own
(the 2026-07-24 DD2D entry opens with ``> [WARN] CORRECTED 2026-07-25``), so a
bare blockquote is not distinguishable from body text.  HTML comments do not
render, so the strip stays visible to a human reader and to a retriever while
staying unambiguous to this parser.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator, Sequence

__all__ = [
    "Chapter",
    "LogEntry",
    "LogSpec",
    "DECISIONS",
    "NOTEBOOK",
    "LOGS",
    "STATUSES",
    "TRACKS",
    "split_monolith",
    "parse_chapter",
    "parse_log",
    "render_chapter",
    "render_strip",
    "normalize_body",
    "check_exact",
    "check_preserved",
    "check_structure",
    "slugify",
]

# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #

#: Entry lifecycle.  Anything other than ``active`` must carry a banner.
STATUSES = (
    "active",
    "amended",
    "partially-superseded",
    "superseded",
    "retracted",
)

#: Closed track vocabulary.  Tracks are the cross-cutting view that chaptering
#: by era deliberately does not provide; an entry may carry several.
TRACKS = (
    "method",
    "evaluation",
    "data",
    "env-dd2d",
    "env-rt2d",
    "env-stickbutton2d",
    "baselines",
    "tooling",
    "infra",
    "process",
)

_LINK_FIELDS = ("supersedes", "superseded_by", "amends")

# --------------------------------------------------------------------------- #
# structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chapter:
    """One era of a log.

    `start` / `end` tile the whole timeline so every date lands in exactly one
    chapter; `end` of ``None`` means the chapter is still open.
    """

    slug: str
    title: str
    start: str
    end: str | None

    def contains(self, date: str) -> bool:
        """Whether `date` (YYYY-MM-DD) belongs to this chapter."""
        if date < self.start:
            return False
        return self.end is None or date <= self.end

    @property
    def is_open(self) -> bool:
        """Whether new entries should be appended here."""
        return self.end is None


@dataclass(frozen=True)
class LogEntry:
    """A single dated entry, plus the metadata the strip carries."""

    entry_id: str
    date: str
    heading: str
    body: str
    status: str = "active"
    tracks: tuple[str, ...] = ()
    defines: tuple[str, ...] = ()
    resolves: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()
    amends: tuple[str, ...] = ()
    ratifies: tuple[str, ...] = ()
    see_also: str = ""
    banner: str = ""

    @property
    def title(self) -> str:
        """Heading text with the leading ``## <date> --`` removed."""
        text = self.heading[3:].strip()
        return re.sub(r"^\d{4}-\d{2}-\d{2}\s*[—–-]\s*", "", text)

    @property
    def block(self) -> str:
        """The entry as it appears in the monolith: heading + body."""
        return f"{self.heading}\n{self.body}"


@dataclass(frozen=True)
class LogSpec:
    """Everything that distinguishes one log from the other."""

    name: str
    monolith: str
    directory: str
    archive: str
    heading: str
    chapters: tuple[Chapter, ...]
    trailer_file: str | None = None


#: Both logs use identical era boundaries so one mental model covers both.
#: Ranges tile the timeline, so an entry can never fail to find a chapter.
_ERAS = (
    Chapter("01-foundations", "Foundations", "2000-01-01", "2026-06-24"),
    Chapter("02-pivot", "Direction pivot", "2026-06-25", "2026-07-11"),
    Chapter("03-dd2d-v2.2", "DD2D integration and v2.2", "2026-07-12", "2026-07-22"),
    Chapter(
        "04-comparison", "Method comparison and VLMPlan", "2026-07-23", "2026-07-25"
    ),
    Chapter("05-v3-migration", "v3 migration", "2026-07-26", "2026-07-26"),
    Chapter("06-v3-performance", "v3 performance push", "2026-07-27", "2026-07-31"),
    # Closed 06 at a named phase boundary: SPECTRE acquires its second evaluation
    # environment. Everything from here is about generalising the v3 method off DD2D
    # rather than improving it on DD2D, which is a different question. The unified
    # coverage/waste entries stay in 06 -- they were measured *on* DD2D, and are what
    # made a second environment tractable rather than part of standing one up.
    Chapter(
        "07-stickbutton2d",
        "StickButton2D as a second environment",
        "2026-08-01",
        None,
    ),
)

DECISIONS = LogSpec(
    name="decisions",
    monolith="docs/decisions.md",
    directory="docs/decisions",
    archive="docs/archive/decisions_2026-07-29_monolithic.md",
    heading="SPECTRE Decisions",
    chapters=_ERAS,
    trailer_file="99-pre-refactor.md",
)

NOTEBOOK = LogSpec(
    name="notebook",
    monolith="docs/notebook.md",
    directory="docs/notebook",
    archive="docs/archive/notebook_2026-07-29_monolithic.md",
    heading="SPECTRE Notebook",
    chapters=_ERAS,
)

LOGS = {spec.name: spec for spec in (DECISIONS, NOTEBOOK)}


# --------------------------------------------------------------------------- #
# slugs
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have in into is it its of on onto
    or that the their to was were what when where which while with without not no
    now then than this these those it's""".split()
)


def slugify(title: str, *, max_words: int = 6) -> str:
    """Build a short, stable slug from an entry title.

    Slugs are minted **once**, at migration, and then live in the file as the
    permanent citation key -- this function is never used to re-derive an
    existing id, so hand-editing a slug afterwards is safe.
    """
    text = title.lower()
    text = re.sub(r"`[^`]*`", " ", text)  # drop code spans, they slug badly
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    words = [w for w in text.split() if w and w not in _STOPWORDS]
    if not words:
        words = ["entry"]
    return "-".join(words[:max_words]).strip("-")


# --------------------------------------------------------------------------- #
# monolith splitting (migration only)
# --------------------------------------------------------------------------- #

_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2})")


def split_monolith(text: str) -> tuple[str, list[LogEntry], str]:
    """Split a pre-chapter log into ``(preamble, entries, trailer)``.

    A naive ``^## `` split is wrong for both logs: `notebook.md` carries the
    format template ``## YYYY-MM-DD -- short title`` inside its header block, and
    `decisions.md` ends with a ``## Pre-refactor decisions`` bullet section.
    Only headings matching a real date open an entry; the template therefore
    stays in the preamble and the pre-refactor section becomes the trailer.
    """
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if _HEADING.match(line)]
    if not starts:
        return text, [], ""

    preamble = "".join(lines[: starts[0]])

    # The trailer is any non-dated ``## `` section after the last dated entry.
    tail = [
        i
        for i, line in enumerate(lines)
        if line.startswith("## ") and i > starts[-1] and not _HEADING.match(line)
    ]
    stop = tail[0] if tail else len(lines)
    trailer = "".join(lines[stop:])

    bounds = starts + [stop]
    entries: list[LogEntry] = []
    for k, begin in enumerate(starts):
        chunk = lines[begin : bounds[k + 1]]
        heading = chunk[0].rstrip("\n")
        date = _HEADING.match(heading).group(1)  # type: ignore[union-attr]
        body = "".join(chunk[1:])
        entry = LogEntry(entry_id="", date=date, heading=heading, body=body)
        entries.append(replace(entry, entry_id=f"{date}-{slugify(entry.title)}"))

    return preamble, _dedupe_ids(entries), trailer


def _dedupe_ids(entries: Sequence[LogEntry]) -> list[LogEntry]:
    """Suffix any colliding ids with -2, -3, ... so ids stay unique."""
    seen: dict[str, int] = {}
    out: list[LogEntry] = []
    for entry in entries:
        seen[entry.entry_id] = seen.get(entry.entry_id, 0) + 1
        count = seen[entry.entry_id]
        if count == 1:
            out.append(entry)
        else:
            out.append(replace(entry, entry_id=f"{entry.entry_id}-{count}"))
    return out


def chapter_for(spec: LogSpec, date: str) -> Chapter:
    """The chapter a given date belongs to."""
    for chapter in spec.chapters:
        if chapter.contains(date):
            return chapter
    raise ValueError(f"no chapter covers {date}")


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

STRIP_OPEN = "<!--strip-->"
STRIP_CLOSE = "<!--/strip-->"
_ANCHOR = re.compile(r'^<a id="([^"]+)"></a>$')


def _fmt_list(label: str, values: Iterable[str]) -> str:
    items = [v for v in values if v]
    return f"**{label}** {', '.join(items)}" if items else ""


def render_strip(entry: LogEntry) -> str:
    """Render the fenced metadata strip for one entry."""
    parts = [
        f"**id** `{entry.entry_id}`",
        f"**status** {entry.status}",
        _fmt_list("tracks", entry.tracks),
        _fmt_list("defines", entry.defines),
        _fmt_list("resolves", entry.resolves),
        _fmt_list("supersedes", entry.supersedes),
        _fmt_list("superseded by", entry.superseded_by),
        _fmt_list("amends", entry.amends),
        _fmt_list("ratifies", entry.ratifies),
    ]
    if entry.see_also:
        parts.append(f"**see also** {entry.see_also}")

    body = " · ".join(p for p in parts if p)
    lines = [STRIP_OPEN]
    lines += [f"> {chunk}" for chunk in _wrap(body, 84)]
    if entry.banner:
        lines.append(">")
        lines += [f"> {chunk}" for chunk in _wrap(entry.banner, 84)]
    lines.append(STRIP_CLOSE)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """Greedy wrap that never splits inside a ``[...](...)`` link."""
    tokens = re.findall(r"\[[^\]]*\]\([^)]*\)\S*|\S+", text)
    out: list[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current} {token}".strip()
        if current and len(candidate) > width:
            out.append(current)
            current = token
        else:
            current = candidate
    if current:
        out.append(current)
    return out or [""]


def render_chapter(spec: LogSpec, chapter: Chapter, entries: Sequence[LogEntry]) -> str:
    """Render one chapter file: header, then each entry with anchor and strip."""
    span = f"{chapter.start} .. {chapter.end}" if chapter.end else f"{chapter.start} .."
    state = "OPEN — new entries go here" if chapter.is_open else "closed"
    head = [
        f"# {spec.heading} — {chapter.title}",
        "",
        f"{len(entries)} entries, {span} ({state}). Newest first.",
        "Index and cross-reference tables: [README.md](README.md).",
        "",
        "---",
        "",
        "",
    ]
    out = ["\n".join(head)]
    for entry in entries:
        out.append(f'<a id="{entry.entry_id}"></a>\n')
        out.append(f"{entry.heading}\n\n")
        out.append(f"{render_strip(entry)}\n\n")
        out.append(entry.body.lstrip("\n"))
    return "".join(out)


# --------------------------------------------------------------------------- #
# parsing chapters back
# --------------------------------------------------------------------------- #

_STRIP_BLOCK = re.compile(
    rf"^{re.escape(STRIP_OPEN)}\n(.*?)^{re.escape(STRIP_CLOSE)}\n\n?",
    re.MULTILINE | re.DOTALL,
)


def _parse_strip(text: str) -> dict[str, str]:
    """Parse a rendered strip's blockquote back into a field mapping."""
    raw = " ".join(line.lstrip("> ").rstrip() for line in text.splitlines())
    fields: dict[str, str] = {}
    for chunk in raw.split("·"):
        match = re.match(r"\s*\*\*([^*]+)\*\*\s*(.*)", chunk)
        if match:
            fields[match.group(1).strip().replace(" ", "_")] = match.group(2).strip()
    return fields


def parse_chapter(text: str) -> tuple[str, list[LogEntry]]:
    """Inverse of `render_chapter`: ``(chapter_header, entries)``."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if _ANCHOR.match(line.rstrip("\n"))]
    if not starts:
        return text, []

    header = "".join(lines[: starts[0]])
    bounds = starts + [len(lines)]
    entries: list[LogEntry] = []
    for k, begin in enumerate(starts):
        chunk = "".join(lines[begin : bounds[k + 1]])
        entries.append(_parse_entry(chunk))
    return header, entries


def _parse_entry(chunk: str) -> LogEntry:
    """Parse one anchor-delimited entry region."""
    lines = chunk.splitlines(keepends=True)
    anchor = _ANCHOR.match(lines[0].rstrip("\n"))
    if anchor is None:
        raise ValueError("entry does not start with an anchor")
    entry_id = anchor.group(1)

    rest = "".join(lines[1:]).lstrip("\n")
    head_line, _, after = rest.partition("\n")
    heading = head_line.rstrip("\n")
    match = _HEADING.match(heading)
    if match is None:
        raise ValueError(f"{entry_id}: anchor is not followed by a dated heading")

    strip_match = _STRIP_BLOCK.search(after)
    if strip_match is None:
        raise ValueError(f"{entry_id}: missing {STRIP_OPEN} block")
    body = after[: strip_match.start()] + after[strip_match.end() :]

    # The banner rides in the same blockquote, separated by a bare ``>`` line.
    # It has to be split off *before* parsing fields, or the last field swallows
    # it -- which silently corrupts ids that follow it.
    head, sep, tail = strip_match.group(1).partition("\n>\n")
    fields = _parse_strip(head)
    banner = ""
    if sep:
        banner = " ".join(
            line.lstrip("> ").rstrip() for line in tail.splitlines()
        ).strip()

    def tup(key: str) -> tuple[str, ...]:
        value = fields.get(key, "")
        return tuple(v.strip() for v in value.split(",") if v.strip())

    return LogEntry(
        entry_id=entry_id,
        date=match.group(1),
        heading=heading,
        body=body,
        status=fields.get("status", "active"),
        tracks=tup("tracks"),
        defines=tup("defines"),
        resolves=tup("resolves"),
        supersedes=tup("supersedes"),
        superseded_by=tup("superseded_by"),
        amends=tup("amends"),
        ratifies=tup("ratifies"),
        see_also=fields.get("see_also", ""),
        banner=banner,
    )


def parse_log(root: Path, spec: LogSpec) -> list[LogEntry]:
    """Read every chapter of a log, newest chapter first."""
    entries: list[LogEntry] = []
    for chapter in reversed(spec.chapters):
        path = root / spec.directory / f"{chapter.slug}.md"
        if path.exists():
            entries.extend(parse_chapter(path.read_text())[1])
    return entries


# --------------------------------------------------------------------------- #
# the two non-lossiness checks
# --------------------------------------------------------------------------- #

_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def normalize_body(text: str) -> str:
    """Collapse markdown links to their visible text.

    In-log citations are *wrapped* rather than replaced -- ``` `decisions.md`
    2026-07-19 ``` becomes ``[`decisions.md` 2026-07-19](...)`` -- precisely so
    this normalization restores the original bytes and `check_preserved` can keep
    holding after the citation pass.
    """
    return _MD_LINK.sub(r"\1", text)


TRAILER_MARK = "<!--trailer-->"


def _trailer_text(root: Path, spec: LogSpec) -> str:
    """The preserved tail section, if this log has one."""
    if not spec.trailer_file:
        return ""
    path = root / spec.directory / spec.trailer_file
    if not path.exists():
        return ""
    text = path.read_text()
    return text.split(TRAILER_MARK + "\n", 1)[-1] if TRAILER_MARK in text else ""


def _reconstruct(root: Path, spec: LogSpec) -> str:
    """Rebuild the monolith's *content* from the chapters, newest entry first.

    The monolith's own preamble is deliberately excluded: it is eight lines of
    boilerplate that the generated README replaces outright.  What has to survive
    byte-for-byte is the entries and the trailer, and those are what this returns.
    """
    blocks = [entry.block for entry in parse_log(root, spec)]
    return "".join(blocks) + _trailer_text(root, spec)


def check_exact(root: Path, spec: LogSpec) -> list[str]:
    """Strict gate: every archive entry survives, in order, with its bytes.

    At migration this ran with nothing normalized and passed on untouched
    output -- that run is the proof the split itself lost nothing.  It stays
    runnable afterwards by normalizing link wrapping the same way
    `check_preserved` does, so it remains the stronger of the two checks: it
    additionally pins the *order* of the archive's entries.

    Entries are emitted **in archive order** for the comparison rather than in
    chapter order, because chaptering by date deliberately normalizes one thing:
    `notebook.md` was not strictly newest-first (a 2026-07-26 entry sat between
    two 2026-07-27 entries), and date-based assignment refiles it.  Reordering
    loses no content, so the gate checks what matters -- that every entry is
    present, exactly once, byte-identical.  Ordering is a separate invariant,
    enforced per chapter by `check_structure`.
    """
    archive = (root / spec.archive).read_text()
    preamble, original, _ = split_monolith(archive)
    want = archive[len(preamble) :]

    current = {entry.heading: entry for entry in parse_log(root, spec)}
    problems = [
        f"{spec.name}: archive entry missing from chapters: {entry.heading}"
        for entry in original
        if entry.heading not in current
    ]
    if problems:
        return problems
    # Entries written after the migration are simply not part of this comparison,
    # so the gate stays runnable as the logs grow rather than expiring on the
    # first new entry.

    got = "".join(current[entry.heading].block for entry in original)
    got += _trailer_text(root, spec)
    want, got = normalize_body(want), normalize_body(got)
    if got == want:
        return []
    diff = difflib.unified_diff(
        want.splitlines(keepends=True),
        got.splitlines(keepends=True),
        fromfile=f"{spec.archive} (entries+trailer)",
        tofile=f"{spec.directory}/* (reconstructed, archive order)",
        n=1,
    )
    return [f"{spec.name}: reconstruction differs from the frozen archive"] + [
        line.rstrip("\n") for line in list(diff)[:60]
    ]


def check_preserved(root: Path, spec: LogSpec) -> list[str]:
    """Permanent invariant: historical bodies unchanged modulo link wrapping.

    Entries are matched to the archive by **heading line**, which is preserved
    byte-for-byte and unique in practice.  Matching on the id would break the
    moment a slug were hand-edited; matching on date-rank would mis-pair if an
    entry were ever back-dated into a date that already had one.
    """
    archive = (root / spec.archive).read_text()
    _, original, _ = split_monolith(archive)
    by_heading = {entry.heading: entry for entry in original}

    problems: list[str] = []
    matched: set[str] = set()
    for entry in parse_log(root, spec):
        want_entry = by_heading.get(entry.heading)
        if want_entry is None:
            continue  # added after the migration; unconstrained
        matched.add(entry.heading)
        want = normalize_body(want_entry.block).rstrip()
        got = normalize_body(entry.block).rstrip()
        if want != got:
            problems.append(
                f"{spec.name}: {entry.entry_id} differs from the frozen archive; "
                "historical entries are append-only -- supersede it, do not edit it"
            )
    for heading in by_heading:
        if heading not in matched:
            problems.append(
                f"{spec.name}: archive entry missing from chapters: {heading}"
            )
    return problems


# --------------------------------------------------------------------------- #
# structural validation
# --------------------------------------------------------------------------- #

_SOFT_LINE_CAP = 650
_SOFT_ENTRY_CAP = 12


def check_structure(root: Path, spec: LogSpec) -> tuple[list[str], list[str]]:
    """Validate ids, statuses, tracks, ordering and referential integrity.

    Returns ``(errors, warnings)``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    entries = parse_log(root, spec)
    ids = {entry.entry_id for entry in entries}

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.entry_id] = counts.get(entry.entry_id, 0) + 1
    errors += [f"{spec.name}: duplicate id {eid}" for eid, n in counts.items() if n > 1]

    for entry in entries:
        if not entry.entry_id.startswith(entry.date):
            errors.append(f"{spec.name}: {entry.entry_id} does not start with its date")
        if entry.status not in STATUSES:
            errors.append(f"{spec.name}: {entry.entry_id} bad status {entry.status!r}")
        for track in entry.tracks:
            if track not in TRACKS:
                errors.append(f"{spec.name}: {entry.entry_id} bad track {track!r}")
        if entry.status != "active" and not entry.banner:
            errors.append(
                f"{spec.name}: {entry.entry_id} is {entry.status} but carries no banner"
            )
        if entry.superseded_by and entry.status == "active":
            errors.append(
                f"{spec.name}: {entry.entry_id} has superseded-by but status active"
            )
        for field_name in _LINK_FIELDS:
            for target in getattr(entry, field_name):
                if target.startswith("20") and target not in ids:
                    errors.append(
                        f"{spec.name}: {entry.entry_id} {field_name} -> "
                        f"unknown id {target}"
                    )

    errors += _check_symmetry(spec, entries)
    errors += _check_chapters(root, spec, entries, warnings)
    errors += check_links(root, spec)
    return errors, warnings


_FENCE = re.compile(r"```.*?```", re.DOTALL)
_LINK = re.compile(r"\]\(([^)]+)\)")


def check_links(root: Path, spec: LogSpec) -> list[str]:
    """Every relative link and ``#anchor`` in a chapter must resolve.

    Splitting the logs moved every entry down one directory, so relative links
    that used to reach a sibling doc silently broke.  Fenced code is skipped --
    it contains illustrative paths, not live links.
    """
    problems: list[str] = []
    directory = root / spec.directory
    for path in sorted(directory.glob("*.md")):
        text = _FENCE.sub("", path.read_text())
        for match in _LINK.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            filename, _, anchor = target.partition("#")
            destination = path if not filename else (path.parent / filename)
            if not destination.exists():
                problems.append(f"{spec.name}/{path.name}: dead link -> {target}")
            elif anchor and f'<a id="{anchor}"></a>' not in destination.read_text():
                problems.append(f"{spec.name}/{path.name}: dead anchor -> {target}")
    return problems


def _check_symmetry(spec: LogSpec, entries: Sequence[LogEntry]) -> list[str]:
    """`supersedes` and `superseded_by` must point at each other."""
    forward = {e.entry_id: set(e.supersedes) for e in entries}
    backward = {e.entry_id: set(e.superseded_by) for e in entries}
    problems: list[str] = []
    for eid, targets in forward.items():
        for target in targets:
            if target in backward and eid not in backward[target]:
                problems.append(
                    f"{spec.name}: {eid} supersedes {target}, but {target} does not "
                    f"list {eid} in superseded-by"
                )
    return problems


def _check_chapters(
    root: Path,
    spec: LogSpec,
    entries: Sequence[LogEntry],
    warnings: list[str],
) -> list[str]:
    """Entries must be newest-first and inside their chapter's date range."""
    problems: list[str] = []
    for chapter in spec.chapters:
        path = root / spec.directory / f"{chapter.slug}.md"
        if not path.exists():
            continue
        text = path.read_text()
        _, chapter_entries = parse_chapter(text)
        dates = [e.date for e in chapter_entries]
        if dates != sorted(dates, reverse=True):
            problems.append(f"{spec.name}/{chapter.slug}: entries are not newest-first")
        for entry in chapter_entries:
            if not chapter.contains(entry.date):
                problems.append(
                    f"{spec.name}/{chapter.slug}: {entry.entry_id} is outside "
                    f"the chapter's range"
                )
        # Only the open chapter can still grow, so only it can act on the cap.
        lines = len(text.splitlines())
        over = lines > _SOFT_LINE_CAP or len(chapter_entries) > _SOFT_ENTRY_CAP
        if over and chapter.is_open:
            warnings.append(
                f"{spec.name}/{chapter.slug}: {lines} lines / "
                f"{len(chapter_entries)} entries -- past the soft cap "
                f"({_SOFT_LINE_CAP}/{_SOFT_ENTRY_CAP}); close this chapter at the "
                "next named phase boundary and open the next one"
            )
    del entries  # validated per chapter above
    return problems


# --------------------------------------------------------------------------- #
# id-vocabulary harvesting (for the README resolution table)
# --------------------------------------------------------------------------- #

_ID_TOKEN = re.compile(
    r"\b(?:G\d+[ab]?|R\d+|A\d+|P-v3-\d+|D-\d+|C\d+|L\d+|D\d+|P\d+)\b"
)


def harvest_ids(entries: Sequence[LogEntry]) -> dict[str, list[str]]:
    """Map each declared `defines` / `resolves` token to the entries using it."""
    table: dict[str, list[str]] = {}
    for entry in entries:
        for token in list(entry.defines) + list(entry.resolves):
            table.setdefault(token, []).append(entry.entry_id)
    return table


@dataclass
class LogView:
    """A loaded log, ready for index rendering."""

    spec: LogSpec
    entries: list[LogEntry] = field(default_factory=list)

    def by_chapter(self) -> Iterator[tuple[Chapter, list[LogEntry]]]:
        """Yield each chapter with its entries, newest chapter first."""
        for chapter in reversed(self.spec.chapters):
            group = [e for e in self.entries if chapter.contains(e.date)]
            if group:
                yield chapter, group
