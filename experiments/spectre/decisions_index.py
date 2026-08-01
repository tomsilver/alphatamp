"""Maintain SPECTRE's chaptered decision / notebook logs.

Thin CLI over `alphatamp.approaches.spectre.doclog`, following the same
library-plus-entry-point split as `dd2d_compare.py` / `spectre_v3_table.py`.

Subcommands::

    split  --log decisions      one-shot migration from the frozen archive
    check  [--against-archive]  validate; --against-archive is the exact gate
    index                       regenerate the GENERATED blocks in both READMEs
    new    --log decisions --title "..." --tracks method,evaluation

The permanent rule the `check` subcommand enforces: a historical entry is
append-only.  To change what an old decision says, add a new entry and mark the
old one superseded -- never edit its body.  See `doclog`'s module docstring.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

from alphatamp.approaches.spectre import doclog
from alphatamp.approaches.spectre.doclog import (
    LOGS,
    LogEntry,
    LogSpec,
    chapter_for,
    check_exact,
    check_preserved,
    check_structure,
    harvest_ids,
    parse_chapter,
    parse_log,
    render_chapter,
    slugify,
    split_monolith,
)

REPO_MARKER = "pyproject.toml"
GEN_OPEN = "<!--BEGIN GENERATED-->"
GEN_CLOSE = "<!--END GENERATED-->"


def repo_root() -> Path:
    """Walk up from this file to the repository root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / REPO_MARKER).exists():
            return parent
    raise SystemExit("could not locate the repository root")


def spectre_root() -> Path:
    """The spectre package directory, which every log path is relative to."""
    return repo_root() / "src" / "alphatamp" / "approaches" / "spectre"


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #


def cmd_split(spec: LogSpec, root: Path) -> int:
    """Write chapter files from the frozen archive.

    Migration only.
    """
    archive = (root / spec.archive).read_text()
    preamble, entries, trailer = split_monolith(archive)
    outdir = root / spec.directory
    outdir.mkdir(parents=True, exist_ok=True)

    for chapter in spec.chapters:
        group = [e for e in entries if chapter.contains(e.date)]
        if not group:
            continue
        path = outdir / f"{chapter.slug}.md"
        path.write_text(render_chapter(spec, chapter, group))
        print(f"  wrote {path.relative_to(root)}  ({len(group)} entries)")

    if trailer and spec.trailer_file:
        path = outdir / spec.trailer_file
        head = (
            f"# {spec.heading} — pre-refactor\n\n"
            "Decisions imported from the pre-refactor spec stack, kept as the "
            "bullet list they were written as.\nIndex: [README.md](README.md).\n\n"
            f"{doclog.TRAILER_MARK}\n"
        )
        path.write_text(head + trailer)
        print(f"  wrote {path.relative_to(root)}  (trailer)")

    stash = outdir / ".original-preamble.md"
    stash.write_text(preamble)
    print(f"  stashed the original preamble at {stash.relative_to(root)}")
    return 0


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def cmd_check(specs: list[LogSpec], root: Path, exact: bool) -> int:
    """Run the validation suite; return a shell exit code."""
    failed = False
    for spec in specs:
        errors: list[str] = []
        warnings: list[str] = []
        if exact:
            errors += check_exact(root, spec)
        else:
            errors += check_preserved(root, spec)
            structure_errors, warnings = check_structure(root, spec)
            errors += structure_errors

        label = "exact round-trip" if exact else "full check"
        if errors:
            failed = True
            print(f"FAIL  {spec.name}: {label}")
            for line in errors[:80]:
                print(f"      {line}")
        else:
            n = len(parse_log(root, spec))
            print(f"ok    {spec.name}: {label} ({n} entries)")
        for line in warnings:
            print(f"warn  {line}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #


def _entry_link(spec: LogSpec, entry: LogEntry) -> str:
    chapter = chapter_for(spec, entry.date)
    return f"[{entry.entry_id}]({chapter.slug}.md#{entry.entry_id})"


def _status_mark(status: str) -> str:
    return {
        "active": "",
        "amended": "amended",
        "partially-superseded": "**partly superseded**",
        "superseded": "**superseded**",
        "retracted": "**retracted**",
    }.get(status, status)


def _render_index(spec: LogSpec, root: Path) -> str:
    """Build the generated half of a log's README."""
    entries = parse_log(root, spec)
    out: list[str] = []

    out.append("## Chapters\n")
    out.append("| Chapter | Entries | Span | State |")
    out.append("|---|---|---|---|")
    for chapter in reversed(spec.chapters):
        group = [e for e in entries if chapter.contains(e.date)]
        if not group:
            continue
        span = f"{group[-1].date} .. {group[0].date}"
        state = "**open**" if chapter.is_open else "closed"
        out.append(
            f"| [{chapter.slug}]({chapter.slug}.md) — {chapter.title} "
            f"| {len(group)} | {span} | {state} |"
        )
    if spec.trailer_file:
        out.append(
            f"| [{spec.trailer_file[:-3]}]({spec.trailer_file}) — pre-refactor "
            "| — | 2026-04 | closed |"
        )
    out.append("")

    out.append("## All entries, newest first\n")
    out.append("| Date | Entry | Tracks | Status |")
    out.append("|---|---|---|---|")
    for entry in entries:
        out.append(
            f"| {entry.date} | {_entry_link(spec, entry)} "
            f"| {', '.join(entry.tracks)} | {_status_mark(entry.status)} |"
        )
    out.append("")

    out.append("## By track\n")
    for track in doclog.TRACKS:
        group = [e for e in entries if track in e.tracks]
        if group:
            links = ", ".join(_entry_link(spec, e) for e in group)
            out.append(f"- **{track}** — {links}")
    out.append("")

    table = harvest_ids(entries)
    if table:
        out.append("## ID resolution\n")
        out.append("Where each gate / revision / prediction / constraint is decided.\n")
        out.append("| ID | Decided in |")
        out.append("|---|---|")
        for token in sorted(table, key=_id_sort_key):
            ids = ", ".join(
                _entry_link(spec, e) for e in entries if e.entry_id in table[token]
            )
            out.append(f"| `{token}` | {ids} |")
        out.append("")

    superseded = [e for e in entries if e.status != "active"]
    if superseded:
        out.append("## Do not quote\n")
        out.append(
            "Conclusions and numbers that later entries retracted or replaced. "
            "Check here before citing any figure from a historical entry.\n"
        )
        out.append("| Entry | Status | What replaced it |")
        out.append("|---|---|---|")
        for entry in superseded:
            repl = ", ".join(
                _entry_link(spec, e)
                for e in entries
                if e.entry_id in entry.superseded_by
            )
            out.append(
                f"| {_entry_link(spec, entry)} | {_status_mark(entry.status)} "
                f"| {repl or entry.banner[:120]} |"
            )
        out.append("")

    out.append("## Legacy citation resolution\n")
    out.append(
        f"Code docstrings cite this log as `` `{spec.name}.md` <date> ``. Dates "
        "collide, so this table resolves each to the entries on that date.\n"
    )
    out.append("| Cited date | Entries |")
    out.append("|---|---|")
    seen: dict[str, list[LogEntry]] = {}
    for entry in entries:
        seen.setdefault(entry.date, []).append(entry)
    for date in sorted(seen, reverse=True):
        links = ", ".join(_entry_link(spec, e) for e in seen[date])
        out.append(f"| {date} | {links} |")
    out.append("")
    return "\n".join(out)


def _id_sort_key(token: str) -> tuple[str, int, str]:
    match = re.match(r"([A-Za-z-]+?)(\d+)([ab]?)$", token)
    if match:
        return (match.group(1), int(match.group(2)), match.group(3))
    return (token, 0, "")


def cmd_index(specs: list[LogSpec], root: Path) -> int:
    """Rewrite the region between the GENERATED markers in each README."""
    for spec in specs:
        path = root / spec.directory / "README.md"
        if not path.exists():
            print(f"skip  {spec.directory}/README.md (write the preamble first)")
            continue
        text = path.read_text()
        if GEN_OPEN not in text or GEN_CLOSE not in text:
            print(f"skip  {spec.directory}/README.md (no GENERATED markers)")
            continue
        head, rest = text.split(GEN_OPEN, 1)
        _, tail = rest.split(GEN_CLOSE, 1)
        body = _render_index(spec, root)
        path.write_text(f"{head}{GEN_OPEN}\n\n{body}\n{GEN_CLOSE}{tail}")
        print(f"  regenerated {path.relative_to(root)}")
    return 0


# --------------------------------------------------------------------------- #
# new
# --------------------------------------------------------------------------- #


def cmd_new(spec: LogSpec, root: Path, title: str, tracks: str, date: str) -> int:
    """Scaffold a new entry at the top of the open chapter."""
    chapter = chapter_for(spec, date)
    path = root / spec.directory / f"{chapter.slug}.md"
    if not path.exists():
        raise SystemExit(f"{path} does not exist")

    entry = LogEntry(
        entry_id=f"{date}-{slugify(title)}",
        date=date,
        heading=f"## {date} — {title}",
        body="\n**Context.**\n\n**Decision.**\n\n**Consequences.**\n\n---\n\n",
        tracks=tuple(t.strip() for t in tracks.split(",") if t.strip()),
    )
    header, entries = parse_chapter(path.read_text())
    rendered = render_chapter(spec, chapter, [entry] + entries)
    # keep whatever header the chapter already had
    path.write_text(header + rendered.split("---\n\n", 1)[1])
    print(f"  added {entry.entry_id} to {path.relative_to(root)}")
    print("  now write context -> decision -> consequences, then run: index")
    return 0


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_split = subs.add_parser("split", help="one-shot migration")
    p_split.add_argument("--log", choices=sorted(LOGS), required=True)

    p_check = subs.add_parser("check", help="validate the logs")
    p_check.add_argument("--log", choices=sorted(LOGS), default=None)
    p_check.add_argument(
        "--against-archive",
        action="store_true",
        help="exact byte-for-byte round trip (the one-time migration gate)",
    )

    p_index = subs.add_parser("index", help="regenerate README index blocks")
    p_index.add_argument("--log", choices=sorted(LOGS), default=None)

    p_new = subs.add_parser("new", help="scaffold a new entry")
    p_new.add_argument("--log", choices=sorted(LOGS), required=True)
    p_new.add_argument("--title", required=True)
    p_new.add_argument("--tracks", default="")
    p_new.add_argument("--date", default=dt.date.today().isoformat())

    args = parser.parse_args(argv)
    root = spectre_root()
    chosen = [LOGS[args.log]] if getattr(args, "log", None) else list(LOGS.values())

    if args.cmd == "split":
        return cmd_split(LOGS[args.log], root)
    if args.cmd == "check":
        return cmd_check(chosen, root, args.against_archive)
    if args.cmd == "index":
        return cmd_index(chosen, root)
    if args.cmd == "new":
        return cmd_new(LOGS[args.log], root, args.title, args.tracks, args.date)
    return 1


if __name__ == "__main__":
    sys.exit(main())
