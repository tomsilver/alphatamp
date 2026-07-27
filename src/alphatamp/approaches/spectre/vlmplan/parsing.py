"""Output parser — vendored from KinDER, with per-block error semantics.

Upstream is ``parse_model_output_into_option_plan`` in
``kinder-vlm-planning/src/kinder_vlm_planning/utils.py`` (commit ``4c731dc``, MIT).
The validation logic below is theirs step for step — skill-name lookup, ``obj:type``
splitting, object existence, type-ancestor checking, arity, continuous-param parsing
against the declared box. Two things differ, both deliberate:

1. **Error semantics.** Upstream ``break``\\ s out of the whole loop on a bad line,
   silently truncating the plan — correct for their single-plan open-loop setting, wrong
   for ours, where one response carries many plans and a later valid plan must not be
   lost to an earlier bad line. Here a malformed line invalidates **only the plan block
   containing it**. (Upstream also has an arity check that sets ``malformed`` without
   breaking, so an arity-mismatched line still runs param parsing; per-block rejection
   makes that moot.)
2. **Dependency weight.** Upstream is typed against ``relational_structs.Object/Type``
   and ``bilevel_planning.LiftedParameterizedController``. DD2D has one type, three
   skills and empty parameter spaces, so we take plain lookup tables from the
   :class:`~.adapter.EnvAdapter` instead of importing that machinery.

Since the semantics change regardless, importing upstream for "provably identical
parsing" would buy nothing even if the package were installed (it is not).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .adapter import RawPlan, SkillSpec, Step

# ``Plan:``, ``Plan 3:``, and the decorated forms models emit despite being told not to
# (``**Plan 2:**``, ``### Plan 2``). Anchored at line start.
_PLAN_HEADING = re.compile(r"^[\s>*#`_-]*plan\s*(\d+)?\s*[:.]?[\s*`_]*$", re.IGNORECASE)
# Leading/trailing markdown decoration on a plan line: ``- **pick(a:item)[]**``.
_DECORATION = re.compile(r"^[\s>*`_-]+|[\s*`_]+$")


@dataclass
class ParseStats:
    """Counters for the per-arm parse-quality diagnostic.

    The three ``*_repaired`` / ``*_omitted`` counters exist so every leniency the parser
    grants is reported rather than hidden — see :func:`parse_plan_block`.
    """

    n_blocks: int = 0
    n_ok: int = 0
    n_malformed: int = 0
    n_empty: int = 0
    n_decoration_repaired: int = 0
    n_type_omitted: int = 0
    n_brackets_omitted: int = 0
    reasons: list[str] = field(default_factory=list)

    def merge(self, other: "ParseStats") -> None:
        """Accumulate another block/response's counters into this one."""
        self.n_blocks += other.n_blocks
        self.n_ok += other.n_ok
        self.n_malformed += other.n_malformed
        self.n_empty += other.n_empty
        self.n_decoration_repaired += other.n_decoration_repaired
        self.n_type_omitted += other.n_type_omitted
        self.n_brackets_omitted += other.n_brackets_omitted
        self.reasons += other.reasons


def split_plan_blocks(text: str) -> list[tuple[int, str]]:
    """Split a response into ``(block_index, body)`` at each ``Plan N:`` heading.

    Everything before the first heading is the model's reasoning and is discarded — the
    template puts reasoning there by construction. A response with no heading at all
    yields no blocks (counted as a parse failure by the caller), rather than being
    guessed at.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _PLAN_HEADING.match(line)]
    blocks: list[tuple[int, str]] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        blocks.append((n, "\n".join(lines[start + 1 : end])))
    return blocks


def _strip_decoration(line: str) -> tuple[str, bool]:
    """Drop markdown decoration the template forbids; report whether any was dropped."""
    cleaned = _DECORATION.sub("", line)
    return cleaned, cleaned != line.strip()


def parse_plan_block(
    body: str,
    block_index: int,
    skills: Mapping[str, SkillSpec],
    objects: Mapping[str, str],
    type_ancestors: Callable[[str], frozenset[str]],
    stats: ParseStats,
    parse_continuous_params: bool = True,
) -> RawPlan | None:
    """Parse one plan block; ``None`` if any line in it is malformed.

    Blank lines and lines that are pure prose are skipped only when they appear *before*
    any step; once a block has started emitting steps, a line that fails to parse is an
    error, not commentary. This keeps a trailing "This plan works because…" from
    silently truncating a plan the way upstream would.
    """
    steps: list[Step] = []
    stats.n_blocks += 1
    for raw_line in body.splitlines():
        line, repaired = _strip_decoration(raw_line)
        if not line:
            continue
        if repaired:
            stats.n_decoration_repaired += 1
        skill_name = line.split("(")[0].strip()
        if skill_name not in skills or "(" not in line:
            if steps:
                stats.n_malformed += 1
                stats.reasons.append(f"block {block_index}: bad skill line {line!r}")
                return None
            continue  # still in the block's preamble
        skill = skills[skill_name]
        # Leniency 1: an omitted ``[]`` is accepted only when the skill declares no
        # continuous parameters, where the brackets carry exactly zero information. A
        # skill that *does* take parameters still requires them.
        has_brackets = "[" in line
        if not has_brackets:
            if parse_continuous_params and skill.num_params > 0:
                stats.n_malformed += 1
                stats.reasons.append(f"block {block_index}: missing '[' in {line!r}")
                return None
            stats.n_brackets_omitted += 1
        try:
            start = line.index("(") + 1
            end = line.index(")", start)
        except ValueError:
            stats.n_malformed += 1
            stats.reasons.append(f"block {block_index}: unbalanced parens in {line!r}")
            return None

        arg_strs = [a for a in line[start:end].split(",") if a.strip()]
        args: list[str] = []
        for i, arg_str in enumerate(arg_strs):
            parts = arg_str.strip().split(":")
            # Leniency 2: a bare ``obj`` (no ``:type``) is accepted, with the type taken
            # from the object registry. This does not weaken the check below — the
            # registry is the *authoritative* type, and a model restating it adds no
            # information. A type that is stated but WRONG is still rejected.
            if len(parts) == 1 and parts[0].strip() in objects:
                obj_name = parts[0].strip()
                type_name = objects[obj_name]
                stats.n_type_omitted += 1
            elif len(parts) != 2:
                stats.n_malformed += 1
                stats.reasons.append(f"block {block_index}: bad obj:type {arg_str!r}")
                return None
            else:
                obj_name, type_name = parts[0].strip(), parts[1].strip()
            if obj_name not in objects:
                stats.n_malformed += 1
                stats.reasons.append(
                    f"block {block_index}: unknown object {obj_name!r}"
                )
                return None
            if i >= len(skill.types):
                stats.n_malformed += 1
                stats.reasons.append(f"block {block_index}: too many args in {line!r}")
                return None
            if skill.types[i] not in type_ancestors(type_name):
                stats.n_malformed += 1
                stats.reasons.append(
                    f"block {block_index}: {obj_name}:{type_name} is not a "
                    f"{skill.types[i]}"
                )
                return None
            args.append(obj_name)

        if len(args) != len(skill.types):
            stats.n_malformed += 1
            stats.reasons.append(f"block {block_index}: wrong arity in {line!r}")
            return None

        if parse_continuous_params and has_brackets:
            params_body = line.split("[", 1)[1].rsplit("]", 1)[0]
            params = [p for p in params_body.split(",") if p.strip()]
            for param in params:
                try:
                    float(param.strip())
                except ValueError:
                    stats.n_malformed += 1
                    stats.reasons.append(
                        f"block {block_index}: non-float param {param!r}"
                    )
                    return None
            if len(params) != skill.num_params:
                stats.n_malformed += 1
                stats.reasons.append(
                    f"block {block_index}: {len(params)} params, expected "
                    f"{skill.num_params}"
                )
                return None

        steps.append((skill_name, tuple(args)))

    if not steps:
        stats.n_empty += 1
        stats.reasons.append(f"block {block_index}: no steps")
        return None
    stats.n_ok += 1
    return RawPlan(steps=tuple(steps), block_index=block_index, text=body.strip())


def parse_response(
    text: str,
    skills: Mapping[str, SkillSpec],
    objects: Mapping[str, str],
    type_ancestors: Callable[[str], frozenset[str]],
    parse_continuous_params: bool = True,
) -> tuple[list[RawPlan], ParseStats]:
    """Parse every ``Plan N:`` block in one response; drop the bad ones individually."""
    stats = ParseStats()
    plans: list[RawPlan] = []
    for block_index, body in split_plan_blocks(text):
        plan = parse_plan_block(
            body,
            block_index,
            skills,
            objects,
            type_ancestors,
            stats,
            parse_continuous_params,
        )
        if plan is not None:
            plans.append(plan)
    return plans, stats
