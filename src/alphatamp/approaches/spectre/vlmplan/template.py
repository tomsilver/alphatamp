"""Prompt assembly: the verbatim KinDER template plus two appended blocks.

The base template file is never edited — deviations are *appended*, so the byte-identical
copy in ``prompts/`` stays checkable against upstream (see ``prompts/PROVENANCE.md`` for
the md5 and the enumerated deviations). Round 1 with ``plans_per_round=1`` reproduces the
literal single-plan KinDER prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

_PROMPT_DIR = Path(__file__).parent / "prompts"
BASE_PROMPT_FILE = _PROMPT_DIR / "kinder_llm_planning_prompt.txt"

# The vendored template's slots, as of kinder-baselines 4c731dc.
BASE_SLOTS = frozenset(
    {"controllers", "typed_objects", "type_hierarchy", "goal_str", "init_state_str"}
)

# Deviation 1 (prompts/PROVENANCE.md): KinDER asks for one plan because their metric is
# open-loop success rate. Ours is attempts-to-first-feasible, which needs an ordered
# sequence, so one response carries several plans in the unmodified per-line format.
_MULTI_PLAN_BLOCK = """
Instead of a single plan, provide {n_plans} DIFFERENT plans that each attempt to achieve \
the goal. Use exactly this structure, with one heading per plan and nothing else between \
the plan lines:
Plan 1:
<skill name>(<obj_name>:<type_name>)[]
...
Plan 2:
<skill name>(<obj_name>:<type_name>)[]
...
(continue through Plan {n_plans})

The plans must all be distinct from one another, and must be ordered from most likely to \
least likely to succeed when a low-level motion planner tries to execute them. Put all of \
your reasoning above the 'Plan 1:' heading.
"""

# Deviation 2: the repeat-suppression block, modelled on upstream's llmplanner
# "Completed plans:" slot. It lists plans and NOTHING ELSE — no outcomes, ever. That is
# the static-method hard line: VLMPlan is the zero-shot endpoint of the data axis, and
# any outcome feedback would make it an adaptive method and a different table row.
_PREVIOUS_BLOCK = """
Previously proposed plans (do not repeat):
{previous_plans}
"""


@dataclass(frozen=True)
class PromptConfig:
    """Knobs for the appended blocks. ``plans_per_round=1`` = the literal template."""

    plans_per_round: int = 10
    include_previous: bool = True


@lru_cache(maxsize=1)
def base_prompt() -> str:
    """The verbatim KinDER template text."""
    return BASE_PROMPT_FILE.read_text(encoding="utf-8")


def check_placeholders(text: str, expected: frozenset[str]) -> None:
    """Fail loudly if the vendored template's placeholder set is not what we fill.

    The real ``str.format`` hazard is a stray brace in the *template* (it would raise or
    silently swallow text), not in the substituted values — ``format`` inserts values
    verbatim and never re-scans them, so a brace inside a DD2D shape descriptor is safe
    and must NOT be escaped. This guard exists because a future re-vendor from upstream
    could add or rename a slot, and a mismatch should be an error here rather than a
    malformed prompt sent to a paid API.
    """
    found = set(re.findall(r"\{([^{}]*)\}", text))
    if found != set(expected):
        raise ValueError(
            f"Prompt template placeholders {sorted(found)} do not match the expected "
            f"{sorted(expected)}; re-check prompts/PROVENANCE.md after re-vendoring."
        )


def build_prompt(
    *,
    controllers: str,
    typed_objects: str,
    type_hierarchy: str,
    goal_str: str,
    init_state_str: str,
    config: PromptConfig,
    previous_plans: Sequence[str] = (),
) -> str:
    """Fill the template and append the enabled extension blocks.

    ``previous_plans`` is only rendered from round 2 onwards, so round 1 is maximally
    template-faithful.
    """
    base = base_prompt()
    check_placeholders(base, BASE_SLOTS)
    prompt = base.format(
        controllers=controllers,
        typed_objects=typed_objects,
        type_hierarchy=type_hierarchy,
        goal_str=goal_str,
        init_state_str=init_state_str,
    )
    if config.plans_per_round > 1:
        prompt += "\n" + _MULTI_PLAN_BLOCK.format(n_plans=config.plans_per_round)
    if config.include_previous and previous_plans:
        listed = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(previous_plans))
        prompt += "\n" + _PREVIOUS_BLOCK.format(previous_plans=listed)
    return prompt
