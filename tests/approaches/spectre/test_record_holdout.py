"""P-1: the ``record_holdout`` flag and the certificate-record token holdout.

``dataset.build_record_arrays`` drops every record that is ``proof_tier ∧ provable``
from the evidence token stream (the historical holdout). ``docs/failed_records_fix.md``
P-1 puts that behind a flag (``record_holdout``, default ``True`` = current behavior) so
a tokens-only arm can see those records for the first time.

Two things are tested:

1. **Wiring (positive control).** With a *synthetic* ``DomainSpec`` that declares a real
   DD2D schema proof-tier, the holdout must drop its provable records — so holdout-ON
   emits strictly fewer tokens than holdout-OFF. This proves the flag changes emission
   whenever a qualifying record exists, independent of what the shipped data contains.

2. **Census (the measured P-1 finding).** With each environment's *real* spec, the
   holdout is **inert on every currently-collected variant** (``delta == 0``): DD2D
   candidates fail only at ``pick``/``place-buffer`` (neither proof-tier — ``retrieve``
   is the only proof-tier DD2D schema and it produces no failure records here), and
   SB2D / restock3d_v3 declare only ``step_certificate`` (``proof_tier()`` stays False).
   So the tokens-only inertness on DD2D is **not** a holdout plumbing artifact — a result
   the doc's C4a box asked to verify empirically rather than assume.

Read-only: it tensorizes stored raw episodes through the real ``build_example`` path and
never trains or re-refines. Skips per-variant when the collection is absent (portable CI).
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from alphatamp.approaches.spectre.dataset import build_example
from alphatamp.approaches.spectre.domain import DomainSpec, QueryAxioms, spec_for
from alphatamp.approaches.spectre.io import load_episode
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[3]

_MAX_EPISODES = 12
_MAX_CONTEXT = 8


def _episodes(variant: str):
    """Yield up to ``_MAX_EPISODES`` failure-bearing train episodes + the vocab."""
    raw = REPO / "data" / "spectre" / "raw" / variant / "train" / "episodes"
    vocab_path = REPO / "data" / "spectre" / "derived" / variant / "train_vocab.json"
    paths = sorted(glob.glob(str(raw / "*")))
    if not paths or not vocab_path.exists():
        pytest.skip(f"no collection for {variant}")
    return paths, Vocab.from_json(vocab_path)


def _token_counts(variant: str, spec: DomainSpec) -> tuple[int, int, int]:
    """``(n_episodes_used, tokens_holdout_on, tokens_holdout_off)`` under ``spec``."""
    paths, vocab = _episodes(variant)
    used = on = off = 0
    for p in paths:
        if used >= _MAX_EPISODES:
            break
        ep = load_episode(Path(p))
        if ep.scene_geometry is None:
            continue
        fail_idx = [i for i, o in enumerate(ep.outcomes) if o.outcome == "fail"]
        if not fail_idx:
            continue
        ctx = frozenset(fail_idx[:_MAX_CONTEXT])
        _, recs_on = build_example(
            ep, vocab, evidence=True, context_f=ctx, spec=spec, record_holdout=True
        )
        _, recs_off = build_example(
            ep, vocab, evidence=True, context_f=ctx, spec=spec, record_holdout=False
        )
        # off never emits fewer tokens than on: the flag only *adds* records back.
        assert len(recs_off) >= len(recs_on)
        on += len(recs_on)
        off += len(recs_off)
        used += 1
    if used == 0:
        pytest.skip(f"no failure-bearing episodes for {variant}")
    return used, on, off


def test_record_holdout_wiring_positive_control() -> None:
    """A synthetic proof-tier spec makes the holdout fire on real DD2D records."""
    # `place-buffer` records are provable on DD2D; declaring the schema proof-tier
    # (monotone ∧ local ∧ exact) makes `proof_tier() ∧ proves_failure()` true for them,
    # so the holdout drops them iff record_holdout is on.
    proof_spec = DomainSpec(
        axioms={"place-buffer": QueryAxioms(monotone=True, local=True, exact=True)}
    )
    used, on, off = _token_counts("dd2d_v4", proof_spec)
    print(f"[positive-control dd2d_v4] episodes={used} on={on} off={off} delta={off-on}")
    assert off - on > 0, "record_holdout did not drop the synthetic proof-tier records"


@pytest.mark.parametrize("variant", ["dd2d_v4", "stickbutton2d_v1", "restock3d_v3"])
def test_record_holdout_inert_on_current_collections(variant: str) -> None:
    """With the real spec, no shipped collection has proof-tier records in the stream."""
    used, on, off = _token_counts(variant, spec_for(variant))
    delta = off - on
    frac = delta / off if off else 0.0
    print(
        f"[{variant}] episodes={used} tokens_on={on} tokens_off={off} "
        f"delta={delta} held-out-fraction={frac:.3f}"
    )
    # If this ever fails, a re-collection introduced proof-tier records into the token
    # stream (e.g. DD2D `retrieve` failures) — the corrected P-1 baseline now differs
    # from the current one and the analysis must be revisited.
    assert delta == 0, f"{variant}: holdout unexpectedly dropped {delta} records"
