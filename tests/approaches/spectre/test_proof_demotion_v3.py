"""The v3 certificate rule reduces to v2.2's subset rule, then improves on it.

Order matters here. The generic rule -- *same query, same args, ``U' superset-eq U``* --
is only allowed to differ from v2.2's ``staged' subset-eq staged`` after it has been shown
to *agree* with it candidate-for-candidate. Otherwise a behaviour change and a bug are
indistinguishable.

The improvement it is then allowed to make is refusing demotions that v2.2 accepted on
budget-exhausted failures, where the refiner named a step it never tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.failure_record import (
    FailureRecord,
    records_for_candidate,
)
from alphatamp.approaches.spectre.proof_demotion_v3 import (
    ProofStateV3,
    candidate_queries,
)

_ROOT = Path(__file__).resolve().parents[3]
_V3 = _ROOT / "data" / "spectre" / "raw" / "dd2d_v3" / "test"
_V4 = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "test"


def _staged_dd2d(skeleton) -> frozenset[str]:
    """v2.2's literal."""
    return frozenset(
        op.parameters[0].name
        for op in skeleton.operator_seq
        if op.name == "place-buffer"
    )


def _v22_dead(episode, failed: list[int]) -> set[int]:
    """v2.2's demotion: subset of an observed-blocked staged set."""
    from alphatamp.approaches.spectre.evidence import observed_blocked

    subsets = [_staged_dd2d(s) for s in episode.skeleton_pool]
    blocked = [
        subsets[i] for i in failed if observed_blocked(episode.outcomes[i], "observed")
    ]
    return {i for i, s in enumerate(subsets) if any(s <= b for b in blocked)}


def _v3_dead(episode, failed: list[int], mode: str) -> set[int]:
    spec = spec_for(episode.provenance.env_variant)
    state = ProofStateV3(candidate_queries(episode, spec), spec, mode=mode)
    for idx in failed:
        state.observe(records_for_candidate(episode, idx, spec))
    return set(state.dead)


# --------------------------------------------------------------------------- #
# semantics (fast, synthetic)
# --------------------------------------------------------------------------- #
def test_budget_exhausted_record_proves_nothing() -> None:
    """The v2.2 unsoundness. `proves_failure` is the guard that closes it."""
    ran = FailureRecord(0, 2, "retrieve", ("target",), unmoved=frozenset({"a"}))
    timed_out = FailureRecord(
        0,
        2,
        "retrieve",
        ("target",),
        unmoved=frozenset({"a"}),
        exhausted=False,
        budget_exhausted=True,
    )
    assert ran.proves_failure()
    assert not timed_out.proves_failure()


def test_strict_mode_refuses_a_budget_exit_permissive_also_refuses_contrary_evidence() -> (
    None
):
    """Permissive relaxes *absent* exhaustion evidence, never *contrary* evidence.

    A pre-v3 record simply has no flag, and v2.2 trusted it -- that is what permissive
    reproduces. A record that positively says "I stopped on the budget" is different in
    kind, and must be refused in both modes.
    """
    from alphatamp.approaches.spectre.domain import DomainSpec, QueryAxioms

    spec = DomainSpec(axioms={"q": QueryAxioms(True, True, True)})
    from alphatamp.approaches.spectre.proof_demotion_v3 import CandidateQuery

    queries = [[CandidateQuery(0, "q", ("x",), frozenset({"a", "b"}))]]
    timed_out = FailureRecord(
        1,
        0,
        "q",
        ("x",),
        unmoved=frozenset({"a"}),
        exhausted=False,
        budget_exhausted=True,
    )
    for mode in ("strict", "permissive"):
        state = ProofStateV3(queries, spec, mode=mode)
        state.observe([timed_out])
        assert not state.dead, mode


def test_undeclared_query_never_demotes() -> None:
    """With an empty registry nothing is proof-tier -- 'learning is the floor'."""
    from alphatamp.approaches.spectre.domain import EMPTY_SPEC
    from alphatamp.approaches.spectre.proof_demotion_v3 import CandidateQuery

    queries = [[CandidateQuery(0, "q", ("x",), frozenset({"a", "b"}))]]
    state = ProofStateV3(queries, EMPTY_SPEC, mode="permissive")
    state.observe([FailureRecord(1, 0, "q", ("x",), unmoved=frozenset({"a"}))])
    assert not state.dead


def test_superset_direction_is_the_sound_one() -> None:
    """``U' superset-eq U`` demotes; the reverse must not.

    Moving *fewer* objects out of the way cannot help a query that already failed. The
    opposite inference -- "it failed with more objects moved, so it fails with fewer" --
    is the unsound direction and would demote genuinely feasible plans.
    """
    from alphatamp.approaches.spectre.domain import DomainSpec, QueryAxioms
    from alphatamp.approaches.spectre.proof_demotion_v3 import CandidateQuery

    spec = DomainSpec(axioms={"q": QueryAxioms(True, True, True)})
    witnessed = FailureRecord(9, 0, "q", ("x",), unmoved=frozenset({"a", "b"}))

    more_left = [[CandidateQuery(0, "q", ("x",), frozenset({"a", "b", "c"}))]]
    fewer_left = [[CandidateQuery(0, "q", ("x",), frozenset({"a"}))]]

    s1 = ProofStateV3(more_left, spec)
    s1.observe([witnessed])
    assert s1.is_dead(0), "U' superset-eq U must demote"

    s2 = ProofStateV3(fewer_left, spec)
    s2.observe([witnessed])
    assert not s2.is_dead(0), "the reverse direction is unsound and must not demote"


# --------------------------------------------------------------------------- #
# G5 acceptance: reduction to v2.2, then the improvement
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not (_V3 / "episodes").is_dir(), reason="dd2d_v3 collection absent")
def test_permissive_mode_reproduces_v22_demotions_candidate_for_candidate() -> None:
    """The reduction test. Pre-v3 records carry no exhaustion flag, so permissive mode
    must make exactly v2.2's decisions on exactly the same candidates."""
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    paths = list_episodes(_V3)
    checked = 0
    for path in paths[:: max(1, len(paths) // 12)][:12]:
        episode = load_episode(path)
        failed = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        # grow the observed set the way a rollout does, checking agreement throughout
        for cut in (1, len(failed) // 4, len(failed) // 2, len(failed)):
            subset = failed[:cut]
            if not subset:
                continue
            assert _v3_dead(episode, subset, "permissive") == _v22_dead(
                episode, subset
            ), (
                path.name,
                cut,
            )
            checked += 1
    assert checked > 0


@pytest.mark.slow
@pytest.mark.skipif(not (_V4 / "episodes").is_dir(), reason="dd2d_v4 collection absent")
def test_strict_mode_is_sound_on_instrumented_records() -> None:
    """No demoted candidate is ever actually feasible.

    Soundness is the whole justification for applying deductions outside the network, so
    it is checked directly rather than inferred from the axioms.
    """
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    violations = successes = 0
    for path in list_episodes(_V4):
        episode = load_episode(path)
        failed = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        dead = _v3_dead(episode, failed, "strict")
        for i, o in enumerate(episode.outcomes):
            if o.outcome != "success":
                continue
            successes += 1
            if i in dead:
                violations += 1
    assert successes > 0
    assert violations == 0, f"{violations}/{successes} demoted candidates were feasible"


@pytest.mark.slow
@pytest.mark.skipif(
    not (_ROOT / "data" / "spectre" / "raw" / "dd2d_v2" / "test" / "episodes").is_dir(),
    reason="dd2d_v2 collection absent",
)
def test_strict_mode_closes_the_dd2d_v2_unsoundness() -> None:
    """The payoff of the exactness axiom, on the data that motivated it.

    dd2d_v2 contains one candidate whose refinement stopped on the wall-clock budget while
    still reporting ``retrieve(target)`` as its failing action. v2.2 trusted that and
    demoted 12 genuinely-feasible plans. Permissive mode reproduces the bug (it is
    v2.2's semantics, and that is the point of having it); strict mode must not.

    Pre-v3 records carry no exhaustion flag, so strict relies on the derived witness: an
    attempt whose total sampler calls equal the minimum possible cannot have re-sampled,
    hence really ran. The offending candidate spent 2406 calls against a floor of ~10.
    """
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    split = _ROOT / "data" / "spectre" / "raw" / "dd2d_v2" / "test"
    spec = spec_for("dd2d_v2")
    counts = {}
    for mode in ("permissive", "strict"):
        violations = successes = 0
        for path in list_episodes(split):
            episode = load_episode(path)
            failed = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
            dead = _v3_dead(episode, failed, mode)
            for i, o in enumerate(episode.outcomes):
                if o.outcome == "success":
                    successes += 1
                    violations += i in dead
        counts[mode] = (violations, successes)

    assert counts["permissive"][0] == 12, counts
    assert counts["strict"][0] == 0, counts


@pytest.mark.slow
@pytest.mark.skipif(not (_V4 / "episodes").is_dir(), reason="dd2d_v4 collection absent")
def test_instrumentation_makes_soundness_free() -> None:
    """On instrumented records, strict mode gives up nothing.

    This is the measurable return on the dd2d_v4 re-collection. Without instrumentation,
    exactness has to be *derived* from a conservative call-count witness, which declines
    ~6% of demotions it cannot vouch for. With the refiner reporting ``exhausted``
    directly, strict and permissive demote identically -- full soundness at zero cost.
    """
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    totals = {}
    for mode in ("permissive", "strict"):
        n_dead = 0
        for path in list_episodes(_V4):
            episode = load_episode(path)
            failed = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
            n_dead += len(_v3_dead(episode, failed, mode))
        totals[mode] = n_dead
    assert totals["strict"] == totals["permissive"], totals
    assert totals["strict"] > 0


@pytest.mark.slow
@pytest.mark.skipif(not (_V4 / "episodes").is_dir(), reason="dd2d_v4 collection absent")
def test_apply_demotion_false_withholds_only_the_offset() -> None:
    """The G7 eval-time axis changes what is *acted on*, not what is *deduced*.

    ``apply_demotion=False`` must leave the proof state advancing exactly as before --
    otherwise the 2x2's two columns would differ in two ways at once (offset withheld
    *and* deductions lost) and neither arm would isolate the offset's contribution.

    So: the traced ``step_dead`` sets must be identical with and without the offset up to
    the point the two rollouts still agree on what to attempt, and withholding the offset
    must actually be capable of changing the attempt order (else the switch is inert and
    the ablation would be vacuous).
    """
    import torch

    from alphatamp.approaches.spectre.inference_v3 import deployed_rollout_v3_traced
    from alphatamp.approaches.spectre.io import list_episodes, load_episode
    from alphatamp.approaches.spectre.model_v3 import SpectreV3Model, V3Config
    from alphatamp.approaches.spectre.vocab import Vocab

    ckpt = (
        _ROOT
        / "data"
        / "spectre"
        / "checkpoints_v3_g6b_recON_ovON"
        / "dd2d_v4"
        / "seed_0"
        / "best.pt"
    )
    if not ckpt.is_file():
        pytest.skip("G6b checkpoint absent")
    vocab = Vocab.from_json(
        _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"
    )
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = SpectreV3Model(
        n_ops=int(ck["n_ops"]),
        max_arity=vocab.max_operator_arity,
        cfg=V3Config(
            n_overlap_feats=2,
            n_prior_feats=0,
            max_tags=32,
            dropout_p=0.0,
            use_records=True,
        ),
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    diverged = 0
    # Stride, never truncate: episodes are stored in seed order and the collector fills
    # strata in seed bands, so a prefix is all stratum 0 -- where the first attempt
    # usually succeeds and demotion never gets to act.
    _paths = list_episodes(_V4)
    for path in _paths[:: max(1, len(_paths) // 12)][:12]:
        episode = load_episode(path)
        on = deployed_rollout_v3_traced(model, episode, vocab, "cpu")[1]
        off = deployed_rollout_v3_traced(
            model, episode, vocab, "cpu", apply_demotion=False
        )[1]
        # deductions are identical while the rollouts are still on the same trajectory
        common = 0
        for a, b in zip(on.order, off.order):
            if a != b:
                break
            common += 1
        assert on.step_dead[:common] == off.step_dead[:common], path.name
        if common < min(len(on.order), len(off.order)):
            diverged += 1
    assert diverged > 0, "withholding demotion never changed the order; switch is inert"
