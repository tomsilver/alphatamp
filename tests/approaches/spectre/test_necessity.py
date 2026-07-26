"""Necessity labels behave as specified, including at the cases that motivated the spec.

The binary alternative (``necessary`` = in *every* minimal feasible subset) was rejected on
measurement, not taste: it is empty in 33.2% of dd2d_v3 episodes and under-estimates
difficulty by an amount that grows with stratum. The synthetic tests below pin the exact
behaviour at the configurations where the two definitions diverge, so a future edit that
quietly reintroduces the intersection semantics fails here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphatamp.approaches.spectre.necessity import necessity_labels

_ROOT = Path(__file__).resolve().parents[3]
_TRAIN = _ROOT / "data" / "spectre" / "raw" / "dd2d_v3" / "train"


class _Op:
    def __init__(self, name: str, args: tuple[str, ...]) -> None:
        self.name = name
        self.parameters = [type("P", (), {"name": a})() for a in args]


class _Skel:
    def __init__(self, staged: tuple[str, ...]) -> None:
        ops = []
        for o in staged:
            ops += [_Op("pick", (o,)), _Op("place-buffer", (o,))]
        ops.append(_Op("retrieve", ("target",)))
        self.operator_seq = ops


class _Out:
    def __init__(self, ok: bool) -> None:
        self.outcome = "success" if ok else "fail"


class _Atom:
    def __init__(self, name: str) -> None:
        self.objects = [type("O", (), {"name": name})()]


class _Prov:
    env_variant = "dd2d_v3"


class _Ep:
    """Minimal duck-typed episode: (staged subset, feasible) pairs."""

    def __init__(self, pool: list[tuple[tuple[str, ...], bool]]) -> None:
        self.skeleton_pool = [_Skel(s) for s, _ in pool]
        self.outcomes = [_Out(ok) for _, ok in pool]
        self.goal_atoms = [_Atom("target")]
        self.provenance = _Prov()
        names = {o for s, _ in pool for o in s} | {"target"}
        self.object_registry = {n: "item" for n in names}


def test_disjoint_minimal_solutions_keep_a_calibrated_estimate() -> None:
    """The case the binary definition gets wrong.

    Two disjoint minimal solutions {A,B} and {C,D}: the intersection is empty, so a binary
    label would supply no positive and estimate difficulty 0. The marginal splits the mass
    and recovers the true size, 2.
    """
    labels = necessity_labels(
        _Ep(
            [
                (("A", "B"), True),
                (("C", "D"), True),
                (("A",), False),
                (("A", "B", "C"), True),  # feasible but not minimal -> excluded
            ]
        )
    )
    assert labels is not None
    assert labels.min_size == 2 and labels.n_minimal == 2
    assert labels.p == {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5}
    assert labels.d_hat == pytest.approx(2.0)


def test_unanimous_solution_gives_hard_labels() -> None:
    """With one minimal solution the marginal degenerates to the binary label."""
    labels = necessity_labels(_Ep([(("A", "B"), True), (("A", "C"), False)]))
    assert labels is not None
    assert labels.p == {"A": 1.0, "B": 1.0}
    assert labels.d_hat == pytest.approx(2.0)


def test_orderings_are_deduped_and_any_ordering_counts() -> None:
    """A subset is one solution however many of its permutations the pool enumerated.

    Counting orderings would weight a subset by how many permutations happened to be
    sampled -- a sampler artifact leaking into a supervised label. And a subset feasible
    under *some* ordering is a real solution.
    """
    # {A,B} appears three times, feasible only once; {C,D} twice, never feasible.
    labels = necessity_labels(
        _Ep(
            [
                (("A", "B"), False),
                (("B", "A"), True),
                (("A", "B"), False),
                (("C", "D"), False),
                (("D", "C"), False),
            ]
        )
    )
    assert labels is not None
    assert (
        labels.n_minimal == 1
    ), "orderings of one subset must collapse to one solution"
    assert labels.p == {"A": 1.0, "B": 1.0}


def test_no_feasible_candidate_yields_none_not_zeros() -> None:
    """All-zeros would teach the head that nothing is ever required."""
    assert necessity_labels(_Ep([(("A",), False), (("A", "B"), False)])) is None


def test_goal_objects_are_excluded_from_the_label() -> None:
    """The target is manipulated by every candidate, so labelling it would spend a logit
    on a constant that ``obj_is_target`` already states."""
    labels = necessity_labels(_Ep([(("A",), True)]))
    assert labels is not None
    assert "target" not in labels.p
    assert labels.d_hat == pytest.approx(1.0)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_TRAIN / "episodes").is_dir(), reason="dd2d_v3 collection absent"
)
def test_d_hat_equals_the_stratum_on_the_real_corpus() -> None:
    """The calibration claim, on real data: ``d_hat == stratum`` exactly.

    This is what makes the estimate usable as a difficulty signal rather than a loose
    bound -- and it is also why the head must be described plainly as a learned difficulty
    estimator rather than dressed up as something subtler.
    """
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    errors = []
    n_minimal = []
    for path in list_episodes(_TRAIN):
        episode = load_episode(path)
        labels = necessity_labels(episode)
        if labels is None:
            continue
        errors.append(abs(labels.d_hat - labels.min_size))
        n_minimal.append(labels.n_minimal)
        assert all(0.0 <= v <= 1.0 for v in labels.p.values())
    assert len(errors) >= 400
    assert max(errors) == pytest.approx(
        0.0, abs=1e-9
    ), f"max |d_hat - stratum| {max(errors)}"
    # multi-solution episodes are exactly where soft beats binary; confirm they exist
    assert sum(1 for n in n_minimal if n > 1) > 50
