"""Record aggregation (P2): one token per failing *query*, not per failed sample.

The instrumented refiner emits one record per failed *sample*, so a candidate whose
``place-buffer(o)`` was retried across many buffer poses would contribute hundreds of
near-identical tokens. ``_aggregate_per_query`` collapses them to one per distinct
``(schema, args)`` -- §6.1's definition of a record -- keeping the deepest occurrence,
summing effort, and unioning culprits, so nothing the token actually encodes is lost.
"""

from __future__ import annotations


def _rec(schema: str, args, step: int, n: int, culprits=()):
    from alphatamp.approaches.spectre.failure_record import FailureRecord

    return FailureRecord(
        candidate_idx=0,
        step_index=step,
        schema=schema,
        args=tuple(args),
        culprits=tuple(culprits),
        unmoved=frozenset(),
        n_step=n,
        exhausted=False,
        budget_exhausted=False,
        effort_is_total=False,
        instrumented=True,
    )


def test_aggregation_keeps_one_record_per_query_and_loses_nothing_encoded() -> None:
    """Effort sums, culprits union, the deepest step survives.

    Everything the token actually encodes is preserved; what is dropped is the
    multiplicity of failed *poses*, which the token never carried in the first place.
    """
    from alphatamp.approaches.spectre.dataset import _aggregate_per_query

    out = _aggregate_per_query(
        [
            _rec("place-buffer", ["o1"], step=2, n=5, culprits=["a"]),
            _rec("place-buffer", ["o1"], step=4, n=7, culprits=["b"]),
            _rec("pick", ["o2"], step=1, n=3),
        ]
    )
    by = {(r.schema, r.args): r for r in out}
    assert len(out) == 2, "distinct (schema, args) must stay distinct"
    pb = by[("place-buffer", ("o1",))]
    assert pb.step_index == 4, "keeps the deepest occurrence"
    assert pb.n_step == 12, "effort is summed"
    assert pb.culprits == ("a", "b"), "culprits are unioned"
    assert by[("pick", ("o2",))].n_step == 3


def test_aggregation_is_idempotent_and_never_grows_the_set() -> None:
    from alphatamp.approaches.spectre.dataset import _aggregate_per_query

    recs = [
        _rec("place-buffer", ["o1"], step=i, n=1, culprits=[f"c{i}"]) for i in range(20)
    ] + [_rec("pick", ["o1"], step=0, n=1)]
    once = _aggregate_per_query(recs)
    assert len(once) == 2 <= len(recs)
    twice = _aggregate_per_query(once)
    assert [(r.schema, r.args, r.step_index, r.n_step) for r in once] == [
        (r.schema, r.args, r.step_index, r.n_step) for r in twice
    ]
