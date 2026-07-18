"""Tote-and-Tray Decluttering (TTD) benchmark.

A self-contained 2D-geometry TAMP benchmark whose feasibility statistic is
``packs(S, tray)`` — joint nesting of a k-object subset of concave footprints into a
tight tray. See ``docs/TTD_SPEC_v1.3.md`` for the authoritative specification and
``docs/TTD_BUILD_PLAN.md`` for the chunked build plan.

Module map (spec §12.1): ``ttd_core`` (authoritative 2D geometry, streams, op
counters), ``ttd_gen`` (instance generator), ``ttd_plan`` (proposer, refiner, evaluated
variants), ``ttd_learn`` (attacks, mid-tier, incumbents), ``ttd_sim`` (PyBullet +
realism tier), ``ttd_eval`` (metrics). This package is deliberately decoupled from the
rest of ``alphatamp`` — it imports neither the spectre model/dataset code nor
``bilevel_planning`` / ``relational_structs`` / ``kinder``.
"""
