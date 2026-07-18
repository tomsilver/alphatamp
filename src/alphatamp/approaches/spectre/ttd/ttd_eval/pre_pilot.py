"""The §10.0 pre-pilot: measure the η landscape (TTD spec §10.0).

This is the project's first go/no-go gate, run **before** any generator, refiner
integration, or learning code. For each cell (operating point × matched-ΣA band ×
clearance dial c_v) it samples random k-subsets from the member-area band, computes the
packing-margin radius η(S) by bisection on the nester (op-counted, so this run doubles as
the §5.3 cost model), and reads off:

* **ρ̂** — the achievable inflated-nest occupancy frontier (a high percentile of the
  packed occupancy of packable subsets), which later fixes the pilot's Φ_f bands.
* **witness supply** = P(η ≥ r_f) — subsets that pack with planting margin.
* **decoy supply** = P(η < r_i) — subsets unreachable by the refiner (infeasible labels).
* **band occupancy** = P(r_i ≤ η < r_f) — the marginal window that must stay thin.

**Go criteria (spec §10.0):** some cell has witness supply ≥ 10%, decoy supply ≥ 10%,
and band occupancy ≤ 15%. If no cell passes, the §2.8 viability condition fails
empirically and the pre-registered fallback (reachability/occlusion coupling) triggers —
do not weaken the criteria to force a pass.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
from shapely.geometry import box

from ..ttd_core import geometry, nesting, shapes
from ..ttd_core.counters import OpCounter
from ..ttd_core.params import MU, OP_A, OP_B, OperatingPoint
from .calibrate import CalibrationTable, calibrate

WITNESS_MIN = 0.10
DECOY_MIN = 0.10
BAND_MAX = 0.15
_RHO_PERCENTILE = 90.0


@dataclass(frozen=True)
class SigmaBand:
    """A matched-ΣA sampling band (a target total-area window)."""

    name: str
    lo_area_cm2: float
    hi_area_cm2: float


def sigma_bands_for(op: OperatingPoint, k: int, n_bands: int = 3) -> list[SigmaBand]:
    """Split the achievable ΣA range for k members into ``n_bands`` equal bands."""
    total_lo = k * op.member_area_lo_cm2
    total_hi = k * op.member_area_hi_cm2
    width = (total_hi - total_lo) / n_bands
    names = (
        ["low", "mid", "high"] if n_bands == 3 else [f"b{i}" for i in range(n_bands)]
    )
    return [
        SigmaBand(names[i], total_lo + i * width, total_lo + (i + 1) * width)
        for i in range(n_bands)
    ]


@dataclass(frozen=True)
class CellResult:
    """Landscape summary + go verdict for one (OP, band, c_v) cell (spec §10.0)."""

    op_name: str
    band_name: str
    c_v_cm: float
    r_i_cm: float
    r_f_cm: float
    n_subsets: int
    n_packable: int
    rho_hat: float
    eta_median: float
    eta_iqr: float
    witness_supply: float
    decoy_supply: float
    band_occupancy: float
    timed_out_frac: float
    passed: bool


@dataclass
class PrePilotReport:
    """Full pre-pilot outcome across all cells (spec §10.0)."""

    cells: list[CellResult]
    op_counts: dict[str, int] = field(default_factory=dict)
    calibrated_cost_s: float = 0.0
    go: bool = False
    passing_cell: CellResult | None = None


def member_pool(op: OperatingPoint, n_pool: int, seed: int) -> list[shapes.Shape]:
    """Generate ``n_pool`` member shapes with areas in the operating point's band."""
    rng = np.random.default_rng(seed)
    pool: list[shapes.Shape] = []
    for i in range(n_pool):
        area = float(rng.uniform(op.member_area_lo_cm2, op.member_area_hi_cm2))
        pool.append(shapes.generate_shape_retry(seed * 100_003 + i, area))
    return pool


def sample_k_subsets(
    areas: Sequence[float],
    k: int,
    band: SigmaBand,
    n_samples: int,
    seed: int,
    *,
    max_attempts_factor: int = 200,
) -> list[tuple[int, ...]]:
    """Rejection-sample ``n_samples`` k-subsets whose ΣA falls in ``band`` (spec
    §10.0)."""
    rng = np.random.default_rng(seed)
    area_arr = np.asarray(areas, dtype=np.float64)
    n = len(area_arr)
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    cap = n_samples * max_attempts_factor
    while len(out) < n_samples and attempts < cap:
        attempts += 1
        idx = tuple(sorted(int(v) for v in rng.choice(n, size=k, replace=False)))
        if idx in seen:
            continue
        total = float(area_arr[list(idx)].sum())
        if band.lo_area_cm2 <= total <= band.hi_area_cm2:
            seen.add(idx)
            out.append(idx)
    return out


def compute_etas(
    subsets: Sequence[tuple[int, ...]],
    pool: Sequence[shapes.Shape],
    container: "geometry.Polygon",
    cfg: nesting.NesterConfig,
    *,
    r_hi: float,
    tol: float,
    counter: OpCounter | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute η and packed occupancy for each subset (spec §10.0).

    Returns ``(etas, packed_occupancy, timed_out_frac)``. ``packed_occupancy[i]`` is the
    inflated occupancy Σ Ã(sⱼ, ηᵢ)/(W·H) at the subset's own margin (NaN when it cannot
    pack at r = 0).
    """
    tray_area = float(container.area)
    etas = np.empty(len(subsets), dtype=np.float64)
    occ = np.full(len(subsets), np.nan, dtype=np.float64)
    n_timeout = 0
    for i, idx in enumerate(subsets):
        polys = [pool[j].polygon() for j in idx]
        eta, timed_out = nesting.packing_margin_radius(
            polys, container, cfg, r_hi=r_hi, tol=tol, counter=counter
        )
        etas[i] = eta
        n_timeout += int(timed_out)
        if np.isfinite(eta) and eta >= 0.0:
            packed = sum(geometry.inflated_area(p, max(eta, 0.0)) for p in polys)
            occ[i] = packed / tray_area
    timed_out_frac = n_timeout / max(1, len(subsets))
    return etas, occ, timed_out_frac


def go_criteria(
    etas: np.ndarray, r_i: float, r_f: float
) -> tuple[float, float, float, bool]:
    """Witness / decoy / band supplies and the pass verdict for one cell (spec
    §10.0)."""
    if etas.size == 0:  # no subset landed in this band → no evidence, cannot pass
        return 0.0, 0.0, 0.0, False
    finite_or_neg = etas  # -inf entries count as "unpackable" → decoy side
    witness = float(np.mean(finite_or_neg >= r_f))
    decoy = float(np.mean(finite_or_neg < r_i))
    band = float(np.mean((finite_or_neg >= r_i) & (finite_or_neg < r_f)))
    passed = (witness >= WITNESS_MIN) and (decoy >= DECOY_MIN) and (band <= BAND_MAX)
    return witness, decoy, band, passed


@dataclass(frozen=True)
class BandLandscape:
    """The c_v-independent η landscape of one (OP, band) — computed once (spec
    §10.0)."""

    etas: np.ndarray
    occ: np.ndarray
    timed_out_frac: float
    n_subsets: int


def run_band(
    op: OperatingPoint,
    band: SigmaBand,
    pool: Sequence[shapes.Shape],
    *,
    k: int,
    n_subsets: int,
    cfg: nesting.NesterConfig,
    r_hi: float,
    tol: float,
    seed: int,
    counter: OpCounter | None = None,
) -> BandLandscape:
    """Sample subsets and compute the η landscape for one (OP, band) (spec §10.0).

    η is intrinsic to the shapes and tray (independent of the clearance dial c_v), so it
    is computed once here and re-thresholded per c_v by :func:`cell_from_landscape`.
    """
    container = box(0.0, 0.0, op.tray_w_cm, op.tray_h_cm)
    areas = [s.area_cm2 for s in pool]
    subsets = sample_k_subsets(areas, k, band, n_subsets, seed)
    etas, occ, timed_out_frac = compute_etas(
        subsets, pool, container, cfg, r_hi=r_hi, tol=tol, counter=counter
    )
    return BandLandscape(etas, occ, timed_out_frac, len(subsets))


def cell_from_landscape(
    op: OperatingPoint,
    band_name: str,
    c_v_cm: float,
    landscape: BandLandscape,
) -> CellResult:
    """Evaluate the go criteria for one c_v against a precomputed landscape (spec
    §10.0)."""
    r_i = c_v_cm / 2.0
    r_f = c_v_cm / 2.0 + MU
    witness, decoy, band_occ, passed = go_criteria(landscape.etas, r_i, r_f)
    packable = landscape.occ[~np.isnan(landscape.occ)]
    rho_hat = float(np.percentile(packable, _RHO_PERCENTILE)) if packable.size else 0.0
    finite = landscape.etas[np.isfinite(landscape.etas)]
    eta_median = float(np.median(finite)) if finite.size else float("-inf")
    eta_iqr = (
        float(np.percentile(finite, 75) - np.percentile(finite, 25))
        if finite.size
        else 0.0
    )
    return CellResult(
        op_name=op.name,
        band_name=band_name,
        c_v_cm=c_v_cm,
        r_i_cm=r_i,
        r_f_cm=r_f,
        n_subsets=landscape.n_subsets,
        n_packable=int(packable.size),
        rho_hat=rho_hat,
        eta_median=eta_median,
        eta_iqr=eta_iqr,
        witness_supply=witness,
        decoy_supply=decoy,
        band_occupancy=band_occ,
        timed_out_frac=landscape.timed_out_frac,
        passed=passed,
    )


def run_cell(
    op: OperatingPoint,
    band: SigmaBand,
    c_v_cm: float,
    pool: Sequence[shapes.Shape],
    *,
    k: int,
    n_subsets: int,
    cfg: nesting.NesterConfig,
    r_hi: float,
    tol: float,
    seed: int,
    counter: OpCounter | None = None,
) -> CellResult:
    """Compute one full cell (landscape + single c_v verdict) for spec §10.0."""
    landscape = run_band(
        op,
        band,
        pool,
        k=k,
        n_subsets=n_subsets,
        cfg=cfg,
        r_hi=r_hi,
        tol=tol,
        seed=seed,
        counter=counter,
    )
    return cell_from_landscape(op, band.name, c_v_cm, landscape)


def run_pre_pilot(
    *,
    ops: Sequence[OperatingPoint] = (OP_A, OP_B),
    c_vs: Sequence[float] = (1.2, 1.6),
    k: int = 5,
    n_subsets: int = 200,
    n_pool: int = 40,
    n_bands: int = 3,
    cfg: nesting.NesterConfig | None = None,
    r_hi: float = 2.0,
    tol: float = 0.05,
    seed: int = 0,
    calibration: CalibrationTable | None = None,
    shard_index: int = 0,
    n_shards: int = 1,
) -> PrePilotReport:
    """Run the §10.0 pre-pilot over all cells and decide GO/NO-GO (spec §10.0).

    ``go`` is True iff some cell meets all three criteria; ``passing_cell`` is the first
    such cell (which fixes the operating point and pilot grid). The (OP, band)
    landscapes are independent, so a SLURM array shards them: with ``n_shards > 1`` this
    process runs only the (OP, band) work items whose index ≡ ``shard_index`` (mod
    ``n_shards``); merge the partial reports afterward with :func:`merge_reports`.
    """
    nester_cfg = (
        cfg if cfg is not None else nesting.NesterConfig.anytime(node_cap=3_000)
    )
    counter = OpCounter()
    cells: list[CellResult] = []
    pool_cache: dict[str, list[shapes.Shape]] = {}
    work = [(op, band) for op in ops for band in sigma_bands_for(op, k, n_bands)]
    for idx, (op, band) in enumerate(work):
        if n_shards > 1 and idx % n_shards != shard_index:
            continue
        if op.name not in pool_cache:
            pool_cache[op.name] = member_pool(op, n_pool, seed)
        landscape = run_band(
            op,
            band,
            pool_cache[op.name],
            k=k,
            n_subsets=n_subsets,
            cfg=nester_cfg,
            r_hi=r_hi,
            tol=tol,
            seed=seed,
            counter=counter,
        )
        for c_v in c_vs:
            cells.append(cell_from_landscape(op, band.name, c_v, landscape))
    passing = next((c for c in cells if c.passed), None)
    table = calibration if calibration is not None else None
    cost_s = table.cost_us(counter) / 1e6 if table is not None else 0.0
    return PrePilotReport(
        cells=cells,
        op_counts=dict(counter.c_ops),
        calibrated_cost_s=cost_s,
        go=passing is not None,
        passing_cell=passing,
    )


def format_report(report: PrePilotReport) -> str:
    """Render a pre-pilot report as a human-readable table (spec §10.0)."""
    header = (
        f"{'OP':5} {'band':5} {'c_v':4} {'wit':>6} {'dec':>6} {'bnd':>6} "
        f"{'rho^':>6} {'etaMd':>6} {'TO':>5} {'pass':>5}"
    )
    lines = [header, "-" * len(header)]
    for c in report.cells:
        lines.append(
            f"{c.op_name:5} {c.band_name:5} {c.c_v_cm:4.1f} "
            f"{c.witness_supply:6.2f} {c.decoy_supply:6.2f} {c.band_occupancy:6.2f} "
            f"{c.rho_hat:6.2f} {c.eta_median:6.2f} {c.timed_out_frac:5.2f} "
            f"{'YES' if c.passed else 'no':>5}"
        )
    verdict = "GO" if report.go else "NO-GO"
    lines.append("")
    lines.append(f"Verdict: {verdict}")
    if report.passing_cell is not None:
        pc = report.passing_cell
        lines.append(f"Passing cell: {pc.op_name}/{pc.band_name}/c_v={pc.c_v_cm}")
    lines.append(f"Calibrated geometric cost: {report.calibrated_cost_s:.2f} s")
    return "\n".join(lines)


def report_to_json(report: PrePilotReport) -> str:
    """Serialize a report to JSON (cells + summary)."""
    payload = {
        "cells": [asdict(c) for c in report.cells],
        "op_counts": report.op_counts,
        "calibrated_cost_s": report.calibrated_cost_s,
        "go": report.go,
        "passing_cell": asdict(report.passing_cell) if report.passing_cell else None,
    }
    return json.dumps(payload, indent=2)


def report_from_json(text: str) -> PrePilotReport:
    """Rebuild a :class:`PrePilotReport` from :func:`report_to_json` output."""
    data = json.loads(text)
    cells = [CellResult(**c) for c in data["cells"]]
    pc = data.get("passing_cell")
    return PrePilotReport(
        cells=cells,
        op_counts=data.get("op_counts", {}),
        calibrated_cost_s=data.get("calibrated_cost_s", 0.0),
        go=data.get("go", False),
        passing_cell=CellResult(**pc) if pc else None,
    )


def merge_reports(
    reports: Sequence[PrePilotReport], *, calibration: CalibrationTable | None = None
) -> PrePilotReport:
    """Merge sharded pre-pilot reports into one, re-deciding GO/NO-GO (spec §10.0)."""
    cells = [c for r in reports for c in r.cells]
    op_counts: dict[str, int] = {}
    for r in reports:
        for kind, n in r.op_counts.items():
            op_counts[kind] = op_counts.get(kind, 0) + n
    if calibration is not None:
        cost_s = calibration.cost_us(OpCounter(c_ops=dict(op_counts))) / 1e6
    else:
        cost_s = sum(r.calibrated_cost_s for r in reports)
    passing = next((c for c in cells if c.passed), None)
    return PrePilotReport(
        cells=cells,
        op_counts=op_counts,
        calibrated_cost_s=cost_s,
        go=passing is not None,
        passing_cell=passing,
    )


def _cmd_run(args: argparse.Namespace) -> None:
    """Run one shard (or the whole grid) of the pre-pilot."""
    ops = {"A": (OP_A,), "B": (OP_B,), "both": (OP_A, OP_B)}[args.op]
    cfg = nesting.NesterConfig(
        rot_grid_deg=args.rot_grid_deg,
        n_restarts=1,
        node_cap=args.node_cap,
        seed=args.seed,
    )
    table = None if args.no_calibrate else calibrate()
    report = run_pre_pilot(
        ops=ops,
        k=args.k,
        n_subsets=args.n_subsets,
        n_pool=args.n_pool,
        cfg=cfg,
        tol=args.tol,
        seed=args.seed,
        calibration=table,
        shard_index=args.shard_index,
        n_shards=args.n_shards,
    )
    print(format_report(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report_to_json(report))


def _cmd_merge(args: argparse.Namespace) -> None:
    """Merge sharded partial reports into one and re-decide GO/NO-GO."""
    reports = []
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as handle:
            reports.append(report_from_json(handle.read()))
    table = None if args.no_calibrate else calibrate()
    merged = merge_reports(reports, calibration=table)
    print(format_report(merged))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report_to_json(merged))


def main() -> None:
    """CLI entry point for the pre-pilot (standalone; no Hydra)."""
    parser = argparse.ArgumentParser(description="TTD §10.0 pre-pilot (go/no-go).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run one shard (or the whole grid)")
    run_p.add_argument("--n-subsets", type=int, default=200)
    run_p.add_argument("--n-pool", type=int, default=40)
    run_p.add_argument("--k", type=int, default=5)
    run_p.add_argument("--node-cap", type=int, default=5_000)
    run_p.add_argument("--rot-grid-deg", type=float, default=10.0)
    run_p.add_argument("--tol", type=float, default=0.05)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--op", choices=["A", "B", "both"], default="both")
    run_p.add_argument("--shard-index", type=int, default=0)
    run_p.add_argument("--n-shards", type=int, default=1)
    run_p.add_argument("--out", type=str, default="")
    run_p.add_argument("--no-calibrate", action="store_true")

    merge_p = sub.add_parser("merge", help="merge sharded reports and re-decide")
    merge_p.add_argument("--inputs", nargs="+", required=True)
    merge_p.add_argument("--out", type=str, default="")
    merge_p.add_argument("--no-calibrate", action="store_true")

    args = parser.parse_args()
    if args.cmd == "merge":
        _cmd_merge(args)
    else:
        _cmd_run(args)


if __name__ == "__main__":
    main()
