"""PIGINet honest eval (Step 8 of docs/piginet_dd2d_plan.md) on the unconfounded raw_v2
data.

Because full-pool collection already refined + labelled every plan in each problem's k=200
astar-dist pool, the held-out eval needs **no re-refinement** — it is score + rank + count.
Per test problem we replay several orderings and count the rollout FP (refinements wasted
before the first success):

* **Baseline** — astar-dist order (= ascending plan_idx, the collection/deployment order).
* **PIGI**     — descending model score.
* **Oracle**   — feasible-first (0 by construction; the headroom floor).
* **length**   — the plan-length heuristic (the old confound), as an explicit control.
* **random**   — reference.

Reported per stratum {0,1,2,3} and overall, with pooled AUPRC/AUROC. The honest question:
does PIGI beat the astar Baseline (and the length control), and how far is it from Oracle?
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from .dataset import PIGINetDataset, precompute_clip_cache
from .encoders import Encoders
from .model import PIGINet


def _rollout_fp_group(scores, labels) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l > 0.5]
    if not pos:
        return None
    m = max(pos)
    strict = sum(1 for s, l in zip(scores, labels) if l < 0.5 and s > m)
    ties = sum(1 for s, l in zip(scores, labels) if l < 0.5 and s == m)
    return strict + 0.5 * ties


@torch.no_grad()
def score_split(ckpt_path, data_root, cache_dir, split, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    enc = Encoders(
        device=device,
        obj_channels=tuple(ckpt.get("obj_channels", ("img", "pose", "shape"))),
    )
    model = PIGINet(enc, device=device, n_max=int(ckpt.get("n_max", 64)))
    # strict=False: slimmed ckpts omit the frozen CLIP params (enc.clip.*), which the fresh
    # Encoders already restored from the pretrained weights. Full (legacy) ckpts also load fine.
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    precompute_clip_cache(data_root, split, enc, cache_dir)
    ds = PIGINetDataset(data_root, split, cache_dir, subsample_k=0)
    rows = []  # (pid, stratum, plan_idx, length, label, score)
    for g in ds.groups:
        logits, _, _ = model([g])
        for rec, lo in zip(g["records"], logits.tolist()):
            rows.append(
                (
                    g["pid"],
                    rec["stratum"],
                    rec["plan_idx"],
                    rec["length"],
                    rec["label"],
                    float(lo),
                )
            )
    return rows, float(ckpt.get("threshold", 0.5)), float(ckpt.get("temperature", 1.0))


def evaluate(
    ckpt_path,
    data_root,
    cache_dir,
    split="test",
    device="cpu",
    out_dir="out_dd2d/piginet_eval",
):
    from sklearn.metrics import average_precision_score, roc_auc_score

    os.makedirs(out_dir, exist_ok=True)
    rows, thr, temp = score_split(ckpt_path, data_root, cache_dir, split, device)

    by = defaultdict(list)
    strat = {}
    for pid, st, pi, ln, la, sc in rows:
        by[pid].append((pi, ln, la, sc))
        strat[pid] = st
    rng = np.random.default_rng(0)
    orderings = {
        "Baseline(astar)": lambda pi, ln, la, sc: -pi,
        "PIGI(model)": lambda pi, ln, la, sc: sc,
        "length_long": lambda pi, ln, la, sc: ln,
        "random": lambda pi, ln, la, sc: rng.random(),
        "Oracle": lambda pi, ln, la, sc: la,  # feasible-first -> 0 FP
    }
    # per-problem rollout-FP for each ordering
    per = {name: defaultdict(list) for name in orderings}  # name -> stratum -> [fp]
    for pid, plans in by.items():
        st = strat[pid]
        for name, scorer in orderings.items():
            f = _rollout_fp_group([scorer(*p) for p in plans], [p[2] for p in plans])
            if f is not None:
                per[name][st].append(f)
                per[name]["all"].append(f)

    strata = sorted({st for st in strat.values()})

    def mean(name, key):
        v = per[name][key]
        return float(np.mean(v)) if v else float("nan")

    # print table
    print(f"\n# === Honest rollout-FP on {split} (lower = better; Oracle=0 floor) ===")
    hdr = f"{'ordering':18s}" + "".join(f"  s{s:>1}" for s in strata) + f"  {'ALL':>6}"
    print(hdr)
    for name in orderings:
        line = (
            f"{name:18s}"
            + "".join(f" {mean(name, s):5.1f}" for s in strata)
            + f"  {mean(name,'all'):6.2f}"
        )
        print(line)

    # classification metrics (pooled)
    y = np.array([r[4] for r in rows])
    p = 1.0 / (1.0 + np.exp(-np.array([r[5] for r in rows]) / temp))
    auprc = float(average_precision_score(y, p))
    auroc = float(roc_auc_score(y, p))
    base, pigi, orac = (
        mean("Baseline(astar)", "all"),
        mean("PIGI(model)", "all"),
        mean("Oracle", "all"),
    )
    print(f"\n# AUPRC {auprc:.3f}  AUROC {auroc:.3f}  (pooled, {len(rows)} plans)")
    print(
        f"# Baseline {base:.2f} -> PIGI {pigi:.2f}  (reduction {(1-pigi/base)*100:.0f}% vs astar; "
        f"Oracle floor 0.0; headroom PIGI-Oracle = {pigi:.2f})"
    )
    print(
        f"# vs controls: length_long {mean('length_long','all'):.2f}, random {mean('random','all'):.2f}"
    )

    # CSV + summary
    import csv

    with open(os.path.join(out_dir, f"{split}_per_problem.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "stratum"] + list(orderings))
        pf = {
            n: {
                pid: _rollout_fp_group(
                    [orderings[n](*p) for p in by[pid]], [p[2] for p in by[pid]]
                )
                for pid in by
            }
            for n in orderings
        }
        for pid in by:
            w.writerow([pid, strat[pid]] + [pf[n][pid] for n in orderings])
    summary = {
        "split": split,
        "n_plans": len(rows),
        "auprc": auprc,
        "auroc": auroc,
        "threshold": thr,
        "temperature": temp,
        "rollout_fp": {
            n: {str(k): mean(n, k) for k in list(strata) + ["all"]} for n in orderings
        },
    }
    with open(os.path.join(out_dir, f"{split}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _plot(
        per, strata, orderings, os.path.join(out_dir, f"{split}_rollout_fp.png"), split
    )
    print(f"# wrote {out_dir}/{split}_summary.json + per_problem.csv + rollout_fp.png")
    return summary


def _plot(per, strata, orderings, path, split):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Baseline(astar)", "PIGI(model)", "Oracle", "length_long", "random"]
    x = np.arange(len(strata) + 1)
    w = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, n in enumerate(names):
        vals = [np.mean(per[n][s]) if per[n][s] else 0 for s in strata] + [
            np.mean(per[n]["all"]) if per[n]["all"] else 0
        ]
        ax.bar(x + (i - 2) * w, vals, w, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in strata] + ["ALL"])
    ax.set_ylabel("rollout FP (refinements before 1st success)")
    ax.set_title(f"PIGINet on DD2D ({split}) — honest rollout-FP by stratum")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default="out_dd2d/piginet_v2/ckpt.pt")
    ap.add_argument("--data-root", default="data/dd2d/raw_v2")
    ap.add_argument("--cache-dir", default="out_dd2d/clip_cache_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="out_dd2d/piginet_eval")
    args = ap.parse_args(argv)
    evaluate(
        args.ckpt, args.data_root, args.cache_dir, args.split, args.device, args.out_dir
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
