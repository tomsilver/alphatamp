"""PIGINet training harness (Step 7 of docs/piginet_dd2d_plan.md).

Trains the model on the DD2D dataset under the 50:1 class imbalance, with the imbalance
strategy selected **empirically** by the Step-7 gate: train arms {weighted_bce, focal,
focal+ranking}, and pick the one with the lowest **val rollout-FP** (the arbiter; ties → val
AUPRC). The winner must beat weighted_bce on rollout-FP and AUPRC and hit the recall target.

Metrics: AUPRC / AUROC / balanced-acc (classification) + **val rollout-FP proxy** — per val
problem, rank its collected plans by model score, FP = #negatives scored >= the positive
(exactly the refinements the model would save over the astar-dist Baseline on those plans; a
faithful, refinement-free proxy for the Step-8 full-pool rollout FP). Also fits a temperature
and a high-recall discard threshold on val for Step 8.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m blocks_tamp.piginet.train --tiny
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m blocks_tamp.piginet.train --arm compare
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PIGINetDataset, collate, precompute_clip_cache
from .encoders import Encoders
from .losses import focal_loss, listwise_ranking_loss, weighted_bce
from .model import PIGINet

ARMS = ("weighted_bce", "focal", "focal_rank")


# --------------------------------------------------------------------------- #
# W&B (opt-in, graceful no-op)
# --------------------------------------------------------------------------- #
class _WandB:
    """Thin wrapper: a no-op unless ``--wandb`` is set, ``wandb`` imports, and
    ``WANDB_API_KEY`` is in the environment (the key is NOT stored in the repo)."""

    def __init__(self, args, config):
        self.run = None
        self._wandb = None
        if not getattr(args, "wandb", False):
            print(
                "# W&B disabled (no --wandb); training proceeds with console logs only"
            )
            return
        if not os.environ.get("WANDB_API_KEY"):
            print("# --wandb set but WANDB_API_KEY not in env; continuing WITHOUT W&B")
            return
        try:
            import wandb
        except Exception as e:  # pragma: no cover - env dependent
            print(
                f"# --wandb set but importing wandb failed ({e}); continuing WITHOUT W&B"
            )
            return
        self._wandb = wandb
        self.run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or None,
            config=config,
        )
        print(f"# W&B run: {self.run.url}")

    def log(self, data, step=None):
        if self.run is not None:
            self._wandb.log(data, step=step)

    def summary(self, data):
        if self.run is not None:
            self.run.summary.update(data)

    def finish(self):
        if self.run is not None:
            self._wandb.finish()


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _rollout_fp_group(scores, labels) -> float | None:
    """Refinements wasted before the first success when refining in DESCENDING score order:
    = #negatives ranked above the best positive, with random tie-break (0.5 per tie). Handles
    multiple positives (stop at the highest-scoring one) and discrete-score ties (length).
    """
    pos = [s for s, l in zip(scores, labels) if l > 0.5]
    if not pos:
        return None
    m = max(pos)
    strict = sum(1 for s, l in zip(scores, labels) if l < 0.5 and s > m)
    ties = sum(1 for s, l in zip(scores, labels) if l < 0.5 and s == m)
    return strict + 0.5 * ties


def _rollout_fp(logits, labels, pids) -> float:
    by = defaultdict(list)
    for lo, la, pi in zip(logits, labels, pids):
        by[pi].append((float(lo), float(la)))
    fps = [_rollout_fp_group([s for s, _ in v], [l for _, l in v]) for v in by.values()]
    fps = [f for f in fps if f is not None]
    return float(np.mean(fps)) if fps else 0.0


def _rollout_fp_by_stratum(logits, labels, pids, strata) -> dict:
    """Per-problem rollout-FP averaged within each stratum (each problem has one
    stratum).

    Returns ``{stratum: mean_fp}`` for the strata present in this split.
    """
    by = defaultdict(list)
    st_of = {}
    for lo, la, pi, st in zip(logits, labels, pids, strata):
        by[pi].append((float(lo), float(la)))
        st_of[pi] = st
    per_stratum = defaultdict(list)
    for pi, v in by.items():
        f = _rollout_fp_group([s for s, _ in v], [l for _, l in v])
        if f is not None:
            per_stratum[st_of[pi]].append(f)
    return {st: float(np.mean(v)) for st, v in per_stratum.items() if v}


def reference_baselines(data_root, split, domain=None) -> dict:
    """Non-learned rollout-FP baselines the model must beat, from the full-pool records:

    astar-order (the collection order = the deployment Baseline), and the plan-length
    heuristic (the confound). Computed once; printed as the honest bar.

    Reads through the domain rather than globbing a record tree, so an environment whose
    examples are built in memory from `EpisodeRecord` pickles gets its bar printed too --
    and gets it from the *same* examples the model trains on, not a parallel path.
    """
    if domain is None:
        from .dd2d_adapter import DD2DDomain

        domain = DD2DDomain(data_root)
    by = defaultdict(list)
    for pid, examples in domain.problems(split):
        for ex in examples:
            by[pid].append(
                (ex.provenance["plan_idx"], len(ex.task_plan), int(ex.label))
            )

    def rfp(scorer):
        fps = []
        for plans in by.values():
            f = _rollout_fp_group(
                [scorer(pi, pl) for pi, pl, _ in plans], [la for *_, la in plans]
            )
            if f is not None:
                fps.append(f)
        return float(np.mean(fps)) if fps else 0.0

    rng = np.random.default_rng(0)
    return {
        "astar_order": rfp(lambda pi, pl: -pi),  # ascending plan_idx = astar Baseline
        "length_long": rfp(lambda pi, pl: pl),  # prefer longer (the old confound)
        "length_short": rfp(lambda pi, pl: -pl),  # prefer shorter
        "random": rfp(lambda pi, pl: rng.random()),
    }


def _classification(logits, labels) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        roc_auc_score,
    )

    y = np.asarray(labels)
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    out = {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "bal_acc": float(balanced_accuracy_score(y, p >= 0.5)),
    }
    return out


def _baseline_rollout_fp(labels, pids) -> float:
    """Baseline (astar-dist collection order): the positive is last, so FP =
    #negatives."""
    by = defaultdict(int)
    pos = defaultdict(bool)
    for la, pi in zip(labels, pids):
        if la > 0.5:
            pos[pi] = True
        else:
            by[pi] += 1
    return float(np.mean([by[p] for p in pos])) if pos else 0.0


# --------------------------------------------------------------------------- #
# eval / calibration
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    logits, labels, pids, strata = [], [], [], []
    for batch in loader:
        lg, gids, lb = model(batch)
        # map within-batch group ids -> problem ids; model preserves group/record order,
        # so per-record stratum aligns with the flattened logits.
        pid_of_group = [g["pid"] for g in batch]
        logits += lg.detach().cpu().tolist()
        labels += lb.detach().cpu().tolist()
        pids += [pid_of_group[int(g)] for g in gids.detach().cpu().tolist()]
        for g in batch:
            strata += [rec.get("stratum") for rec in g["records"]]
    cls = _classification(logits, labels)
    cls["rollout_fp"] = _rollout_fp(logits, labels, pids)
    cls["rollout_fp_by_stratum"] = _rollout_fp_by_stratum(logits, labels, pids, strata)
    cls["baseline_fp"] = _baseline_rollout_fp(labels, pids)
    cls["_logits"], cls["_labels"] = logits, labels
    return cls


def fit_temperature(logits, labels) -> float:
    lg = torch.tensor(logits)
    y = torch.tensor(labels)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def pick_threshold(logits, labels, T, target_recall=0.98) -> float:
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits) / T))
    pos = p[np.asarray(labels) > 0.5]
    if len(pos) == 0:
        return 0.5
    return float(
        np.quantile(pos, 1.0 - target_recall)
    )  # keeps ~target_recall of positives


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _loss_fn(arm, pos_weight, lam=1.0):
    def fn(logits, gids, labels):
        if arm == "weighted_bce":
            return weighted_bce(logits, labels, pos_weight)
        if arm == "focal":
            return focal_loss(logits, labels)
        if arm == "focal_rank":
            return focal_loss(logits, labels) + lam * listwise_ranking_loss(
                logits, gids, labels
            )
        raise ValueError(arm)

    return fn


def train_arm(
    arm,
    train_ds,
    val_ds,
    enc,
    device,
    epochs,
    batch_problems,
    lr,
    pos_weight,
    patience=10,
    dropout=0.2,
    feat_noise=0.08,
    weight_decay=2e-2,
    select="rollout_fp",
    warmup_epochs=2,
    rank_lambda=1.0,
    on_epoch=None,
    verbose=True,
) -> dict:
    """Selection/early-stop on **val rollout-FP** (the arbiter; lower is better), NOT
    AUPRC — AUPRC anti-correlates with the deployment metric under this imbalance.

    ``on_epoch(ep, vm, trainloss, lr)`` is an optional per-epoch callback (used to
    stream metrics to W&B).
    """
    model = PIGINet(enc, device=device, dropout=dropout, feat_noise=feat_noise)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = _loss_fn(arm, pos_weight, lam=rank_lambda)
    tl = DataLoader(
        train_ds, batch_size=batch_problems, shuffle=True, collate_fn=collate
    )
    vl = DataLoader(
        val_ds, batch_size=batch_problems, shuffle=False, collate_fn=collate
    )

    def better(
        vm, best
    ):  # lower rollout_fp better (tie-break higher AUPRC); AUPRC arm for --tiny
        if select == "auprc":
            return vm["auprc"] > best.get("auprc", -1)
        return (vm["rollout_fp"], -vm["auprc"]) < (
            best.get("rollout_fp", 1e9),
            -best.get("auprc", -1),
        )

    best, best_state, best_ll, bad = {}, None, None, 0
    last_vm = None
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for batch in tl:
            opt.zero_grad()
            logits, gids, labels = model(batch)
            loss = loss_fn(logits, gids, labels)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        vm = evaluate(model, vl)
        trainloss = tot / max(len(tl), 1)
        if verbose:
            by = vm["rollout_fp_by_stratum"]
            strat_str = " ".join(f"s{s}={by[s]:.1f}" for s in sorted(by))
            print(
                f"  [{arm}] ep {ep:02d} trainloss {trainloss:.4f} | "
                f"val AUPRC {vm['auprc']:.3f} AUROC {vm['auroc']:.3f} "
                f"rolloutFP {vm['rollout_fp']:.2f} (base {vm['baseline_fp']:.1f}) [{strat_str}]",
                flush=True,
            )
        if on_epoch is not None:
            on_epoch(ep, vm, trainloss, cur_lr)
        last_vm = vm
        if ep >= warmup_epochs and better(
            vm, best
        ):  # ignore under-trained early epochs
            best = {k: v for k, v in vm.items() if not k.startswith("_")}
            best_ll = (vm["_logits"], vm["_labels"])
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        elif best:
            bad += 1
            if bad >= patience:
                break
    if (
        best_state is None
    ):  # never beat warmup (e.g. --epochs 1 smoke): use the last epoch
        best = {k: v for k, v in last_vm.items() if not k.startswith("_")}
        best_ll = (last_vm["_logits"], last_vm["_labels"])
    else:
        model.load_state_dict(best_state)
    T = fit_temperature(*best_ll)
    thr = pick_threshold(*best_ll, T)
    best.update(temperature=T, threshold=thr, arm=arm)
    return {"model": model, "metrics": best}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _pos_weight(ds) -> float:
    n_pos = n_neg = 0
    for g in ds.groups:
        for r in g["records"]:
            n_pos += r["label"] > 0.5
            n_neg += r["label"] < 0.5
    return float(n_neg) / max(n_pos, 1)


def _slim_state_dict(model) -> dict:
    """Drop the frozen CLIP params (``enc.clip.*``, ~605MB) from the checkpoint.

    On load the Encoders reconstruct CLIP from the pretrained weights, so
    ``load_state_dict(strict=False)`` restores identical behaviour — the ckpt shrinks
    617MB -> ~12MB (only the trainable MLPs + transformer + head).
    """
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("enc.clip.")
    }


def _build_domain(args):
    """The environment adapter this run trains against.

    DD2D is the default so every existing command line is unchanged. The adapters differ
    in more than data location: their value normalisers are in different units, and using
    the wrong one silently collapses every shape feature toward zero rather than raising
    (measured: StickButton2D shapes read |mean| 0.006 against DD2D's centimetre divisors,
    0.372 against its own).
    """
    if args.domain == "stickbutton2d":
        from .sb2d_adapter import make_sb2d_domain

        # Factory picks the crop source by variant: kinder-rendered PNGs for
        # `stickbutton2d_v1_kinder`, the schematic rasteriser otherwise.
        return make_sb2d_domain(args.data_root, args.env_variant)
    if args.domain == "restock3d":
        from .restock_adapter import make_restock_domain

        # Crops are the env's own oblique render (reconstructed from the seed), so a tall
        # block is visually distinct from a cube -- the F3 signal a footprint crop loses.
        return make_restock_domain(args.data_root, args.env_variant)
    from .dd2d_adapter import DD2DDomain

    return DD2DDomain(args.data_root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", default=os.path.join("data", "dd2d", "raw"))
    ap.add_argument(
        "--domain",
        default="dd2d",
        choices=("dd2d", "stickbutton2d", "restock3d"),
        help="which environment's adapter supplies the vocabulary, the value "
        "normalisers and the split. Defaults to dd2d, so every pre-2026-08-01 command "
        "line trains exactly what it trained before.",
    )
    ap.add_argument(
        "--env-variant",
        default="stickbutton2d_v1",
        help="collection the stickbutton2d domain reads (ignored for dd2d)",
    )
    ap.add_argument(
        "--cache-dir", default=os.path.join("data", "dd2d", "out_dd2d", "clip_cache")
    )
    ap.add_argument(
        "--out", default=os.path.join("data", "dd2d", "out_dd2d", "piginet")
    )
    ap.add_argument("--arm", default="compare", choices=("compare",) + ARMS)
    ap.add_argument(
        "--select",
        default="rollout_fp",
        choices=("rollout_fp", "auprc"),
        help="checkpoint/early-stop metric. rollout_fp = deployment arbiter (default); "
        "auprc = paper-faithful classification selection.",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--k", type=int, default=16, help="max negatives kept per problem")
    ap.add_argument("--batch-problems", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--rank-lambda",
        type=float,
        default=1.0,
        help="weight on the listwise ranking term in focal_rank",
    )
    ap.add_argument(
        "--patience", type=int, default=10, help="early-stop patience (val rollout-FP)"
    )
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--feat-noise", type=float, default=0.08)
    ap.add_argument("--weight-decay", type=float, default=2e-2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="training seed: torch init + the dataset's negative subsampling. Without "
        "it every run is identical, so a multi-seed spread would be fabricated rather "
        "than measured.",
    )
    ap.add_argument(
        "--tiny", action="store_true", help="20-problem overfit sanity gate"
    )
    ap.add_argument(
        "--train-strata",
        type=int,
        nargs="*",
        default=None,
        help="restrict train AND val to these stratum indices (0-3; empty/omitted = "
        "all). Held-out-stratum generalization: --train-strata 0 1 2 trains on s0-s2 "
        "(DD2D) / b1-b3 (SB2D) and, because val is filtered too, never lets the "
        "checkpoint selector see the held-out stratum.",
    )
    # W&B (opt-in; auth via WANDB_API_KEY env var, never stored in the repo)
    ap.add_argument(
        "--wandb", action="store_true", help="log curves to Weights & Biases"
    )
    ap.add_argument("--wandb-project", default="piginet-dd2d")
    ap.add_argument("--wandb-entity", default="josephxu-lilliput")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    domain = _build_domain(args)
    enc = Encoders(device=args.device, domain=domain)
    print(
        f"# domain={domain.name} frame={domain.frame_extent} "
        f"shape_max={list(domain.shape_max)}"
    )
    print("# precomputing CLIP caches (frozen, one-time) …")
    for sp in ("train", "val"):
        precompute_clip_cache(args.data_root, sp, enc, args.cache_dir, domain=domain)

    if args.tiny:
        import glob

        pids = [
            os.path.basename(p)
            for p in sorted(glob.glob(os.path.join(args.data_root, "train", "dd2d_*")))[
                :20
            ]
        ]
        tds = PIGINetDataset(
            args.data_root, "train", args.cache_dir, subsample_k=0, problem_ids=pids
        )
        res = train_arm(
            "weighted_bce",
            tds,
            tds,
            enc,
            args.device,
            epochs=60,
            batch_problems=args.batch_problems,
            lr=3e-4,
            pos_weight=_pos_weight(tds),
            select="auprc",
            dropout=0.0,
            feat_noise=0.0,
            warmup_epochs=0,
        )
        m = res["metrics"]
        print(
            f"# TINY-OVERFIT: train AUPRC {m['auprc']:.3f} rolloutFP {m['rollout_fp']:.2f} "
            f"(baseline {m['baseline_fp']:.1f}) -> {'PASS' if m['auprc'] > 0.98 else 'FAIL'}"
        )
        return 0

    # Both sources of randomness, seeded together: torch's parameter init and the
    # dataset's choice of which negatives to keep. `vds` keeps every candidate
    # (`subsample_k=0`), so its seed is inert -- passed anyway so the two constructions
    # do not diverge if that default ever changes.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # None (default) keeps every stratum; an empty --train-strata is treated as None too.
    keep_strata = set(args.train_strata) if args.train_strata else None
    tds = PIGINetDataset(
        args.data_root,
        "train",
        args.cache_dir,
        subsample_k=args.k,
        seed=args.seed,
        domain=domain,
        keep_strata=keep_strata,
    )
    vds = PIGINetDataset(
        args.data_root,
        "val",
        args.cache_dir,
        subsample_k=0,
        seed=args.seed,
        domain=domain,
        keep_strata=keep_strata,
    )
    pw = _pos_weight(tds)
    print(f"# train groups {len(tds)} | val groups {len(vds)} | pos_weight {pw:.1f}")
    base = reference_baselines(args.data_root, "val", domain)
    print(
        f"# VAL rollout-FP baselines to beat: astar-order {base['astar_order']:.2f} | "
        f"length-long {base['length_long']:.2f} | length-short {base['length_short']:.2f} | "
        f"random {base['random']:.2f}  (lower is better; the model must beat these)"
    )

    wb = _WandB(
        args,
        config={
            "arm": args.arm,
            "epochs": args.epochs,
            "k": args.k,
            "rank_lambda": args.rank_lambda,
            "patience": args.patience,
            "lr": args.lr,
            "dropout": args.dropout,
            "feat_noise": args.feat_noise,
            "weight_decay": args.weight_decay,
            "batch_problems": args.batch_problems,
            "data_root": args.data_root,
            "train_groups": len(tds),
            "val_groups": len(vds),
            "pos_weight": pw,
            "baseline_astar": base["astar_order"],
            "baseline_length_long": base["length_long"],
            "baseline_random": base["random"],
        },
    )

    def make_logger(arm):
        # constant baseline series -> flat reference lines on the same rollout-FP chart
        def _log(ep, vm, trainloss, lr):
            d = {
                f"{arm}/train_loss": trainloss,
                f"{arm}/lr": lr,
                f"{arm}/val_auprc": vm["auprc"],
                f"{arm}/val_auroc": vm["auroc"],
                f"{arm}/val_rollout_fp": vm["rollout_fp"],
                f"{arm}/val_baseline_fp": vm["baseline_fp"],
                f"{arm}/ref_astar": base["astar_order"],
                f"{arm}/ref_length_long": base["length_long"],
                f"{arm}/ref_random": base["random"],
            }
            for st, fp in vm["rollout_fp_by_stratum"].items():
                d[f"{arm}/val_rollout_fp_s{st}"] = fp
            wb.log(d, step=ep)

        return _log

    arms = ARMS if args.arm == "compare" else (args.arm,)
    results = {}
    for arm in arms:
        print(f"\n# === arm: {arm} ===")
        r = train_arm(
            arm,
            tds,
            vds,
            enc,
            args.device,
            args.epochs,
            args.batch_problems,
            args.lr,
            pw,
            patience=args.patience,
            dropout=args.dropout,
            feat_noise=args.feat_noise,
            weight_decay=args.weight_decay,
            select=args.select,
            rank_lambda=args.rank_lambda,
            on_epoch=make_logger(arm),
        )
        results[arm] = r["metrics"]
        torch.save(
            {
                "state_dict": _slim_state_dict(r["model"]),
                **r["metrics"],
                "obj_channels": enc.obj_channels,
                "n_max": r["model"].n_max,
            },
            os.path.join(args.out, f"ckpt_{arm}.pt"),
        )

    # cross-arm select: by --select. rollout_fp = the deployment arbiter (default);
    # auprc = paper-faithful classification metric. For a single --arm run the winner
    # is trivially that arm, so --select only governs the in-arm early-stop above.
    base = results.get("weighted_bce")
    if args.select == "auprc":
        winner = max(results, key=lambda a: results[a]["auprc"])
        worse = (
            base
            and winner != "weighted_bce"
            and (results[winner]["auprc"] < base["auprc"])
        )
    else:
        winner = min(
            results, key=lambda a: (results[a]["rollout_fp"], -results[a]["auprc"])
        )
        worse = (
            base
            and winner != "weighted_bce"
            and (results[winner]["rollout_fp"] > base["rollout_fp"])
        )
    if worse:
        print(
            f"# gate: {winner} did not beat weighted_bce on {args.select} "
            "-> fall back to weighted_bce"
        )
        winner = "weighted_bce"
    print(f"\n# SELECTED arm: {winner} (by val {args.select})")
    for a, m in results.items():
        print(
            f"  {a:12s}: AUPRC {m['auprc']:.3f} AUROC {m['auroc']:.3f} "
            f"rolloutFP {m['rollout_fp']:.2f} (baseline {m['baseline_fp']:.1f}) thr {m['threshold']:.3f}"
        )
    import shutil

    shutil.copyfile(
        os.path.join(args.out, f"ckpt_{winner}.pt"), os.path.join(args.out, "ckpt.pt")
    )
    with open(os.path.join(args.out, "train_metrics.json"), "w") as f:
        json.dump(
            {
                "selected": winner,
                "seed": args.seed,
                "arms": results,
                "baseline_rollout_fp": results[winner]["baseline_fp"],
            },
            f,
            indent=2,
        )
    print(f"# wrote {args.out}/ckpt.pt + train_metrics.json")

    win = results[winner]
    wb.summary(
        {
            "selected_arm": winner,
            "val_rollout_fp": win["rollout_fp"],
            "val_auprc": win["auprc"],
            "val_auroc": win["auroc"],
            "temperature": win["temperature"],
            "threshold": win["threshold"],
            **{
                f"val_rollout_fp_s{st}": fp
                for st, fp in win["rollout_fp_by_stratum"].items()
            },
        }
    )
    wb.finish()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
