"""Train a masked autoencoder on encoder dataset artifacts.

Target: predict masked entries of `steps_completed_fraction` (soft-label BCE).

Inputs per row:
- x_steps: revealed steps_completed_fraction values (0 for hidden)
- m: reveal mask
- A: applicability mask

Model input is concatenated [x_steps, m, A] with dimension 3M.
Model output has dimension M: logits for steps_completed_fraction.
Falls back to binary `success` when `steps_completed_fraction` is absent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import hydra
import kinder
import numpy as np
import torch
import torch.nn.functional as F
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


@dataclass(frozen=True)
class SplitTensors:
    """Tensorized split data."""

    applicability: torch.Tensor
    success: torch.Tensor
    steps_completed_fraction: (
        torch.Tensor
    )  # falls back to binary success for old datasets
    vocab: list[Any]


class EncoderMAE(nn.Module):
    """Simple MLP masked autoencoder head producing M logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("model.hidden_dims must be non-empty")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("model.dropout must be in [0, 1)")

        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
        self._network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return per-column logits for masked reconstruction."""
        return self._network(inputs)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(hydra.utils.get_original_cwd()) / path


def _bootstrap_env_model_modules(cfg: DictConfig) -> None:
    """Register dynamic modules needed for dill deserialization."""
    env_id = str(cfg.bootstrap.env_id)
    model_name = str(cfg.bootstrap.model_name)
    num_obstructions = int(cfg.bootstrap.num_obstructions)

    kinder.register_all_environments()
    env = kinder.make(env_id)
    try:
        _ = create_bilevel_planning_models(
            model_name,
            env.observation_space,
            env.action_space,
            num_obstructions=num_obstructions,
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]


def _load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as file:
        payload = dill.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def _is_binary_matrix(arr: np.ndarray) -> bool:
    return bool(np.all(np.isclose(arr, 0.0) | np.isclose(arr, 1.0)))


def _extract_split_tensors(payload: dict[str, Any], split_name: str) -> SplitTensors:
    if "dataset" not in payload:
        raise KeyError(f"{split_name}: missing 'dataset' key")
    dataset = payload["dataset"]
    if not isinstance(dataset, dict):
        raise TypeError(f"{split_name}: dataset must be a dict")

    required = {"applicability", "success", "op_sequence_vocab"}
    missing = required - set(dataset)
    if missing:
        raise KeyError(f"{split_name}: missing dataset keys: {sorted(missing)}")

    applicability = np.asarray(dataset["applicability"], dtype=np.float32)
    success = np.asarray(dataset["success"], dtype=np.float32)
    vocab = list(dataset["op_sequence_vocab"])

    if applicability.ndim != 2 or success.ndim != 2:
        raise ValueError(f"{split_name}: applicability/success must be rank-2")
    if applicability.shape != success.shape:
        raise ValueError(
            f"{split_name}: shape mismatch A{applicability.shape} vs Y{success.shape}"
        )
    if applicability.shape[1] != len(vocab):
        raise ValueError(
            f"{split_name}: column count {applicability.shape[1]} "
            f"!= vocab size {len(vocab)}"
        )

    if not _is_binary_matrix(applicability):
        raise ValueError(f"{split_name}: applicability must be binary")
    if not _is_binary_matrix(success):
        raise ValueError(f"{split_name}: success must be binary")
    if np.any(success > applicability):
        raise ValueError(f"{split_name}: found success=1 where applicability=0")

    # Rich failure features — fall back to binary success / zeros for old datasets.
    if "steps_completed_fraction" in dataset:
        steps = np.asarray(dataset["steps_completed_fraction"], dtype=np.float32)
        if steps.shape != applicability.shape:
            raise ValueError(
                f"{split_name}: steps_completed_fraction shape {steps.shape} "
                f"!= applicability shape {applicability.shape}"
            )
    else:
        steps = success.copy()

    return SplitTensors(
        applicability=torch.from_numpy(applicability),
        success=torch.from_numpy(success),
        steps_completed_fraction=torch.from_numpy(steps),
        vocab=vocab,
    )


def _assert_same_vocab(
    train_vocab: list[Any], other_vocab: list[Any], name: str
) -> None:
    if len(train_vocab) != len(other_vocab):
        raise ValueError(
            f"Vocab mismatch for {name}: len(train)={len(train_vocab)} "
            f"len({name})={len(other_vocab)}"
        )
    for index, (train_entry, other_entry) in enumerate(zip(train_vocab, other_vocab)):
        if train_entry != other_entry:
            raise ValueError(
                f"Vocab mismatch for {name} at index {index}: train and {name} differ"
            )


def _select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _sample_reveal_mask(
    applicability: torch.Tensor,
    reveal_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if reveal_probability < 0.0 or reveal_probability > 1.0:
        raise ValueError("reveal_probability must be in [0, 1]")

    # Sample on CPU for generator compatibility, then map back to input device.
    applicability_cpu = applicability.to("cpu")
    applicable_cpu = applicability_cpu > 0.5
    reveal_mask_cpu = torch.zeros_like(applicable_cpu, dtype=torch.bool)

    num_rows = applicable_cpu.shape[0]
    for row_idx in range(num_rows):
        applicable_indices = torch.nonzero(
            applicable_cpu[row_idx], as_tuple=False
        ).view(-1)
        num_applicable = int(applicable_indices.numel())

        if num_applicable == 0:
            continue

        if num_applicable == 1:
            # Degenerate case: cannot have both revealed and hidden applicable entries.
            # Keep hidden target available by revealing none.
            continue

        # For k >= 2, enforce at least 1 revealed and at least 1 hidden.
        # Bias expected reveal count by reveal_probability, then clamp to [1, k-1].
        target = torch.binomial(
            torch.tensor(float(num_applicable)),
            torch.tensor(reveal_probability),
            generator=generator,
        ).item()
        num_revealed = max(1, min(num_applicable - 1, int(target)))

        perm = torch.randperm(num_applicable, generator=generator)
        chosen = applicable_indices[perm[:num_revealed]]
        reveal_mask_cpu[row_idx, chosen] = True

    return reveal_mask_cpu.to(applicability.device)


def _masked_bce_loss(
    logits: torch.Tensor,
    targets_steps: torch.Tensor,
    applicability: torch.Tensor,
    reveal_mask: torch.Tensor,
    pos_weight: float,
) -> tuple[torch.Tensor | None, int]:
    """Soft-label BCE loss on hidden applicable entries.

    Args:
        logits: shape (B, M) — steps_completed_fraction logits.
        targets_steps: shape (B, M) — steps_completed_fraction in [0, 1].
        applicability: shape (B, M) — binary applicability mask.
        reveal_mask: shape (B, M) — True where entry is revealed (not hidden).
        pos_weight: multiplier applied to targets > 0.5 (addresses sparse successes).
    """
    hidden_applicable = (applicability > 0.5) & (~reveal_mask)
    hidden_count = int(hidden_applicable.sum().item())
    if hidden_count == 0:
        return None, 0

    hidden_logits = logits[hidden_applicable]
    hidden_targets = targets_steps[hidden_applicable]
    unreduced = F.binary_cross_entropy_with_logits(
        hidden_logits,
        hidden_targets,
        reduction="none",
    )
    # Up-weight entries where the plan made meaningful progress (target > 0.5).
    weights = torch.ones_like(hidden_targets)
    if pos_weight > 1.0:
        weights = torch.where(hidden_targets > 0.5, pos_weight, 1.0)
    loss = (unreduced * weights).mean()
    return loss, hidden_count


def _evaluate(
    model: nn.Module,
    split: "SplitTensors",
    reveal_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
    pos_weight: float,
    top_k_values: list[int],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_hidden = 0
    total_correct = 0
    topk_sum: dict[int, float] = {k: 0.0 for k in top_k_values}
    topk_count: dict[int, int] = {k: 0 for k in top_k_values}
    all_hidden_scores: list[np.ndarray] = []
    all_hidden_labels: list[np.ndarray] = []

    success = split.success
    applicability = split.applicability
    steps = split.steps_completed_fraction

    with torch.no_grad():
        num_rows = success.shape[0]
        for start in range(0, num_rows, batch_size):
            stop = min(start + batch_size, num_rows)
            success_batch = success[start:stop].to(device)
            applicability_batch = applicability[start:stop].to(device)
            reveal_batch = reveal_mask[start:stop].to(device)
            steps_batch = steps[start:stop].to(device)

            x_steps_batch = steps_batch * reveal_batch.float()
            model_input = torch.cat(
                [x_steps_batch, reveal_batch.float(), applicability_batch], dim=1
            )
            logits = model(model_input)
            loss, hidden_count = _masked_bce_loss(
                logits,
                steps_batch,
                applicability_batch,
                reveal_batch,
                pos_weight,
            )
            if loss is None:
                continue
            total_loss += float(loss.item()) * hidden_count
            total_hidden += hidden_count

            # Compare predicted steps against binary success for AUROC/AP.
            probs = torch.sigmoid(logits)
            hidden_mask = (applicability_batch > 0.5) & (~reveal_batch)
            hidden_probs = probs[hidden_mask]
            hidden_labels = success_batch[hidden_mask]
            hidden_preds = (hidden_probs >= 0.5).float()
            total_correct += int((hidden_preds == hidden_labels).sum().item())

            all_hidden_scores.append(hidden_probs.detach().cpu().numpy())
            all_hidden_labels.append(hidden_labels.detach().cpu().numpy())

            # Top-k success precision among untried applicable columns.
            for row_idx in range(success_batch.shape[0]):
                row_mask = hidden_mask[row_idx]
                row_candidates = int(row_mask.sum().item())
                if row_candidates == 0:
                    continue
                row_scores = probs[row_idx][row_mask]
                row_labels = success_batch[row_idx][row_mask]
                for top_k in top_k_values:
                    effective_k = min(top_k, row_candidates)
                    if effective_k <= 0:
                        continue
                    chosen = torch.topk(row_scores, k=effective_k, largest=True).indices
                    precision = float(row_labels[chosen].mean().item())
                    topk_sum[top_k] += precision
                    topk_count[top_k] += 1

    mean_loss = total_loss / max(1, total_hidden)
    hidden_accuracy = float(total_correct / max(1, total_hidden))

    if all_hidden_scores:
        hidden_scores_np = np.concatenate(all_hidden_scores, axis=0)
        hidden_labels_np = np.concatenate(all_hidden_labels, axis=0)
    else:
        hidden_scores_np = np.array([], dtype=np.float32)
        hidden_labels_np = np.array([], dtype=np.float32)

    if hidden_scores_np.size == 0 or np.unique(hidden_labels_np).size < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(hidden_labels_np, hidden_scores_np))

    if hidden_scores_np.size == 0:
        average_precision = float("nan")
    else:
        average_precision = float(
            average_precision_score(hidden_labels_np, hidden_scores_np)
        )

    topk_precision = {
        top_k: float(topk_sum[top_k] / max(1, topk_count[top_k]))
        for top_k in top_k_values
    }

    return {
        "loss": mean_loss,
        "hidden_count": total_hidden,
        "hidden_accuracy": hidden_accuracy,
        "auroc": auroc,
        "average_precision": average_precision,
        "topk_precision": topk_precision,
        "topk_count": topk_count,
    }


def _evaluate_over_masks(
    model: nn.Module,
    split: "SplitTensors",
    reveal_masks: list[torch.Tensor],
    batch_size: int,
    device: torch.device,
    pos_weight: float,
    top_k_values: list[int],
) -> dict[str, Any]:
    if not reveal_masks:
        raise ValueError("Validation reveal mask list must be non-empty")

    total_weighted_loss = 0.0
    total_hidden = 0
    total_correct_weighted = 0.0
    auroc_weighted_sum = 0.0
    auroc_weight = 0
    ap_weighted_sum = 0.0
    ap_weight = 0
    topk_weighted_sum: dict[int, float] = {k: 0.0 for k in top_k_values}
    topk_total_count: dict[int, int] = {k: 0 for k in top_k_values}

    for reveal_mask in reveal_masks:
        metrics = _evaluate(
            model,
            split,
            reveal_mask,
            batch_size,
            device,
            pos_weight,
            top_k_values,
        )
        mask_loss = float(metrics["loss"])
        mask_hidden = int(metrics["hidden_count"])
        total_weighted_loss += mask_loss * mask_hidden
        total_hidden += mask_hidden
        total_correct_weighted += float(metrics["hidden_accuracy"]) * mask_hidden

        mask_auroc = float(metrics["auroc"])
        if not np.isnan(mask_auroc):
            auroc_weighted_sum += mask_auroc * mask_hidden
            auroc_weight += mask_hidden

        mask_ap = float(metrics["average_precision"])
        if not np.isnan(mask_ap):
            ap_weighted_sum += mask_ap * mask_hidden
            ap_weight += mask_hidden

        mask_topk = metrics["topk_precision"]
        mask_topk_count = metrics["topk_count"]
        for top_k in top_k_values:
            count = int(mask_topk_count[top_k])
            topk_weighted_sum[top_k] += float(mask_topk[top_k]) * count
            topk_total_count[top_k] += count

    mean_loss = total_weighted_loss / max(1, total_hidden)
    hidden_accuracy = float(total_correct_weighted / max(1, total_hidden))
    auroc = (
        float(auroc_weighted_sum / auroc_weight) if auroc_weight > 0 else float("nan")
    )
    average_precision = (
        float(ap_weighted_sum / ap_weight) if ap_weight > 0 else float("nan")
    )
    topk_precision = {
        top_k: float(topk_weighted_sum[top_k] / max(1, topk_total_count[top_k]))
        for top_k in top_k_values
    }

    return {
        "loss": mean_loss,
        "hidden_count": total_hidden,
        "hidden_accuracy": hidden_accuracy,
        "auroc": auroc,
        "average_precision": average_precision,
        "topk_precision": topk_precision,
    }


def _compute_static_first_index(
    success: torch.Tensor,
    applicability: torch.Tensor,
) -> int:
    """Choose a single global skeleton index for static-first rollout.

    Uses training-set success rate among applicable rows, with success count as tie-
    breaker.
    """

    applicable_counts = applicability.sum(dim=0)
    success_counts = success.sum(dim=0)

    valid = applicable_counts > 0.5
    if not bool(torch.any(valid).item()):
        raise ValueError(
            "No applicable skeleton columns available for static-first rollout"
        )

    rates = torch.zeros_like(success_counts)
    rates[valid] = success_counts[valid] / applicable_counts[valid]

    candidate_indices = torch.nonzero(valid, as_tuple=False).view(-1)
    candidate_rates = rates[candidate_indices]
    best_rate = float(candidate_rates.max().item())
    near_best = torch.isclose(candidate_rates, torch.tensor(best_rate), atol=1e-8)
    near_best_indices = candidate_indices[near_best]

    if near_best_indices.numel() == 1:
        return int(near_best_indices.item())

    tie_success = success_counts[near_best_indices]
    tie_best = int(torch.argmax(tie_success).item())
    return int(near_best_indices[tie_best].item())


def _sequential_rollout_metric(
    model: nn.Module,
    split: "SplitTensors",
    device: torch.device,
    seed: int,
    first_pick_mode: str,
    static_first_index: int | None,
) -> dict[str, float]:
    """Sequentially reveal outcomes and count tries to first success.

    Procedure per row:
    1) Reveal one initial applicable skeleton outcome.
       - random: random applicable skeleton.
             - static-first: fixed global skeleton if applicable, else random
                 applicable fallback.
    2) Predict remaining untried applicable columns using the steps head.
    3) Try highest predicted feasible/untried applicable skeleton.
    4) Repeat until success or exhaustion.
    """

    if first_pick_mode not in {"random", "static-first"}:
        raise ValueError("first_pick_mode must be one of {'random', 'static-first'}")
    if first_pick_mode == "static-first" and static_first_index is None:
        raise ValueError(
            "static_first_index is required when first_pick_mode='static-first'"
        )

    rng = np.random.default_rng(seed)
    model.eval()

    success = split.success
    applicability = split.applicability
    steps = split.steps_completed_fraction

    num_rows, vocab_size = success.shape
    tries_until_stop: list[int] = []
    tries_to_success_on_solvable: list[int] = []

    with torch.no_grad():
        for row_idx in range(num_rows):
            row_applicable = applicability[row_idx] > 0.5
            applicable_indices = torch.nonzero(row_applicable, as_tuple=False).view(-1)
            num_applicable = int(applicable_indices.numel())
            if num_applicable == 0:
                continue

            row_success = success[row_idx]
            row_steps = steps[row_idx]
            solvable = bool(torch.any(row_success[row_applicable] > 0.5).item())

            x_steps = torch.zeros((1, vocab_size), dtype=torch.float32, device=device)
            m = torch.zeros((1, vocab_size), dtype=torch.float32, device=device)
            a = applicability[row_idx : row_idx + 1].to(device)

            applicable_indices_np = applicable_indices.cpu().numpy()

            if first_pick_mode == "random":
                first_choice = int(rng.choice(applicable_indices_np))
            else:
                assert static_first_index is not None
                if int(static_first_index) in applicable_indices_np.tolist():
                    first_choice = int(static_first_index)
                else:
                    first_choice = int(rng.choice(applicable_indices_np))

            first_outcome_success = float(row_success[first_choice].item())
            x_steps[0, first_choice] = float(row_steps[first_choice].item())
            m[0, first_choice] = 1.0

            tried: set[int] = {first_choice}
            tries = 1
            success_found = first_outcome_success > 0.5

            while (not success_found) and (len(tried) < num_applicable):
                model_input = torch.cat([x_steps, m, a], dim=1)
                probs = torch.sigmoid(model(model_input)[0])

                remaining = [
                    idx for idx in applicable_indices_np.tolist() if idx not in tried
                ]
                remaining_tensor = torch.tensor(
                    remaining, device=device, dtype=torch.long
                )
                remaining_scores = probs[remaining_tensor]
                best_pos = int(torch.argmax(remaining_scores).item())
                next_choice = int(remaining[best_pos])

                outcome = float(row_success[next_choice].item())
                x_steps[0, next_choice] = float(row_steps[next_choice].item())
                m[0, next_choice] = 1.0
                tried.add(next_choice)
                tries += 1
                success_found = outcome > 0.5

            tries_until_stop.append(tries)
            if solvable:
                tries_to_success_on_solvable.append(tries)

    if not tries_until_stop:
        return {
            "rollout_mean_tries_until_stop": float("nan"),
            "rollout_mean_tries_to_success_on_solvable": float("nan"),
            "rollout_num_rows": 0.0,
            "rollout_num_solvable_rows": 0.0,
        }

    return {
        "rollout_mean_tries_until_stop": float(np.mean(tries_until_stop)),
        "rollout_mean_tries_to_success_on_solvable": (
            float(np.mean(tries_to_success_on_solvable))
            if tries_to_success_on_solvable
            else float("nan")
        ),
        "rollout_num_rows": float(len(tries_until_stop)),
        "rollout_num_solvable_rows": float(len(tries_to_success_on_solvable)),
    }


@hydra.main(
    config_path="conf",
    config_name="train_encoder_mae_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Train the MAE model and optionally evaluate on test artifacts."""
    torch.manual_seed(int(cfg.train.seed))
    np.random.seed(int(cfg.train.seed))

    if bool(cfg.bootstrap.enabled):
        _bootstrap_env_model_modules(cfg)

    train_path = _resolve_path(str(cfg.data.train_path))
    val_path = _resolve_path(str(cfg.data.val_path))
    test_path_cfg = cfg.data.test_path
    if not train_path.exists():
        raise FileNotFoundError(f"Train artifact not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation artifact not found: {val_path}")

    train_payload = _load_pickle(train_path)
    val_payload = _load_pickle(val_path)
    test_payload: dict[str, Any] | None = None

    if test_path_cfg is not None:
        test_path = _resolve_path(str(test_path_cfg))
        if not test_path.exists():
            raise FileNotFoundError(f"Test artifact not found: {test_path}")
        test_payload = _load_pickle(test_path)

    train_split = _extract_split_tensors(train_payload, "train")
    val_split = _extract_split_tensors(val_payload, "validation")
    _assert_same_vocab(train_split.vocab, val_split.vocab, "validation")
    test_split: SplitTensors | None = None
    if test_payload is not None:
        test_split = _extract_split_tensors(test_payload, "test")
        _assert_same_vocab(train_split.vocab, test_split.vocab, "test")

    num_rows, vocab_size = train_split.success.shape
    val_rows, val_vocab_size = val_split.success.shape
    print(
        "Loaded splits: "
        f"train_rows={num_rows}, val_rows={val_rows}, vocab_size={vocab_size}"
    )
    if val_vocab_size != vocab_size:
        raise ValueError("Train/validation vocab_size mismatch")

    device = _select_device(str(cfg.train.device))
    print(f"Using device: {device}")

    model = EncoderMAE(
        input_dim=3 * vocab_size,
        hidden_dims=[int(x) for x in cfg.model.hidden_dims],
        output_dim=vocab_size,
        dropout=float(cfg.model.dropout),
        use_layer_norm=bool(cfg.model.use_layer_norm),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    train_success = train_split.success.float()
    train_applicability = train_split.applicability.float()
    train_steps = train_split.steps_completed_fraction.float()

    applicable_mask = train_applicability > 0.5
    # pos_weight is based on full successes (steps == 1.0) vs other applicable entries.
    pos = float((train_steps[applicable_mask] > 0.5).sum().item())
    total = int(applicable_mask.sum().item())
    neg = float(total) - pos
    if pos <= 0:
        raise ValueError("No positive applicable training examples.")
    pos_weight = min(neg / pos, 20.0)
    print(
        "Training label stats over applicable entries: "
        f"total={total}, pos={int(pos)}, neg={int(neg)}, pos_weight={pos_weight:.4f}"
    )

    train_mask_rng = torch.Generator(device="cpu")
    train_mask_rng.manual_seed(int(cfg.train.seed) + 100)
    shuffle_rng = torch.Generator(device="cpu")
    shuffle_rng.manual_seed(int(cfg.train.seed) + 200)

    val_applicability = val_split.applicability.float()

    num_val_masks = int(getattr(cfg.val, "num_masks", 1))
    if num_val_masks < 1:
        raise ValueError("val.num_masks must be >= 1")
    val_reveal_masks: list[torch.Tensor] = []
    for mask_idx in range(num_val_masks):
        val_mask_rng = torch.Generator(device="cpu")
        val_mask_rng.manual_seed(int(cfg.val.mask_seed) + mask_idx)
        val_reveal_masks.append(
            _sample_reveal_mask(
                val_applicability,
                float(cfg.val.reveal_probability),
                val_mask_rng,
            )
        )

    output_dir = _resolve_path(str(cfg.checkpoint.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / str(cfg.checkpoint.best_filename)
    last_path = output_dir / str(cfg.checkpoint.last_filename)
    metrics_path = output_dir / "encoder_mae_metrics.npz"

    num_epochs = int(cfg.train.num_epochs)
    batch_size = int(cfg.train.batch_size)
    grad_clip_norm = float(cfg.train.grad_clip_norm)
    top_k_values = [int(value) for value in cfg.metrics.top_k_values]
    if not top_k_values:
        raise ValueError("metrics.top_k_values must be non-empty")
    rollout_seed = int(cfg.rollout.seed)
    rollout_static_first_index_cfg = getattr(cfg.rollout, "static_first_index", None)
    if rollout_static_first_index_cfg is None:
        rollout_static_first_index = _compute_static_first_index(
            train_success,
            train_applicability,
        )
        print("Rollout static-first index (auto): " f"{rollout_static_first_index}")
    else:
        rollout_static_first_index = int(rollout_static_first_index_cfg)
        if rollout_static_first_index < 0 or rollout_static_first_index >= vocab_size:
            raise ValueError("rollout.static_first_index must be in [0, vocab_size)")
        print("Rollout static-first index (config): " f"{rollout_static_first_index}")

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_hidden_accuracy: list[float] = []
    val_auroc: list[float] = []
    val_average_precision: list[float] = []
    val_rollout_tries_random: list[float] = []
    val_rollout_tries_static_first: list[float] = []
    val_topk_history: dict[int, list[float]] = {k: [] for k in top_k_values}

    run_start_time = time.perf_counter()
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        perm = torch.randperm(train_success.shape[0], generator=shuffle_rng)

        epoch_weighted_loss = 0.0
        epoch_hidden_total = 0

        for start in range(0, train_success.shape[0], batch_size):
            stop = min(start + batch_size, train_success.shape[0])
            batch_indices = perm[start:stop]

            applicability_batch = train_applicability[batch_indices]

            reveal_mask = _sample_reveal_mask(
                applicability_batch,
                float(cfg.train.reveal_probability),
                train_mask_rng,
            )

            applicability_batch = applicability_batch.to(device)
            reveal_mask = reveal_mask.to(device)
            steps_batch = train_steps[batch_indices].to(device)

            x_steps_batch = steps_batch * reveal_mask.float()
            model_input = torch.cat(
                [x_steps_batch, reveal_mask.float(), applicability_batch], dim=1
            )

            logits = model(model_input)
            loss, hidden_count = _masked_bce_loss(
                logits,
                steps_batch,
                applicability_batch,
                reveal_mask,
                pos_weight,
            )
            if loss is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_weighted_loss += float(loss.item()) * hidden_count
            epoch_hidden_total += hidden_count
            global_step += 1

            if global_step % int(cfg.log_every_steps) == 0:
                print(
                    f"step={global_step} epoch={epoch} "
                    f"batch_loss={float(loss.item()):.6f} hidden={hidden_count}"
                )

        train_loss = epoch_weighted_loss / max(1, epoch_hidden_total)
        train_losses.append(train_loss)

        val_metrics = _evaluate_over_masks(
            model,
            val_split,
            val_reveal_masks,
            batch_size,
            device,
            pos_weight,
            top_k_values,
        )
        val_loss = float(val_metrics["loss"])
        val_hidden = int(val_metrics["hidden_count"])
        val_acc = float(val_metrics["hidden_accuracy"])
        val_auc = float(val_metrics["auroc"])
        val_ap = float(val_metrics["average_precision"])
        val_topk = val_metrics["topk_precision"]

        val_losses.append(val_loss)
        val_hidden_accuracy.append(val_acc)
        val_auroc.append(val_auc)
        val_average_precision.append(val_ap)
        for top_k in top_k_values:
            val_topk_history[top_k].append(float(val_topk[top_k]))

        rollout_metrics_random = _sequential_rollout_metric(
            model,
            val_split,
            device,
            seed=rollout_seed,
            first_pick_mode="random",
            static_first_index=None,
        )
        rollout_metrics_static_first = _sequential_rollout_metric(
            model,
            val_split,
            device,
            seed=rollout_seed,
            first_pick_mode="static-first",
            static_first_index=rollout_static_first_index,
        )
        val_rollout_tries_random.append(
            float(rollout_metrics_random["rollout_mean_tries_to_success_on_solvable"])
        )
        val_rollout_tries_static_first.append(
            float(
                rollout_metrics_static_first[
                    "rollout_mean_tries_to_success_on_solvable"
                ]
            )
        )
        rollout_random_tries = float(
            rollout_metrics_random["rollout_mean_tries_to_success_on_solvable"]
        )
        rollout_static_tries = float(
            rollout_metrics_static_first["rollout_mean_tries_to_success_on_solvable"]
        )

        topk_summary = " ".join(
            [f"top{top_k}={float(val_topk[top_k]):.4f}" for top_k in top_k_values]
        )
        epoch_elapsed = time.perf_counter() - epoch_start_time
        total_elapsed = time.perf_counter() - run_start_time
        avg_epoch_time = total_elapsed / epoch
        remaining_epochs = num_epochs - epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        print(
            f"epoch={epoch}/{num_epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_hidden={epoch_hidden_total} val_hidden={val_hidden} "
            f"val_acc={val_acc:.4f} val_auroc={val_auc:.4f} val_ap={val_ap:.4f} "
            f"{topk_summary} "
            "rollout_tries_random="
            f"{rollout_random_tries:.4f} "
            "rollout_tries_static_first="
            f"{rollout_static_tries:.4f} "
            f"epoch_time={_format_duration(epoch_elapsed)} "
            f"elapsed={_format_duration(total_elapsed)} "
            f"eta={_format_duration(eta_seconds)}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "vocab_size": vocab_size,
                    "input_dim": 3 * vocab_size,
                    "output_dim": vocab_size,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                best_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
            "best_val_loss": best_val_loss,
            "vocab_size": vocab_size,
            "input_dim": 3 * vocab_size,
            "output_dim": vocab_size,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        last_path,
    )

    np.savez(
        metrics_path,
        train_loss=np.asarray(train_losses, dtype=np.float32),
        val_loss=np.asarray(val_losses, dtype=np.float32),
        val_hidden_accuracy=np.asarray(val_hidden_accuracy, dtype=np.float32),
        val_auroc=np.asarray(val_auroc, dtype=np.float32),
        val_average_precision=np.asarray(val_average_precision, dtype=np.float32),
        val_rollout_tries_random=np.asarray(val_rollout_tries_random, dtype=np.float32),
        val_rollout_tries_static_first=np.asarray(
            val_rollout_tries_static_first,
            dtype=np.float32,
        ),
        best_val_loss=np.float32(best_val_loss),
        vocab_size=np.int32(vocab_size),
    )

    summary: dict[str, Any] = {
        "best_val_loss": best_val_loss,
        "final_val_hidden_accuracy": (
            val_hidden_accuracy[-1] if val_hidden_accuracy else float("nan")
        ),
        "final_val_auroc": val_auroc[-1] if val_auroc else float("nan"),
        "final_val_average_precision": (
            val_average_precision[-1] if val_average_precision else float("nan")
        ),
        "final_val_rollout_tries_random": (
            val_rollout_tries_random[-1] if val_rollout_tries_random else float("nan")
        ),
        "final_val_rollout_tries_static_first": (
            val_rollout_tries_static_first[-1]
            if val_rollout_tries_static_first
            else float("nan")
        ),
        "rollout_static_first_index": rollout_static_first_index,
        "final_val_topk_precision": {
            str(top_k): (
                val_topk_history[top_k][-1] if val_topk_history[top_k] else float("nan")
            )
            for top_k in top_k_values
        },
    }

    if test_split is not None:
        best_checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])

        test_applicability = test_split.applicability.float()
        test_reveal_masks: list[torch.Tensor] = []
        num_test_masks = int(getattr(cfg.test, "num_masks", num_val_masks))
        for mask_idx in range(num_test_masks):
            test_mask_rng = torch.Generator(device="cpu")
            test_mask_rng.manual_seed(int(cfg.test.mask_seed) + mask_idx)
            test_reveal_masks.append(
                _sample_reveal_mask(
                    test_applicability,
                    float(cfg.test.reveal_probability),
                    test_mask_rng,
                )
            )

        test_metrics = _evaluate_over_masks(
            model,
            test_split,
            test_reveal_masks,
            batch_size,
            device,
            pos_weight,
            top_k_values,
        )
        test_rollout_random = _sequential_rollout_metric(
            model,
            test_split,
            device,
            seed=int(cfg.test.rollout_seed),
            first_pick_mode="random",
            static_first_index=None,
        )
        test_rollout_static_first = _sequential_rollout_metric(
            model,
            test_split,
            device,
            seed=int(cfg.test.rollout_seed),
            first_pick_mode="static-first",
            static_first_index=rollout_static_first_index,
        )

        summary["test"] = {
            "loss": float(test_metrics["loss"]),
            "hidden_count": int(test_metrics["hidden_count"]),
            "hidden_accuracy": float(test_metrics["hidden_accuracy"]),
            "auroc": float(test_metrics["auroc"]),
            "average_precision": float(test_metrics["average_precision"]),
            "topk_precision": {
                str(top_k): float(test_metrics["topk_precision"][top_k])
                for top_k in top_k_values
            },
            "rollout_random": test_rollout_random,
            "rollout_static_first": test_rollout_static_first,
        }

        test_rollout_random_tries = summary["test"]["rollout_random"][
            "rollout_mean_tries_to_success_on_solvable"
        ]
        test_rollout_static_tries = summary["test"]["rollout_static_first"][
            "rollout_mean_tries_to_success_on_solvable"
        ]

        print(
            "test "
            f"loss={summary['test']['loss']:.6f} "
            f"acc={summary['test']['hidden_accuracy']:.4f} "
            f"auroc={summary['test']['auroc']:.4f} "
            f"ap={summary['test']['average_precision']:.4f} "
            "rollout_tries_random="
            f"{test_rollout_random_tries:.4f} "
            "rollout_tries_static_first="
            f"{test_rollout_static_tries:.4f}"
        )

    summary_path = output_dir / "encoder_mae_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
