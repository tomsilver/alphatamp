"""Offline evaluation for Encoder MAE vs static baseline on test matrices.

This script compares two policies using only precomputed matrices from dataset artifacts:
- applicability: whether a skeleton can be attempted for a seed
- success: whether refinement would succeed for that (seed, skeleton)
- refinement_time: recorded refinement runtime for that (seed, skeleton)

Policies:
1) Baseline fixed-order:
        - Compute global skeleton ranking from training split success rate among
            applicable rows.
   - For each test seed, try applicable skeletons in that static order.
2) Encoder-guided:
        - First try a static-first skeleton (best training success-rate column)
            when applicable.
        - If not applicable, deterministically fallback to model-best applicable
            with empty observations.
   - Then iteratively choose highest-probability feasible untried applicable skeleton
     from the MAE given partial observations.

Budgeting:
- Strict per-seed time budget.
- An attempt is only allowed if its recorded refinement_time <= remaining budget.
- If no further allowed attempts remain and no success was found, mark failure and total
  time for that seed as full budget.

Outputs:
- JSON summary
- NPZ metrics arrays
- Three PNG figures:
    1) success rate vs budget curve with configured budget marker
    2) successful-only refinement time vs budget
    3) total refinement time vs budget including failures
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import hydra
import kinder
import matplotlib.pyplot as plt
import numpy as np
import torch
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from omegaconf import DictConfig, OmegaConf
from torch import nn


@dataclass(frozen=True)
class SplitData:
    """Tensorized split matrices and optional per-seed planner context."""

    applicability: np.ndarray
    success: np.ndarray
    refinement_time: np.ndarray
    steps_completed_fraction: (
        np.ndarray
    )  # falls back to binary success for old datasets
    has_true_steps_completed_fraction: bool
    vocab: list[Any]
    initial_low_level_states: list[Any] | None = None
    initial_abstract_states: list[Any] | None = None
    problem_goals: list[Any] | None = None


class EncoderMAE(nn.Module):
    """Simple MLP masked autoencoder head producing M logits.

    Accepts inputs of shape (B, M, 3) and flattens to (B, 3*M) internally.
    """

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
        """Return logits for all vocabulary columns.

        Args:
            inputs: (B, M, 3) observation features.

        Returns:
            logits: (B, M)
        """
        B, M, _ = inputs.shape
        return self._network(inputs.reshape(B, M * 3))


class SkeletonTransformer(nn.Module):
    """Transformer encoder that re-ranks skeletons via cross-skeleton attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim_multiplier: int = 4,
        dropout: float = 0.1,
        use_id_embed: bool = False,
    ) -> None:
        super().__init__()
        self.obs_embed = nn.Linear(3, embed_dim)
        self.id_embed = nn.Embedding(vocab_size, embed_dim)
        self.prior_head = nn.Linear(embed_dim, 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * ffn_dim_multiplier,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.delta_head = nn.Linear(embed_dim, 1)
        self._use_id_embed = use_id_embed

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return per-skeleton logits.

        Args:
            obs: (B, M, 3) — [x_steps, reveal, applicability] per skeleton.

        Returns:
            logits: (B, M)
        """
        prior = self.prior_head(self.id_embed.weight).squeeze(-1)  # (M,)
        tokens = self.obs_embed(obs)  # (B, M, d)
        if self._use_id_embed:
            tokens = tokens + self.id_embed.weight  # broadcast (M, d) → (B, M, d)
        tokens = self.encoder(tokens)  # (B, M, d)
        delta = self.delta_head(tokens).squeeze(-1)  # (B, M)
        return prior.unsqueeze(0) + delta  # (B, M)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(hydra.utils.get_original_cwd()) / path


def _bootstrap_env_model_modules(cfg: DictConfig) -> None:
    """Register dynamic modules needed for dill deserialization."""
    env_id = str(cfg.bootstrap.env_id)
    model_name = str(cfg.bootstrap.model_name)
    model_kwargs = dict(OmegaConf.to_container(cfg.bootstrap.model_kwargs, resolve=True))

    kinder.register_all_environments()
    env = kinder.make(env_id)
    try:
        _ = create_bilevel_planning_models(
            model_name,
            env.observation_space,
            env.action_space,
            **model_kwargs,
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


def _extract_split_data(payload: dict[str, Any], split_name: str) -> SplitData:
    if "dataset" not in payload:
        raise KeyError(f"{split_name}: missing 'dataset' key")
    dataset = payload["dataset"]
    if not isinstance(dataset, dict):
        raise TypeError(f"{split_name}: dataset must be a dict")

    required = {"applicability", "success", "refinement_time", "op_sequence_vocab"}
    missing = required - set(dataset)
    if missing:
        raise KeyError(f"{split_name}: missing dataset keys: {sorted(missing)}")

    applicability = np.asarray(dataset["applicability"], dtype=np.float32)
    success = np.asarray(dataset["success"], dtype=np.float32)
    refinement_time = np.asarray(dataset["refinement_time"], dtype=np.float32)
    vocab = list(dataset["op_sequence_vocab"])

    if applicability.ndim != 2 or success.ndim != 2 or refinement_time.ndim != 2:
        raise ValueError(
            f"{split_name}: applicability/success/refinement_time must be rank-2"
        )
    if (
        applicability.shape != success.shape
        or applicability.shape != refinement_time.shape
    ):
        raise ValueError(
            f"{split_name}: shape mismatch A{applicability.shape} "
            f"Y{success.shape} T{refinement_time.shape}"
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
    if np.any(refinement_time < 0.0):
        raise ValueError(f"{split_name}: refinement_time must be non-negative")

    # Rich failure signal — fall back to binary success for old datasets.
    has_true_steps_completed_fraction = "steps_completed_fraction" in dataset
    if has_true_steps_completed_fraction:
        steps = np.asarray(dataset["steps_completed_fraction"], dtype=np.float32)
    else:
        steps = success.copy()

    initial_low_level_states = dataset.get("initial_low_level_states")
    initial_abstract_states = dataset.get("initial_abstract_states")
    problem_goals = dataset.get("problem_goals")

    num_rows = int(applicability.shape[0])
    if (
        initial_low_level_states is not None
        and len(initial_low_level_states) != num_rows
    ):
        raise ValueError(
            f"{split_name}: len(initial_low_level_states)="
            f"{len(initial_low_level_states)} "
            f"!= num_rows={num_rows}"
        )
    if initial_abstract_states is not None and len(initial_abstract_states) != num_rows:
        raise ValueError(
            f"{split_name}: len(initial_abstract_states)={len(initial_abstract_states)} "
            f"!= num_rows={num_rows}"
        )
    if problem_goals is not None and len(problem_goals) != num_rows:
        raise ValueError(
            f"{split_name}: len(problem_goals)={len(problem_goals)} "
            f"!= num_rows={num_rows}"
        )

    return SplitData(
        applicability=applicability,
        success=success,
        refinement_time=refinement_time,
        steps_completed_fraction=steps,
        has_true_steps_completed_fraction=has_true_steps_completed_fraction,
        vocab=vocab,
        initial_low_level_states=(
            list(initial_low_level_states)
            if initial_low_level_states is not None
            else None
        ),
        initial_abstract_states=(
            list(initial_abstract_states)
            if initial_abstract_states is not None
            else None
        ),
        problem_goals=list(problem_goals) if problem_goals is not None else None,
    )


def _extract_training_timeout_seconds(payload: dict[str, Any]) -> float | None:
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        return None
    timeout = cfg.get("training_planning_timeout")
    if timeout is None:
        return None
    timeout_value = float(timeout)
    if timeout_value <= 0:
        raise ValueError("training_planning_timeout in payload config must be > 0")
    return timeout_value


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


def _compute_training_ranking(train: SplitData) -> tuple[np.ndarray, int]:
    """Return global descending order by train success rate and static-first index.

    Tie-breaks:
    1) higher success rate,
    2) higher success count,
    3) lower index.
    """

    applicability = train.applicability
    success = train.success
    applicable_counts = applicability.sum(axis=0)
    success_counts = success.sum(axis=0)

    rates = np.zeros_like(success_counts, dtype=np.float64)
    valid = applicable_counts > 0.5
    rates[valid] = success_counts[valid] / applicable_counts[valid]

    n_cols = applicability.shape[1]
    indices = np.arange(n_cols)

    order = sorted(
        indices.tolist(),
        key=lambda idx: (-rates[idx], -success_counts[idx], idx),
    )

    if not np.any(valid):
        raise ValueError("No applicable skeleton columns available in training split")

    valid_indices = np.nonzero(valid)[0].tolist()
    static_first = sorted(
        valid_indices,
        key=lambda idx: (-rates[idx], -success_counts[idx], idx),
    )[0]

    return np.asarray(order, dtype=np.int64), int(static_first)


def _strict_attempt_allowed(
    attempt_time: float, remaining: float, epsilon: float
) -> bool:
    return attempt_time <= (remaining + epsilon)


def _freeze_ground_op_sequence(skeleton: Any) -> tuple[Any, ...]:
    return tuple(skeleton[1])


def _build_vocab_to_index(vocab: list[Any]) -> dict[Any, int]:
    return {entry: idx for idx, entry in enumerate(vocab)}


def _run_baseline_row(
    applicable_row: np.ndarray,
    success_row: np.ndarray,
    time_row: np.ndarray,
    global_order: np.ndarray,
    budget_seconds: float,
    epsilon: float,
) -> tuple[bool, float]:
    remaining = float(budget_seconds)
    elapsed = 0.0
    solved = False

    applicable_indices = set(np.nonzero(applicable_row > 0.5)[0].tolist())
    if not applicable_indices:
        return False, float(budget_seconds)

    for col_idx in global_order.tolist():
        if col_idx not in applicable_indices:
            continue

        attempt_time = float(time_row[col_idx])
        if not _strict_attempt_allowed(attempt_time, remaining, epsilon):
            break

        elapsed += attempt_time
        remaining -= attempt_time

        if success_row[col_idx] > 0.5:
            solved = True
            break

    if not solved:
        return False, float(budget_seconds)
    return True, float(min(elapsed, budget_seconds))


def _run_generator_order_baseline_row(
    applicable_row: np.ndarray,
    success_row: np.ndarray,
    time_row: np.ndarray,
    ordered_vocab_indices: list[int],
    budget_seconds: float,
    epsilon: float,
) -> tuple[bool, float]:
    remaining = float(budget_seconds)
    elapsed = 0.0

    for col_idx in ordered_vocab_indices:
        if applicable_row[col_idx] <= 0.5:
            continue

        attempt_time = float(time_row[col_idx])
        if not _strict_attempt_allowed(attempt_time, remaining, epsilon):
            break

        elapsed += attempt_time
        remaining -= attempt_time

        if success_row[col_idx] > 0.5:
            return True, float(min(elapsed, budget_seconds))

    return False, float(budget_seconds)


def _compute_generator_order_indices(
    cfg: DictConfig,
    test_split: SplitData,
    vocab_to_idx: dict[Any, int],
    timeout_seconds: float,
    max_generated_skeletons_per_row: int,
) -> list[list[int]]:
    if (
        test_split.initial_low_level_states is None
        or test_split.initial_abstract_states is None
        or test_split.problem_goals is None
    ):
        raise ValueError(
            "Generator-order baseline requires dataset keys: "
            "initial_low_level_states, initial_abstract_states, problem_goals"
        )

    env_id = str(cfg.bootstrap.env_id)
    model_name = str(cfg.bootstrap.model_name)
    model_kwargs = dict(OmegaConf.to_container(cfg.bootstrap.model_kwargs, resolve=True))

    heuristic_name = str(cfg.generator_baseline.heuristic_name)
    generator_seed = int(cfg.generator_baseline.seed)

    kinder.register_all_environments()
    env = kinder.make(env_id)
    try:
        env_models = create_bilevel_planning_models(
            model_name,
            env.observation_space,
            env.action_space,
            **model_kwargs,
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]

    abstract_plan_generator: RelationalHeuristicSearchAbstractPlanGenerator = (
        RelationalHeuristicSearchAbstractPlanGenerator(
            env_models.types,
            env_models.predicates,
            env_models.operators,
            heuristic_name,
            seed=generator_seed,
        )
    )

    ordered_indices_per_row: list[list[int]] = []
    num_rows = int(test_split.applicability.shape[0])

    for row_idx in range(num_rows):
        x0 = test_split.initial_low_level_states[row_idx]
        s0 = test_split.initial_abstract_states[row_idx]
        goal = test_split.problem_goals[row_idx]

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        row_order: list[int] = []
        seen_vocab_indices: set[int] = set()
        row_applicable_count = int(np.sum(test_split.applicability[row_idx] > 0.5))

        generated_count = 0
        generator = abstract_plan_generator(x0, s0, goal, timeout_seconds, bpg)
        for skeleton in generator:
            generated_count += 1

            op_sequence = _freeze_ground_op_sequence(skeleton)
            col_idx = vocab_to_idx.get(op_sequence)
            if col_idx is None:
                if 0 < max_generated_skeletons_per_row <= generated_count:
                    break
                continue

            if col_idx not in seen_vocab_indices:
                row_order.append(col_idx)
                seen_vocab_indices.add(col_idx)

            if len(row_order) >= row_applicable_count:
                break
            if 0 < max_generated_skeletons_per_row <= generated_count:
                break

        ordered_indices_per_row.append(row_order)

    return ordered_indices_per_row


def _model_scores_for_row(
    model: nn.Module,
    x_steps: np.ndarray,
    m: np.ndarray,
    a: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return sigmoid probabilities for a single row."""
    x_steps_t = torch.from_numpy(x_steps.astype(np.float32)).unsqueeze(0).to(device)
    m_t = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).to(device)
    a_t = torch.from_numpy(a.astype(np.float32)).unsqueeze(0).to(device)
    model_input = torch.stack([x_steps_t, m_t, a_t], dim=2)
    with torch.no_grad():
        logits = model(model_input)
        probs = torch.sigmoid(logits[0]).detach().cpu().numpy()
    return probs


def _pick_best_untried_applicable(
    probs: np.ndarray,
    applicable_indices: list[int],
    tried: set[int],
) -> int | None:
    remaining = [idx for idx in applicable_indices if idx not in tried]
    if not remaining:
        return None
    scores = probs[np.asarray(remaining, dtype=np.int64)]
    best_pos = int(np.argmax(scores))
    return int(remaining[best_pos])


def _run_encoder_row(
    model: nn.Module,
    applicable_row: np.ndarray,
    success_row: np.ndarray,
    time_row: np.ndarray,
    steps_row: np.ndarray,
    budget_seconds: float,
    epsilon: float,
    device: torch.device,
) -> tuple[bool, float]:
    remaining = float(budget_seconds)
    elapsed = 0.0

    applicable_indices = np.nonzero(applicable_row > 0.5)[0].tolist()
    if not applicable_indices:
        return False, float(budget_seconds)

    x_steps = np.zeros_like(success_row, dtype=np.float32)
    m = np.zeros_like(success_row, dtype=np.float32)
    a = applicable_row.astype(np.float32)

    # Reveal all inapplicable skeletons upfront — their performance (0) is free
    # information that is known before any attempts are made.
    m[applicable_row <= 0.5] = 1.0

    tried: set[int] = set()

    first_probs = _model_scores_for_row(model, x_steps, m, a, device)
    first_choice = _pick_best_untried_applicable(first_probs, applicable_indices, tried)
    if first_choice is None:
        return False, float(budget_seconds)

    while True:
        attempt_time = float(time_row[first_choice])
        if not _strict_attempt_allowed(attempt_time, remaining, epsilon):
            break

        elapsed += attempt_time
        remaining -= attempt_time

        x_steps[first_choice] = float(steps_row[first_choice])
        m[first_choice] = 1.0
        tried.add(first_choice)

        if float(success_row[first_choice]) > 0.5:
            return True, float(min(elapsed, budget_seconds))

        probs = _model_scores_for_row(model, x_steps, m, a, device)
        next_choice = _pick_best_untried_applicable(probs, applicable_indices, tried)
        if next_choice is None:
            break
        first_choice = next_choice

    return False, float(budget_seconds)


def _run_oracle_steps_then_time_row(
    applicable_row: np.ndarray,
    success_row: np.ndarray,
    time_row: np.ndarray,
    steps_row: np.ndarray,
    budget_seconds: float,
    epsilon: float,
) -> tuple[bool, float]:
    """Oracle rollout by descending steps-completed, tie-broken by lower time."""
    remaining = float(budget_seconds)
    elapsed = 0.0

    applicable_indices = np.nonzero(applicable_row > 0.5)[0].tolist()
    if not applicable_indices:
        return False, float(budget_seconds)

    ordered = sorted(
        applicable_indices,
        key=lambda idx: (-float(steps_row[idx]), float(time_row[idx]), int(idx)),
    )

    for col_idx in ordered:
        attempt_time = float(time_row[col_idx])
        if not _strict_attempt_allowed(attempt_time, remaining, epsilon):
            break

        elapsed += attempt_time
        remaining -= attempt_time

        if float(success_row[col_idx]) > 0.5:
            return True, float(min(elapsed, budget_seconds))

    return False, float(budget_seconds)


def _evaluate_policy(
    method_name: str,
    run_row_fn: Any,
    test: SplitData,
    budget_seconds: float,
) -> dict[str, Any]:
    outcomes: list[bool] = []
    elapsed_times: list[float] = []

    n_rows = test.success.shape[0]
    for row_idx in range(n_rows):
        solved, elapsed = run_row_fn(
            row_idx,
            test.applicability[row_idx],
            test.success[row_idx],
            test.refinement_time[row_idx],
            budget_seconds,
        )
        outcomes.append(bool(solved))
        elapsed_times.append(float(elapsed))

    outcomes_np = np.asarray(outcomes, dtype=bool)
    elapsed_np = np.asarray(elapsed_times, dtype=np.float64)
    solved_times = elapsed_np[outcomes_np]

    return {
        "method": method_name,
        "num_rows": int(n_rows),
        "solved_count": int(outcomes_np.sum()),
        "failed_count": int((~outcomes_np).sum()),
        "success_rate": float(outcomes_np.mean()) if n_rows > 0 else float("nan"),
        "mean_time_success_only": (
            float(solved_times.mean()) if solved_times.size > 0 else float("nan")
        ),
        "mean_time_total": (
            float(elapsed_np.mean()) if elapsed_np.size > 0 else float("nan")
        ),
        "outcomes": outcomes_np,
        "elapsed_times": elapsed_np,
    }


def _plot_success_curve(
    budgets: np.ndarray,
    baseline_success: np.ndarray,
    generator_order_success: np.ndarray | None,
    box_offline_success: np.ndarray | None,
    oracle_steps_time_success: np.ndarray | None,
    encoder_success: np.ndarray,
    budget_marker: float,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(budgets, baseline_success, label="Baseline (fixed-order)", linewidth=2)
    if generator_order_success is not None:
        plt.plot(
            budgets,
            generator_order_success,
            label="Baseline (generator-order)",
            linewidth=2,
        )
    if box_offline_success is not None:
        plt.plot(
            budgets,
            box_offline_success,
            label="BOX (offline)",
            linewidth=2,
        )
    if oracle_steps_time_success is not None:
        plt.plot(
            budgets,
            oracle_steps_time_success,
            label="Oracle (steps->time)",
            linewidth=2,
        )
    plt.plot(budgets, encoder_success, label="Encoder-guided", linewidth=2)
    plt.axvline(
        budget_marker, linestyle="--", linewidth=1.5, label=f"Budget={budget_marker:g}s"
    )
    plt.ylim(0.0, 1.0)
    plt.xlabel("Time budget (seconds)")
    plt.ylabel("Success rate")
    plt.title("Success rate vs budget")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _plot_time_curve(
    budgets: np.ndarray,
    baseline_values: np.ndarray,
    generator_order_values: np.ndarray | None,
    box_offline_values: np.ndarray | None,
    oracle_steps_time_values: np.ndarray | None,
    encoder_values: np.ndarray,
    budget_marker: float,
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(budgets, baseline_values, label="Baseline (fixed-order)", linewidth=2)
    if generator_order_values is not None:
        plt.plot(
            budgets,
            generator_order_values,
            label="Baseline (generator-order)",
            linewidth=2,
        )
    if box_offline_values is not None:
        plt.plot(
            budgets,
            box_offline_values,
            label="BOX (offline)",
            linewidth=2,
        )
    if oracle_steps_time_values is not None:
        plt.plot(
            budgets,
            oracle_steps_time_values,
            label="Oracle (steps->time)",
            linewidth=2,
        )
    plt.plot(budgets, encoder_values, label="Encoder-guided", linewidth=2)
    plt.axvline(
        budget_marker, linestyle="--", linewidth=1.5, label=f"Budget={budget_marker:g}s"
    )
    plt.title(title)
    plt.xlabel("Time budget (seconds)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


@hydra.main(
    config_path="conf",
    config_name="offline_encoder_rollout_eval_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run offline rollout evaluation and write metrics/plots artifacts."""
    if bool(cfg.bootstrap.enabled):
        _bootstrap_env_model_modules(cfg)

    train_path = _resolve_path(str(cfg.data.train_path))
    test_path = _resolve_path(str(cfg.data.test_path))
    ckpt_path = _resolve_path(str(cfg.checkpoint.path))
    out_dir = _resolve_path(str(cfg.output.dir))

    if not train_path.exists():
        raise FileNotFoundError(f"Train artifact not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test artifact not found: {test_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    train_payload = _load_pickle(train_path)
    test_payload = _load_pickle(test_path)

    train_split = _extract_split_data(train_payload, "train")
    test_split = _extract_split_data(test_payload, "test")
    _assert_same_vocab(train_split.vocab, test_split.vocab, "test")

    global_order, static_first_index = _compute_training_ranking(train_split)
    vocab_to_idx = _build_vocab_to_index(train_split.vocab)
    box_offline_enabled = bool(cfg.box_offline.enabled)
    box_offline_approach: Any | None = None
    oracle_steps_then_time_enabled = bool(cfg.oracle_steps_then_time.enabled)

    if (
        oracle_steps_then_time_enabled
        and not test_split.has_true_steps_completed_fraction
    ):
        raise ValueError(
            "oracle_steps_then_time.enabled requires true "
            "steps_completed_fraction in test dataset"
        )

    generator_order_enabled = bool(cfg.generator_baseline.enabled)
    generator_order_indices_per_row: list[list[int]] | None = None
    generator_timeout_cfg = cfg.generator_baseline.timeout_seconds
    if generator_timeout_cfg is None:
        inferred_timeout = _extract_training_timeout_seconds(test_payload)
        if inferred_timeout is None:
            inferred_timeout = _extract_training_timeout_seconds(train_payload)
        if inferred_timeout is None:
            raise ValueError(
                "generator_baseline.timeout_seconds is null and no "
                "training_planning_timeout found in dataset payload config"
            )
        generator_timeout_seconds = float(inferred_timeout)
    else:
        generator_timeout_seconds = float(generator_timeout_cfg)
    if generator_timeout_seconds <= 0:
        raise ValueError("generator_baseline.timeout_seconds must be > 0")
    max_generated = int(cfg.generator_baseline.max_generated_skeletons_per_row)
    if max_generated < 0:
        raise ValueError(
            "generator_baseline.max_generated_skeletons_per_row must be >= 0"
        )

    device = _select_device(str(cfg.train.device))
    if box_offline_enabled:
        box_module = importlib.import_module("alphatamp.approaches.box_approach")
        BoxApproach = getattr(box_module, "BoxApproach")

        env_id = str(cfg.bootstrap.env_id)
        model_name = str(cfg.bootstrap.model_name)
        model_kwargs = dict(OmegaConf.to_container(cfg.bootstrap.model_kwargs, resolve=True))

        kinder.register_all_environments()
        env = kinder.make(env_id)
        try:
            box_env_models = create_bilevel_planning_models(
                model_name,
                env.observation_space,
                env.action_space,
                **model_kwargs,
            )
        finally:
            env.close()  # type: ignore[no-untyped-call]

        box_offline_approach = BoxApproach(
            env_models=box_env_models,
            seed=int(cfg.box_offline.seed),
            exploration_constant=float(cfg.box_offline.exploration_constant),
            failure_penalty_multiplier=float(
                cfg.box_offline.failure_penalty_multiplier
            ),
        )
        box_offline_approach.build_box_model_from_encoder_dataset_artifact(train_path)
        box_offline_approach.load_offline_planning_dataset_from_encoder_dataset_artifact(
            test_path
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict")

    ckpt_vocab_size = int(checkpoint["vocab_size"])
    if ckpt_vocab_size != train_split.applicability.shape[1]:
        raise ValueError(
            "Checkpoint vocab_size mismatch with dataset columns: "
            f"ckpt={ckpt_vocab_size} dataset={train_split.applicability.shape[1]}"
        )

    ckpt_cfg = checkpoint.get("config")
    if not isinstance(ckpt_cfg, dict):
        raise TypeError("Checkpoint missing resolved config dict")
    if "model" not in ckpt_cfg:
        raise KeyError("Checkpoint config missing 'model' section")

    model_cfg = ckpt_cfg["model"]
    arch = str(model_cfg.get("arch", "mlp"))
    dropout = float(model_cfg["dropout"])

    if arch == "transformer":
        model: nn.Module = SkeletonTransformer(
            vocab_size=ckpt_vocab_size,
            embed_dim=int(model_cfg["embed_dim"]),
            n_heads=int(model_cfg["n_heads"]),
            n_layers=int(model_cfg["n_layers"]),
            ffn_dim_multiplier=int(model_cfg["ffn_dim_multiplier"]),
            dropout=dropout,
            use_id_embed=bool(model_cfg["use_id_embed"]),
        ).to(device)
    else:
        hidden_dims = [int(x) for x in model_cfg["hidden_dims"]]
        use_layer_norm = bool(model_cfg["use_layer_norm"])
        # input_dim/output_dim are stored in new-format checkpoints; fall back to old
        # 3M/M defaults so old checkpoints continue to load correctly.
        ckpt_input_dim = int(checkpoint.get("input_dim", 3 * ckpt_vocab_size))
        ckpt_output_dim = int(checkpoint.get("output_dim", ckpt_vocab_size))
        model = EncoderMAE(
            input_dim=ckpt_input_dim,
            hidden_dims=hidden_dims,
            output_dim=ckpt_output_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    budget_seconds = float(cfg.budget.seconds)
    if budget_seconds <= 0:
        raise ValueError("budget.seconds must be > 0")
    epsilon = float(cfg.runtime.epsilon)
    if epsilon < 0:
        raise ValueError("runtime.epsilon must be >= 0")

    def baseline_row_runner(
        row_idx: int,
        applicable_row: np.ndarray,
        success_row: np.ndarray,
        time_row: np.ndarray,
        budget: float,
    ) -> tuple[bool, float]:
        del row_idx
        return _run_baseline_row(
            applicable_row,
            success_row,
            time_row,
            global_order,
            budget,
            epsilon,
        )

    def generator_order_row_runner(
        row_idx: int,
        applicable_row: np.ndarray,
        success_row: np.ndarray,
        time_row: np.ndarray,
        budget: float,
    ) -> tuple[bool, float]:
        if generator_order_indices_per_row is None:
            raise RuntimeError("Generator-order indices have not been initialized")
        return _run_generator_order_baseline_row(
            applicable_row,
            success_row,
            time_row,
            generator_order_indices_per_row[row_idx],
            budget,
            epsilon,
        )

    def encoder_row_runner(
        row_idx: int,
        applicable_row: np.ndarray,
        success_row: np.ndarray,
        time_row: np.ndarray,
        budget: float,
    ) -> tuple[bool, float]:
        return _run_encoder_row(
            model,
            applicable_row,
            success_row,
            time_row,
            test_split.steps_completed_fraction[row_idx],
            budget,
            epsilon,
            device,
        )

    def box_offline_row_runner(
        row_idx: int,
        applicable_row: np.ndarray,
        success_row: np.ndarray,
        time_row: np.ndarray,
        budget: float,
    ) -> tuple[bool, float]:
        del applicable_row, success_row, time_row
        if box_offline_approach is None:
            raise RuntimeError("BOX offline approach has not been initialized")
        result = box_offline_approach.run_offline_planning_by_row_index(row_idx, budget)
        return bool(result["success"]), float(result["elapsed_time"])

    def oracle_steps_then_time_row_runner(
        row_idx: int,
        applicable_row: np.ndarray,
        success_row: np.ndarray,
        time_row: np.ndarray,
        budget: float,
    ) -> tuple[bool, float]:
        return _run_oracle_steps_then_time_row(
            applicable_row,
            success_row,
            time_row,
            test_split.steps_completed_fraction[row_idx],
            budget,
            epsilon,
        )

    if generator_order_enabled:
        print("Computing per-row generator-order baseline indices...")
        generator_order_indices_per_row = _compute_generator_order_indices(
            cfg,
            test_split,
            vocab_to_idx,
            timeout_seconds=generator_timeout_seconds,
            max_generated_skeletons_per_row=max_generated,
        )

    baseline_metrics = _evaluate_policy(
        "baseline_fixed_order",
        baseline_row_runner,
        test_split,
        budget_seconds,
    )
    generator_order_metrics: dict[str, Any] | None = None
    if generator_order_enabled:
        generator_order_metrics = _evaluate_policy(
            "baseline_generator_order",
            generator_order_row_runner,
            test_split,
            budget_seconds,
        )
    box_offline_metrics: dict[str, Any] | None = None
    if box_offline_enabled:
        box_offline_metrics = _evaluate_policy(
            "box_offline",
            box_offline_row_runner,
            test_split,
            budget_seconds,
        )
    oracle_steps_then_time_metrics: dict[str, Any] | None = None
    if oracle_steps_then_time_enabled:
        oracle_steps_then_time_metrics = _evaluate_policy(
            "oracle_steps_then_time",
            oracle_steps_then_time_row_runner,
            test_split,
            budget_seconds,
        )
    encoder_metrics = _evaluate_policy(
        "encoder_guided",
        encoder_row_runner,
        test_split,
        budget_seconds,
    )

    sweep_min = float(cfg.budget.sweep_min_seconds)
    sweep_max = float(cfg.budget.sweep_max_seconds)
    sweep_num = int(cfg.budget.sweep_num_points)
    if sweep_min <= 0 or sweep_max <= 0:
        raise ValueError("budget sweep bounds must be > 0")
    if sweep_max < sweep_min:
        raise ValueError("budget.sweep_max_seconds must be >= budget.sweep_min_seconds")
    if sweep_num < 2:
        raise ValueError("budget.sweep_num_points must be >= 2")

    budgets = np.linspace(sweep_min, sweep_max, sweep_num, dtype=np.float64)
    baseline_success_curve = []
    generator_order_success_curve = []
    box_offline_success_curve = []
    oracle_steps_then_time_success_curve = []
    encoder_success_curve = []
    baseline_time_success_only_curve = []
    generator_order_time_success_only_curve = []
    box_offline_time_success_only_curve = []
    oracle_steps_then_time_time_success_only_curve = []
    encoder_time_success_only_curve = []
    baseline_time_total_curve = []
    generator_order_time_total_curve = []
    box_offline_time_total_curve = []
    oracle_steps_then_time_time_total_curve = []
    encoder_time_total_curve = []
    for sweep_budget in budgets.tolist():
        baseline_sweep = _evaluate_policy(
            "baseline_fixed_order",
            baseline_row_runner,
            test_split,
            float(sweep_budget),
        )
        generator_order_sweep: dict[str, Any] | None = None
        if generator_order_enabled:
            generator_order_sweep = _evaluate_policy(
                "baseline_generator_order",
                generator_order_row_runner,
                test_split,
                float(sweep_budget),
            )
        box_offline_sweep: dict[str, Any] | None = None
        if box_offline_enabled:
            box_offline_sweep = _evaluate_policy(
                "box_offline",
                box_offline_row_runner,
                test_split,
                float(sweep_budget),
            )
        oracle_steps_then_time_sweep: dict[str, Any] | None = None
        if oracle_steps_then_time_enabled:
            oracle_steps_then_time_sweep = _evaluate_policy(
                "oracle_steps_then_time",
                oracle_steps_then_time_row_runner,
                test_split,
                float(sweep_budget),
            )
        encoder_sweep = _evaluate_policy(
            "encoder_guided",
            encoder_row_runner,
            test_split,
            float(sweep_budget),
        )
        baseline_success_curve.append(float(baseline_sweep["success_rate"]))
        if generator_order_sweep is not None:
            generator_order_success_curve.append(
                float(generator_order_sweep["success_rate"])
            )
        if box_offline_sweep is not None:
            box_offline_success_curve.append(float(box_offline_sweep["success_rate"]))
        if oracle_steps_then_time_sweep is not None:
            oracle_steps_then_time_success_curve.append(
                float(oracle_steps_then_time_sweep["success_rate"])
            )
        encoder_success_curve.append(float(encoder_sweep["success_rate"]))
        baseline_time_success_only_curve.append(
            float(baseline_sweep["mean_time_success_only"])
        )
        if generator_order_sweep is not None:
            generator_order_time_success_only_curve.append(
                float(generator_order_sweep["mean_time_success_only"])
            )
        if box_offline_sweep is not None:
            box_offline_time_success_only_curve.append(
                float(box_offline_sweep["mean_time_success_only"])
            )
        if oracle_steps_then_time_sweep is not None:
            oracle_steps_then_time_time_success_only_curve.append(
                float(oracle_steps_then_time_sweep["mean_time_success_only"])
            )
        encoder_time_success_only_curve.append(
            float(encoder_sweep["mean_time_success_only"])
        )
        baseline_time_total_curve.append(float(baseline_sweep["mean_time_total"]))
        if generator_order_sweep is not None:
            generator_order_time_total_curve.append(
                float(generator_order_sweep["mean_time_total"])
            )
        if box_offline_sweep is not None:
            box_offline_time_total_curve.append(
                float(box_offline_sweep["mean_time_total"])
            )
        if oracle_steps_then_time_sweep is not None:
            oracle_steps_then_time_time_total_curve.append(
                float(oracle_steps_then_time_sweep["mean_time_total"])
            )
        encoder_time_total_curve.append(float(encoder_sweep["mean_time_total"]))

    baseline_success_curve_np = np.asarray(baseline_success_curve, dtype=np.float32)
    encoder_success_curve_np = np.asarray(encoder_success_curve, dtype=np.float32)
    generator_order_success_curve_np = (
        np.asarray(generator_order_success_curve, dtype=np.float32)
        if generator_order_enabled
        else None
    )
    box_offline_success_curve_np = (
        np.asarray(box_offline_success_curve, dtype=np.float32)
        if box_offline_enabled
        else None
    )
    oracle_steps_then_time_success_curve_np = (
        np.asarray(oracle_steps_then_time_success_curve, dtype=np.float32)
        if oracle_steps_then_time_enabled
        else None
    )
    baseline_time_success_only_curve_np = np.asarray(
        baseline_time_success_only_curve,
        dtype=np.float32,
    )
    encoder_time_success_only_curve_np = np.asarray(
        encoder_time_success_only_curve,
        dtype=np.float32,
    )
    generator_order_time_success_only_curve_np = (
        np.asarray(generator_order_time_success_only_curve, dtype=np.float32)
        if generator_order_enabled
        else None
    )
    box_offline_time_success_only_curve_np = (
        np.asarray(box_offline_time_success_only_curve, dtype=np.float32)
        if box_offline_enabled
        else None
    )
    oracle_steps_then_time_time_success_only_curve_np = (
        np.asarray(oracle_steps_then_time_time_success_only_curve, dtype=np.float32)
        if oracle_steps_then_time_enabled
        else None
    )
    baseline_time_total_curve_np = np.asarray(
        baseline_time_total_curve,
        dtype=np.float32,
    )
    encoder_time_total_curve_np = np.asarray(
        encoder_time_total_curve,
        dtype=np.float32,
    )
    generator_order_time_total_curve_np = (
        np.asarray(generator_order_time_total_curve, dtype=np.float32)
        if generator_order_enabled
        else None
    )
    box_offline_time_total_curve_np = (
        np.asarray(box_offline_time_total_curve, dtype=np.float32)
        if box_offline_enabled
        else None
    )
    oracle_steps_then_time_time_total_curve_np = (
        np.asarray(oracle_steps_then_time_time_total_curve, dtype=np.float32)
        if oracle_steps_then_time_enabled
        else None
    )

    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "paths": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "checkpoint_path": str(ckpt_path),
            "output_dir": str(out_dir),
        },
        "ranking": {
            "static_first_index": int(static_first_index),
            "top10_indices": global_order[:10].tolist(),
        },
        "generator_baseline": {
            "enabled": generator_order_enabled,
            "timeout_seconds": generator_timeout_seconds,
            "max_generated_skeletons_per_row": max_generated,
        },
        "box_offline_config": {
            "enabled": box_offline_enabled,
            "seed": int(cfg.box_offline.seed),
            "exploration_constant": float(cfg.box_offline.exploration_constant),
            "failure_penalty_multiplier": float(
                cfg.box_offline.failure_penalty_multiplier
            ),
        },
        "oracle_steps_then_time_config": {
            "enabled": oracle_steps_then_time_enabled,
        },
        "budget_seconds": budget_seconds,
        "baseline": {
            "success_rate": float(baseline_metrics["success_rate"]),
            "solved_count": int(baseline_metrics["solved_count"]),
            "failed_count": int(baseline_metrics["failed_count"]),
            "mean_time_success_only": float(baseline_metrics["mean_time_success_only"]),
            "mean_time_total": float(baseline_metrics["mean_time_total"]),
        },
        "encoder": {
            "success_rate": float(encoder_metrics["success_rate"]),
            "solved_count": int(encoder_metrics["solved_count"]),
            "failed_count": int(encoder_metrics["failed_count"]),
            "mean_time_success_only": float(encoder_metrics["mean_time_success_only"]),
            "mean_time_total": float(encoder_metrics["mean_time_total"]),
        },
    }
    if generator_order_metrics is not None:
        summary["baseline_generator_order"] = {
            "success_rate": float(generator_order_metrics["success_rate"]),
            "solved_count": int(generator_order_metrics["solved_count"]),
            "failed_count": int(generator_order_metrics["failed_count"]),
            "mean_time_success_only": float(
                generator_order_metrics["mean_time_success_only"]
            ),
            "mean_time_total": float(generator_order_metrics["mean_time_total"]),
        }
    if box_offline_metrics is not None:
        summary["box_offline"] = {
            "success_rate": float(box_offline_metrics["success_rate"]),
            "solved_count": int(box_offline_metrics["solved_count"]),
            "failed_count": int(box_offline_metrics["failed_count"]),
            "mean_time_success_only": float(
                box_offline_metrics["mean_time_success_only"]
            ),
            "mean_time_total": float(box_offline_metrics["mean_time_total"]),
        }
    if oracle_steps_then_time_metrics is not None:
        summary["oracle_steps_then_time"] = {
            "success_rate": float(oracle_steps_then_time_metrics["success_rate"]),
            "solved_count": int(oracle_steps_then_time_metrics["solved_count"]),
            "failed_count": int(oracle_steps_then_time_metrics["failed_count"]),
            "mean_time_success_only": float(
                oracle_steps_then_time_metrics["mean_time_success_only"]
            ),
            "mean_time_total": float(
                oracle_steps_then_time_metrics["mean_time_total"]
            ),
        }

    summary_path = out_dir / "offline_encoder_eval_summary.json"
    metrics_path = out_dir / "offline_encoder_eval_metrics.npz"
    success_curve_path = out_dir / "success_rate_vs_budget.png"
    success_only_time_path = out_dir / "time_success_only_vs_budget.png"
    total_time_path = out_dir / "time_total_vs_budget.png"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    np.savez(
        metrics_path,
        budgets=budgets.astype(np.float32),
        baseline_success_curve=baseline_success_curve_np,
        encoder_success_curve=encoder_success_curve_np,
        baseline_generator_success_curve=(
            generator_order_success_curve_np
            if generator_order_success_curve_np is not None
            else np.asarray([], dtype=np.float32)
        ),
        box_offline_success_curve=(
            box_offline_success_curve_np
            if box_offline_success_curve_np is not None
            else np.asarray([], dtype=np.float32)
        ),
        oracle_steps_then_time_success_curve=(
            oracle_steps_then_time_success_curve_np
            if oracle_steps_then_time_success_curve_np is not None
            else np.asarray([], dtype=np.float32)
        ),
        baseline_time_success_only_curve=baseline_time_success_only_curve_np,
        encoder_time_success_only_curve=encoder_time_success_only_curve_np,
        baseline_generator_time_success_only_curve=(
            generator_order_time_success_only_curve_np
            if generator_order_time_success_only_curve_np is not None
            else np.asarray([], dtype=np.float32)
        ),
        box_offline_time_success_only_curve=(
            box_offline_time_success_only_curve_np
            if box_offline_time_success_only_curve_np is not None
            else np.asarray([], dtype=np.float32)
        ),
        oracle_steps_then_time_time_success_only_curve=(
            oracle_steps_then_time_time_success_only_curve_np
            if oracle_steps_then_time_time_success_only_curve_np is not None
            else np.asarray([], dtype=np.float32)
        ),
        baseline_time_total_curve=baseline_time_total_curve_np,
        encoder_time_total_curve=encoder_time_total_curve_np,
        baseline_generator_time_total_curve=(
            generator_order_time_total_curve_np
            if generator_order_time_total_curve_np is not None
            else np.asarray([], dtype=np.float32)
        ),
        box_offline_time_total_curve=(
            box_offline_time_total_curve_np
            if box_offline_time_total_curve_np is not None
            else np.asarray([], dtype=np.float32)
        ),
        oracle_steps_then_time_time_total_curve=(
            oracle_steps_then_time_time_total_curve_np
            if oracle_steps_then_time_time_total_curve_np is not None
            else np.asarray([], dtype=np.float32)
        ),
        baseline_outcomes=baseline_metrics["outcomes"].astype(np.int8),
        encoder_outcomes=encoder_metrics["outcomes"].astype(np.int8),
        baseline_generator_outcomes=(
            generator_order_metrics["outcomes"].astype(np.int8)
            if generator_order_metrics is not None
            else np.asarray([], dtype=np.int8)
        ),
        box_offline_outcomes=(
            box_offline_metrics["outcomes"].astype(np.int8)
            if box_offline_metrics is not None
            else np.asarray([], dtype=np.int8)
        ),
        oracle_steps_then_time_outcomes=(
            oracle_steps_then_time_metrics["outcomes"].astype(np.int8)
            if oracle_steps_then_time_metrics is not None
            else np.asarray([], dtype=np.int8)
        ),
        baseline_elapsed_times=baseline_metrics["elapsed_times"].astype(np.float32),
        encoder_elapsed_times=encoder_metrics["elapsed_times"].astype(np.float32),
        baseline_generator_elapsed_times=(
            generator_order_metrics["elapsed_times"].astype(np.float32)
            if generator_order_metrics is not None
            else np.asarray([], dtype=np.float32)
        ),
        box_offline_elapsed_times=(
            box_offline_metrics["elapsed_times"].astype(np.float32)
            if box_offline_metrics is not None
            else np.asarray([], dtype=np.float32)
        ),
        oracle_steps_then_time_elapsed_times=(
            oracle_steps_then_time_metrics["elapsed_times"].astype(np.float32)
            if oracle_steps_then_time_metrics is not None
            else np.asarray([], dtype=np.float32)
        ),
        baseline_success_rate=np.float32(baseline_metrics["success_rate"]),
        encoder_success_rate=np.float32(encoder_metrics["success_rate"]),
        baseline_generator_success_rate=np.float32(
            generator_order_metrics["success_rate"]
            if generator_order_metrics is not None
            else np.nan
        ),
        box_offline_success_rate=np.float32(
            box_offline_metrics["success_rate"]
            if box_offline_metrics is not None
            else np.nan
        ),
        oracle_steps_then_time_success_rate=np.float32(
            oracle_steps_then_time_metrics["success_rate"]
            if oracle_steps_then_time_metrics is not None
            else np.nan
        ),
        baseline_mean_time_success_only=np.float32(
            baseline_metrics["mean_time_success_only"]
        ),
        encoder_mean_time_success_only=np.float32(
            encoder_metrics["mean_time_success_only"]
        ),
        baseline_generator_mean_time_success_only=np.float32(
            generator_order_metrics["mean_time_success_only"]
            if generator_order_metrics is not None
            else np.nan
        ),
        box_offline_mean_time_success_only=np.float32(
            box_offline_metrics["mean_time_success_only"]
            if box_offline_metrics is not None
            else np.nan
        ),
        oracle_steps_then_time_mean_time_success_only=np.float32(
            oracle_steps_then_time_metrics["mean_time_success_only"]
            if oracle_steps_then_time_metrics is not None
            else np.nan
        ),
        baseline_mean_time_total=np.float32(baseline_metrics["mean_time_total"]),
        encoder_mean_time_total=np.float32(encoder_metrics["mean_time_total"]),
        baseline_generator_mean_time_total=np.float32(
            generator_order_metrics["mean_time_total"]
            if generator_order_metrics is not None
            else np.nan
        ),
        box_offline_mean_time_total=np.float32(
            box_offline_metrics["mean_time_total"]
            if box_offline_metrics is not None
            else np.nan
        ),
        oracle_steps_then_time_mean_time_total=np.float32(
            oracle_steps_then_time_metrics["mean_time_total"]
            if oracle_steps_then_time_metrics is not None
            else np.nan
        ),
    )

    dpi = int(cfg.plot.dpi)
    _plot_success_curve(
        budgets,
        baseline_success_curve_np,
        generator_order_success_curve_np,
        box_offline_success_curve_np,
        oracle_steps_then_time_success_curve_np,
        encoder_success_curve_np,
        budget_seconds,
        success_curve_path,
        dpi,
    )
    _plot_time_curve(
        budgets,
        baseline_time_success_only_curve_np,
        generator_order_time_success_only_curve_np,
        box_offline_time_success_only_curve_np,
        oracle_steps_then_time_time_success_only_curve_np,
        encoder_time_success_only_curve_np,
        budget_seconds,
        title="Refinement time on success vs budget",
        ylabel="Mean time to success (seconds)",
        output_path=success_only_time_path,
        dpi=dpi,
    )
    _plot_time_curve(
        budgets,
        baseline_time_total_curve_np,
        generator_order_time_total_curve_np,
        box_offline_time_total_curve_np,
        oracle_steps_then_time_time_total_curve_np,
        encoder_time_total_curve_np,
        budget_seconds,
        title="Total refinement time vs budget (including failures)",
        ylabel="Mean total time per seed (seconds)",
        output_path=total_time_path,
        dpi=dpi,
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plot: {success_curve_path}")
    print(f"Saved plot: {success_only_time_path}")
    print(f"Saved plot: {total_time_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
