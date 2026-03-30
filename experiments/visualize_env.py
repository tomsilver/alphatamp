"""Visualize a kinder environment at a given seed.

Renders the initial state of the specified kinder environment and displays
or saves the resulting image.

Usage::

    python experiments/visualize_env.py
    python experiments/visualize_env.py --env obstruction2d --complexity 3 --seed 42
    python experiments/visualize_env.py --env clutteredstorage2d
        --complexity 2 --seed 7 --save out.png
"""

from __future__ import annotations

import argparse

import kinder
import matplotlib.pyplot as plt
import numpy as np

# Registry of supported kinder environments.
# Each entry maps a short name → (env_id_template, model_name, complexity_kwarg).
_ENV_REGISTRY: dict[str, tuple[str, str, str]] = {
    "clutteredretrieval2d": (
        "kinder/ClutteredRetrieval2D-o{n}-v0",
        "clutteredretrieval2d",
        "num_obstructions",
    ),
    "obstruction2d": (
        "kinder/Obstruction2D-o{n}-v0",
        "obstruction2d",
        "num_obstructions",
    ),
    "dynobstruction2d": (
        "kinder/DynObstruction2D-o{n}-v0",
        "dynobstruction2d",
        "num_obstructions",
    ),
    "clutteredstorage2d": (
        "kinder/ClutteredStorage2D-b{n}-v0",
        "clutteredstorage2d",
        "num_boxes",
    ),
}


def main() -> None:
    """Prints initial image of the specified environment."""
    parser = argparse.ArgumentParser(
        description="Visualize a kinder environment at a given seed."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="obstruction2d",
        choices=sorted(_ENV_REGISTRY),
        help="Environment short name (default: obstruction2d).",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=1,
        help="Complexity level, e.g. number of obstructions or boxes (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for env.reset() (default: 0).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If provided, save the image to this path instead of displaying.",
    )
    args = parser.parse_args()

    env_id_template, _, _ = _ENV_REGISTRY[args.env]
    env_id = env_id_template.format(n=args.complexity)

    kinder.register_all_environments()
    env = kinder.make(env_id, render_mode="rgb_array")

    env.reset(seed=args.seed)
    frame: np.ndarray = env.render()  # type: ignore[assignment]
    assert frame is not None

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(frame)
    ax.set_title(f"{env_id}  (seed={args.seed})")
    ax.axis("off")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
