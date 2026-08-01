"""Which VLMPlan adapter and off-pool labeler an environment uses.

One place, so the two entry points (``vlmplan_run.py``, ``vlmplan_score.py``) dispatch
identically and a third environment is an entry rather than an ``if`` in each of them.

The two halves are registered separately on purpose. Generation needs only the adapter
and is the model-dependent, expensive stage; labelling needs a live refiner and is local
and free. Keeping them apart is what lets a run be re-scored after a re-collection
without re-querying the model — the same split ``score.py``'s module docstring argues
for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .adapter import EnvAdapter, Labeler


def make_adapter(
    env_variant: str, with_images: bool = True, image_width_px: int = 1024
) -> EnvAdapter:
    """The generation-side adapter for a collection."""
    if env_variant.startswith("stickbutton2d"):
        # pylint: disable=import-outside-toplevel
        from .sb2d_adapter import SB2DAdapter

        return SB2DAdapter(with_images=with_images, image_width_px=image_width_px)
    # pylint: disable=import-outside-toplevel
    from .dd2d_adapter import DD2DAdapter

    return DD2DAdapter(with_images=with_images, image_width_px=image_width_px)


def make_labeler_factory(
    env_variant: str, memo_path: Path | None = None
) -> Callable[[], Labeler]:
    """A zero-arg factory for the scoring-side labeler.

    A *factory* rather than an instance because ``score.label_agreement`` wants a fresh
    labeler per episode (so one episode's memo cannot mask another's disagreement),
    while ``score_sequence`` wants one shared across a whole run.
    """
    if env_variant.startswith("stickbutton2d"):
        # pylint: disable=import-outside-toplevel
        from .sb2d_label import SB2DOffPoolLabeler

        return lambda: SB2DOffPoolLabeler(memo_path)
    # pylint: disable=import-outside-toplevel
    from .score import OffPoolLabeler

    return lambda: OffPoolLabeler(memo_path=memo_path, env_variant=env_variant)
