"""Tests for the kinder-rendered StickButton2D PIGINet crop source.

``SB2DKinderDomain`` reads per-object crop PNGs that ``sb2d_render_convert.py``
materialises from kinder's own renderer, in place of ``SB2DDomain``'s schematic (a lone
polygon on a blank background). The failure modes guarded here are all silent: a factory
that returns the wrong crop source, a crop-window constant that drifts from the
converter, a converter that mutates the record it was only supposed to re-image, and a
render that is not reproducible from the seed (which would make the "reconstruct from
seed" exception unsound).

The fixture converts two real ``stickbutton2d_v1`` episodes into a tmp tree, so the test
exercises the actual convert -> read path rather than a mock. It skips when the v1
collection (gitignored) is absent.
"""

from __future__ import annotations

# Test idioms pylint's defaults flag: imports inside tests (C0415) so collection is cheap
# and skips when data is absent; fixtures passed by name (W0621); poking module/env
# internals on purpose (W0212). `sb2d_render_convert` (an experiments/ script) is loaded
# by file path via `_load_convert`: pytest puts the test's own dir -- not the repo root
# -- on sys.path, so `from experiments.spectre import ...` does not resolve.
# pylint: disable=C0415,W0621,W0212
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_V1 = _ROOT / "data" / "spectre" / "raw" / "stickbutton2d_v1"

pytestmark = pytest.mark.skipif(
    not (_V1 / "test" / "episodes").is_dir(),
    reason="stickbutton2d_v1 collection not present (gitignored data)",
)


def _load_convert() -> "ModuleType":
    """Import ``experiments/spectre/sb2d_render_convert.py`` by file path.

    pytest puts the test's directory on ``sys.path``, not the repo root, so
    ``from experiments.spectre import sb2d_render_convert`` does not resolve under
    ``pytest tests/``. The script imports only ``alphatamp.*`` + third-party, so loading
    it by path is self-contained. Mirrors ``test_refine_cap._load_precompute``.
    """
    import importlib.util

    path = _ROOT / "experiments" / "spectre" / "sb2d_render_convert.py"
    spec = importlib.util.spec_from_file_location("sb2d_render_convert", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Convert one b1 and one b5 episode into a tmp ``stickbutton2d_v1_kinder`` tree.

    Returns ``(data_root, [(pid, source_episode), ...])``. Two episodes keep it fast
    while still covering a multi-button scene (b5) for the context / non-degeneracy
    checks.
    """
    # pylint: disable=import-outside-toplevel
    import kinder

    from alphatamp.approaches.spectre.collect import episode_path
    from alphatamp.approaches.spectre.env_registry import register_extra_envs
    from alphatamp.approaches.spectre.envs.stickbutton2d.strata import decode, env_id
    from alphatamp.approaches.spectre.io import (
        atomic_write_pickle_gz,
        list_episodes,
        load_episode,
    )

    conv = _load_convert()

    register_extra_envs()
    paths = list_episodes(_V1 / "test")
    picks = [paths[0], paths[-1]]  # first (b1) and last (b5)
    data_root = tmp_path_factory.mktemp("spectre_kinder")
    out = []
    env_cache: dict[int, object] = {}
    for p in picks:
        ep = load_episode(p)
        pid = ep.provenance.problem_id
        _s, nb, _i = decode(pid)
        env = env_cache.setdefault(nb, kinder.make(env_id(nb), render_mode="rgb_array"))
        images_dir = (
            data_root / "raw" / "stickbutton2d_v1_kinder" / "test" / "images" / str(pid)
        )
        conv._render_problem_images(env, ep, images_dir, overwrite=False)
        new_ep = replace(
            ep, provenance=replace(ep.provenance, env_variant="stickbutton2d_v1_kinder")
        )
        atomic_write_pickle_gz(
            new_ep, episode_path(data_root, "stickbutton2d_v1_kinder", "test", pid)
        )
        out.append((pid, ep))
    for env in env_cache.values():
        env.close()
    return data_root, out


# --------------------------------------------------------------------------- #
# factory + constants
# --------------------------------------------------------------------------- #
def test_factory_dispatches_on_variant() -> None:
    """The kinder variant gets the PNG-reading domain; everything else the schematic."""
    from alphatamp.approaches.spectre.baselines.piginet.sb2d_adapter import (
        SB2DDomain,
        SB2DKinderDomain,
        make_sb2d_domain,
    )

    # Exact-class checks (`.__class__ is`), not isinstance: SB2DKinderDomain subclasses
    # SB2DDomain, so isinstance cannot tell the two crop sources apart.
    assert make_sb2d_domain("x", "stickbutton2d_v1").__class__ is SB2DDomain
    assert (
        make_sb2d_domain("x", "stickbutton2d_v1_kinder").__class__ is SB2DKinderDomain
    )
    # Unknown variants fall back to the schematic, so the factory is a strict superset.
    assert make_sb2d_domain("x", "who_knows").__class__ is SB2DDomain


def test_crop_window_constant_stays_in_sync() -> None:
    """The converter frames each object at the same world window the adapter documents.

    They live in two modules; a silent drift would reframe every kinder crop against the
    scale the schematic (and the pose/shape features) assume.
    """
    from alphatamp.approaches.spectre.baselines.piginet import sb2d_adapter

    conv = _load_convert()

    assert conv.CROP_WORLD == sb2d_adapter._CROP_WORLD


# --------------------------------------------------------------------------- #
# the record is only re-imaged, never mutated (justifies the SPECTRE graft)
# --------------------------------------------------------------------------- #
def test_records_are_copied_verbatim(converted) -> None:
    """Everything but the images is byte-identical, so SPECTRE's inputs cannot change.

    This is what licenses grafting SPECTRE's numbers from ``stickbutton2d_v1`` rather
    than retraining on the kinder variant.
    """
    from alphatamp.approaches.spectre.collect import episode_path
    from alphatamp.approaches.spectre.io import load_episode

    data_root, items = converted
    for pid, src in items:
        got = load_episode(
            episode_path(data_root, "stickbutton2d_v1_kinder", "test", pid)
        )
        assert got.provenance.env_variant == "stickbutton2d_v1_kinder"
        assert got.scene_geometry == src.scene_geometry
        assert got.skeleton_pool == src.skeleton_pool
        assert got.outcomes == src.outcomes
        assert got.object_registry == src.object_registry
        assert got.goal_atoms == src.goal_atoms


# --------------------------------------------------------------------------- #
# crops
# --------------------------------------------------------------------------- #
def test_crops_cover_every_object_and_scene_png_exists(converted) -> None:
    """One crop per geometry object, plus a full-scene overview for future use."""
    from alphatamp.approaches.spectre.baselines.piginet.sb2d_adapter import (
        SB2DKinderDomain,
    )

    data_root, items = converted
    dom = SB2DKinderDomain(data_root, "stickbutton2d_v1_kinder")
    for pid, src in items:
        spid = f"sb2d_s{pid}"
        crops = dom.crops("test", spid)
        assert set(crops) == {o.name for o in src.scene_geometry.objects}
        assert all(min(img.size) > 0 for img in crops.values())
        scene = (
            data_root
            / "raw"
            / "stickbutton2d_v1_kinder"
            / "test"
            / "images"
            / str(pid)
            / "scene.png"
        )
        assert scene.is_file(), "full-scene render should be materialised too"


def test_missing_png_is_omitted_not_errored(converted) -> None:
    """A missing crop becomes a zero vector downstream, matching DD2D's stored-PNG
    path."""
    from alphatamp.approaches.spectre.baselines.piginet.sb2d_adapter import (
        SB2DKinderDomain,
    )

    data_root, items = converted
    pid = items[0][0]
    (
        data_root
        / "raw"
        / "stickbutton2d_v1_kinder"
        / "test"
        / "images"
        / str(pid)
        / "robot.png"
    ).unlink()
    dom = SB2DKinderDomain(data_root, "stickbutton2d_v1_kinder")
    crops = dom.crops("test", f"sb2d_s{pid}")
    assert "robot" not in crops and crops, "missing PNG omitted, the rest still present"


def test_button_crops_carry_context_not_identical(converted) -> None:
    """The point of the change: kinder crops are *not* the schematic's identical discs.

    In the schematic every unpressed button renders as the same red disc on a blank
    background, so their crops are pixel-identical. A per-object crop taken from the true
    scene carries positional context (which wall, how near the table band), so on a
    multi-button scene the button crops differ. This is the direct contrast to
    ``test_piginet_sb2d.test_crops_preserve_relative_scale``.
    """
    from alphatamp.approaches.spectre.baselines.piginet.sb2d_adapter import (
        SB2DKinderDomain,
    )

    data_root, items = converted
    # the b5 episode -- the one with several buttons
    pid, src = max(items, key=lambda it: len(it[1].scene_geometry.objects))
    button_names = [
        o.name for o in src.scene_geometry.objects if o.name.startswith("button")
    ]
    assert len(button_names) >= 2, "need a multi-button scene for this contrast"
    dom = SB2DKinderDomain(data_root, "stickbutton2d_v1_kinder")
    crops = dom.crops("test", f"sb2d_s{pid}")
    arrs = [np.asarray(crops[b].resize((96, 96))) for b in button_names]
    all_identical = all(np.array_equal(arrs[0], a) for a in arrs[1:])
    assert not all_identical, "kinder button crops should differ by scene context"


def test_render_is_deterministic_from_the_seed(converted) -> None:
    """Re-rendering the same problem reproduces the same pixels.

    ``env.reset(seed=pid)`` is the sanctioned "reconstruct from seed" exception; if it
    or the renderer were nondeterministic the kinder crops would be an unreproducible
    artifact.
    """
    # pylint: disable=import-outside-toplevel
    import kinder
    from kinder.envs.utils import render_2dstate

    from alphatamp.approaches.spectre.env_registry import register_extra_envs
    from alphatamp.approaches.spectre.envs.stickbutton2d.strata import decode, env_id

    CROP_WORLD = _load_convert().CROP_WORLD

    _data_root, items = converted
    pid, src = items[0]
    _s, nb, _i = decode(pid)
    register_extra_envs()

    def _render_first_object() -> np.ndarray:
        env = kinder.make(env_id(nb), render_mode="rgb_array")
        env.reset(seed=pid)
        oc = env.unwrapped._object_centric_env  # type: ignore[attr-defined]  # pylint: disable=protected-access
        state = oc._current_state.copy()  # pylint: disable=protected-access
        state.data.update(oc.initial_constant_state.data)
        obj = src.scene_geometry.objects[0]
        cx, cy = float(obj.pose[0]), float(obj.pose[1])
        half = CROP_WORLD / 2.0
        img = render_2dstate(
            state,
            oc._static_object_body_cache,  # pylint: disable=protected-access
            cx - half,
            cx + half,
            cy - half,
            cy + half,
            int(oc.config.render_dpi),
        )
        env.close()
        return img

    assert np.array_equal(_render_first_object(), _render_first_object())
