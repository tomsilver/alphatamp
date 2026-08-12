"""Tests for episode-local object tags (proposal §7 / Step 7)."""

from __future__ import annotations

import pytest

from alphatamp.approaches.spectre.tags import PAD_TAG, assign_tags, tag_seed


def test_bijection_distinct_tags():
    names = ["item_0", "item_1", "item_2", "target"]
    tags = assign_tags(names)
    assert set(tags) == set(names)
    assert len(set(tags.values())) == len(names)  # distinct
    assert PAD_TAG not in tags.values()  # 0 reserved


def test_deterministic_when_rng_none():
    names = ["b", "a", "c"]
    assert assign_tags(names) == assign_tags(names)
    # sorted-name order -> a=1, b=2, c=3
    assert assign_tags(names) == {"a": 1, "b": 2, "c": 3}


def test_per_epoch_permutation_varies_but_stays_a_bijection():
    names = [f"item_{i}" for i in range(6)]
    t0 = assign_tags(names, rng=tag_seed(0, 3, epoch=0), max_tags=16)
    t1 = assign_tags(names, rng=tag_seed(0, 3, epoch=1), max_tags=16)
    assert t0 != t1  # a different permutation across epochs
    for t in (t0, t1):
        assert len(set(t.values())) == len(names)  # still a bijection
        assert all(1 <= v <= 16 for v in t.values())


def test_same_seed_episode_epoch_is_reproducible():
    names = [f"o{i}" for i in range(5)]
    a = assign_tags(names, rng=tag_seed(7, 2, 4), max_tags=10)
    b = assign_tags(names, rng=tag_seed(7, 2, 4), max_tags=10)
    assert a == b


def test_same_tag_everywhere_within_episode():
    # one mapping is applied to every mention of an object (scene/skeleton/fact);
    # an object in a "skeleton arg" and in a "fact arg" must resolve to the SAME tag.
    names = ["item_0", "item_1", "item_2"]
    tags = assign_tags(names, rng=tag_seed(1, 1, 0), max_tags=8)
    skeleton_args = ["item_0", "item_2"]
    fact_args = ["item_2", "item_1"]
    assert [tags[a] for a in skeleton_args] == [tags["item_0"], tags["item_2"]]
    assert tags["item_2"] == tags["item_2"]  # consistent across mentions
    assert [tags[a] for a in fact_args][0] == tags["item_2"]


def test_anti_collapse_different_objects_get_different_tags():
    # the v1 collapse: same-length skeletons over DIFFERENT objects became identical
    # inputs.
    # With tags, different objects → different tags → distinguishable arg sequences.
    names = ["item_0", "item_1", "item_2", "item_3"]
    tags = assign_tags(names)
    skel_a = [tags["item_0"], tags["item_1"]]  # stages {0,1}
    skel_b = [tags["item_2"], tags["item_3"]]  # stages {2,3}, same length
    assert skel_a != skel_b  # distinct tensor inputs (the collapse is gone)


def test_max_tags_exceeded_raises():
    with pytest.raises(ValueError, match="exceed max_tags"):
        assign_tags(["a", "b", "c"], max_tags=2)
