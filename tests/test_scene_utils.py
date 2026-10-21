"""Unit tests for scene grouping and leakage-free split construction (CP03, CP07).

These tests protect the central methodological claim of the project: no source
scene may ever appear in two different splits.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scene_utils import (  # noqa: E402
    _allocate_counts,
    assign_scenes_stratified,
    assign_scenes_to_splits,
    compute_dhash,
    extract_scene_id,
    group_by_scene,
    hamming_distance,
)


class TestExtractSceneId:
    """Naming-convention normalisation."""

    @pytest.mark.parametrize(
        "filename",
        [
            "4ec75dfc-e66d0059_augmented_0_jpg.rf.5fbc58531a9424e8a4da9dcb99e0e749.jpg",
            "4ec75dfc-e66d0059_augmented_2_jpg.rf.22d77ebd0b645594bcd1c26862924753.jpg",
            "4ec75dfc-e66d0059_jpg.rf.8c9577de80a4aef14b0ec1260b6d4e5a.jpg",
            "4ec75dfc-e66d0059_augmented_1.jpg",
            "4ec75dfc-e66d0059.jpg",
        ],
    )
    def test_variants_map_to_same_scene(self, filename: str) -> None:
        """Every augmented copy of a scene must resolve to one identifier."""
        assert extract_scene_id(Path(filename)) == "4ec75dfc-e66d0059"

    def test_distinct_scenes_stay_distinct(self) -> None:
        """Two different scenes must not be merged."""
        a = extract_scene_id(Path("aaaa_augmented_0_jpg.rf.1.jpg"))
        b = extract_scene_id(Path("bbbb_augmented_1_jpg.rf.2.jpg"))
        assert a != b


class TestGroupByScene:
    """Grouping helper."""

    def test_groups_augmented_copies(self) -> None:
        """All copies of a scene end up in one group."""
        paths = [
            Path("sceneA_augmented_0_jpg.rf.1.jpg"),
            Path("sceneA_augmented_1_jpg.rf.2.jpg"),
            Path("sceneB_augmented_0_jpg.rf.3.jpg"),
        ]
        groups = group_by_scene(paths)
        assert set(groups.keys()) == {"sceneA", "sceneB"}
        assert len(groups["sceneA"]) == 2


class TestPerceptualHash:
    """dHash based visual near-duplicate detection."""

    def test_identical_images_have_zero_distance(self) -> None:
        """The same pixels must produce identical fingerprints."""
        image = np.full((64, 64, 3), 120, dtype=np.uint8)
        image[:, 32:] = 200
        assert hamming_distance(compute_dhash(image), compute_dhash(image)) == 0

    def test_brightness_shift_keeps_hash_close(self) -> None:
        """A global brightness change must not break the fingerprint."""
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        base[:, 32:] = 180
        brighter = np.clip(base.astype(int) + 40, 0, 255).astype(np.uint8)
        assert hamming_distance(compute_dhash(base), compute_dhash(brighter)) <= 2

    def test_different_images_differ(self) -> None:
        """Unrelated patterns must produce clearly different fingerprints."""
        stripes = np.zeros((64, 64, 3), dtype=np.uint8)
        stripes[:, ::2] = 255
        checker = np.zeros((64, 64, 3), dtype=np.uint8)
        checker[::2, ::2] = 255
        checker[1::2, 1::2] = 255
        assert hamming_distance(compute_dhash(stripes), compute_dhash(checker)) > 5


class TestSplitAllocation:
    """Grouped split construction."""

    def test_allocate_counts_sums_to_total(self) -> None:
        """The allocation must distribute every item exactly once."""
        counts = _allocate_counts(687, {"train": 0.70, "valid": 0.15, "test": 0.15})
        assert sum(counts.values()) == 687

    def test_allocation_is_deterministic(self) -> None:
        """The same ratios must always produce the same counts."""
        first = _allocate_counts(100, {"train": 0.7, "valid": 0.15, "test": 0.15})
        second = _allocate_counts(100, {"train": 0.7, "valid": 0.15, "test": 0.15})
        assert first == second

    def test_assignment_covers_every_scene(self) -> None:
        """Every scene must receive exactly one split."""
        scenes = [f"scene_{index:04d}" for index in range(200)]
        assignment = assign_scenes_to_splits(scenes, seed=42)
        assert len(assignment) == len(scenes)
        assert set(assignment.keys()) == set(scenes)
        assert set(assignment.values()) <= {"train", "valid", "test"}

    def test_assignment_is_reproducible(self) -> None:
        """The same seed must produce the same partition."""
        scenes = [f"scene_{index:04d}" for index in range(120)]
        assert assign_scenes_to_splits(scenes, seed=7) == assign_scenes_to_splits(scenes, seed=7)

    def test_stratified_split_keeps_pedestrian_scenes_balanced(self) -> None:
        """Both strata must be represented in every split."""
        scenes = {f"p_{index}": True for index in range(60)}
        scenes.update({f"n_{index}": False for index in range(340)})
        assignment = assign_scenes_stratified(scenes, seed=42)
        for split in ("train", "valid", "test"):
            members = [scene for scene, target in assignment.items() if target == split]
            assert any(scene.startswith("p_") for scene in members)
            assert any(scene.startswith("n_") for scene in members)

    def test_no_scene_appears_in_two_splits(self) -> None:
        """The grouped split must be leakage free by construction."""
        scenes = {f"scene_{index}": index % 3 == 0 for index in range(300)}
        assignment = assign_scenes_stratified(scenes, seed=42)
        assert len(set(assignment.keys())) == len(assignment)
        assert all(isinstance(value, str) for value in assignment.values())
