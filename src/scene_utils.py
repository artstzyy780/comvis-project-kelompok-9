"""Scene-level dataset utilities: source-scene grouping, near-duplicate
detection, and leakage-free (grouped) split construction.

Why this module exists (Checkpoint CP03 / CP07)
-------------------------------------------------
The Roboflow ``Low-light v2`` export ships several *augmented copies* of every
source scene. The file name encodes the origin, for example::

    4ec75dfc-e66d0059_augmented_0_jpg.rf.5fbc58531a9424e8a4da9dcb99e0e749.jpg
    4ec75dfc-e66d0059_augmented_2_jpg.rf.22d77ebd0b645594bcd1c26862924753.jpg

Both files stem from the *same* photographic scene, yet the exporter distributed
them across ``train``, ``valid`` and ``test``. A scene therefore leaks between
the training set and the evaluation set, which inflates every reported metric.

Two independent detectors are provided:

1. ``extract_scene_id`` — cheap, exact, driven by the file naming convention.
2. ``compute_dhash`` / ``find_visual_near_duplicates`` — perceptual hashing
   (difference hash) that catches duplicates the naming convention misses.

Example:
    >>> groups = group_by_scene([Path("a_augmented_0_jpg.rf.x.jpg")])
    >>> sorted(groups)
    ['a']
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from config import SplitLeakageReport


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
AUGMENTED_MARKER: str = "_augmented_"
ROBOFLOW_HASH_MARKER: str = ".rf."
IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_DHASH_SIZE: int = 8
DEFAULT_DHASH_MAX_DISTANCE: int = 5
DEFAULT_SPLIT_RATIOS: Dict[str, float] = {"train": 0.70, "valid": 0.15, "test": 0.15}


# ----------------------------------------------------------------------------
# Scene Identification
# ----------------------------------------------------------------------------
def extract_scene_id(image_path: Path) -> str:
    """Derive the source-scene identifier from a Roboflow file name.

    The identifier is the part of the file stem before the augmentation marker,
    stripped from the Roboflow ``.rf.<hash>`` suffix and the trailing image
    extension token (e.g. ``_jpg``).

    Args:
        image_path: Path (or plain name) of the image file.

    Returns:
        The source-scene identifier.

    Example:
        >>> extract_scene_id(Path("abc_augmented_0_jpg.rf.deadbeef.jpg"))
        'abc'
    """
    stem: str = Path(image_path).stem

    if ROBOFLOW_HASH_MARKER in stem:
        stem = stem.split(ROBOFLOW_HASH_MARKER)[0]

    if AUGMENTED_MARKER in stem:
        stem = stem.split(AUGMENTED_MARKER)[0]

    for suffix in ("_jpg", "_jpeg", "_png", "_bmp", "_webp"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    return stem.strip("_")


def collect_images(root: Path, split: Optional[str] = None) -> List[Path]:
    """List image files under a dataset split directory.

    Args:
        root: Dataset root containing ``train``/``valid``/``test`` folders.
        split: Optional split name; when omitted all splits are scanned.

    Returns:
        Sorted list of image paths.
    """
    splits: List[str] = [split] if split else ["train", "valid", "test"]
    images: List[Path] = []
    for name in splits:
        images_dir: Path = root / name / "images"
        if not images_dir.exists():
            continue
        images.extend(
            sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        )
    return images


def group_by_scene(image_paths: Iterable[Path]) -> Dict[str, List[Path]]:
    """Group image paths by their source scene.

    Args:
        image_paths: Iterable of image paths.

    Returns:
        Mapping of scene identifier to the list of its image copies.
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in image_paths:
        groups[extract_scene_id(path)].append(path)
    return dict(groups)


# ----------------------------------------------------------------------------
# Perceptual Hashing (Visual Near-Duplicate Detection)
# ----------------------------------------------------------------------------
def compute_dhash(image: np.ndarray, hash_size: int = DEFAULT_DHASH_SIZE) -> str:
    """Compute the difference hash (dHash) of an image.

    The image is reduced to ``hash_size + 1`` by ``hash_size`` pixels in
    grayscale; each output bit encodes whether the right pixel is brighter than
    its left neighbour. The result is a perceptual fingerprint that is robust
    to rescaling, mild cropping, and brightness jitter.

    Args:
        image: Input image as a BGR or grayscale NumPy array.
        hash_size: Size of the square hash (8 -> 64-bit fingerprint).

    Returns:
        Hexadecimal string of length ``hash_size ** 2``.

    Raises:
        ValueError: If the image array is empty or not 2-dimensional.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D image array, got shape {image.shape}")

    grayscale: np.ndarray = (
        image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    resized: np.ndarray = cv2.resize(
        grayscale, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )
    diff: np.ndarray = resized[:, 1:] > resized[:, :-1]
    bits: str = "".join("1" if value else "0" for value in diff.flatten())
    return f"{int(bits, 2):0{hash_size ** 2 // 4}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute the Hamming distance between two hexadecimal dHash strings.

    Args:
        hash_a: First hexadecimal fingerprint.
        hash_b: Second hexadecimal fingerprint.

    Returns:
        Number of differing bits.

    Raises:
        ValueError: If the fingerprints have different lengths.
    """
    if len(hash_a) != len(hash_b):
        raise ValueError(
            f"Hash length mismatch: {len(hash_a)} != {len(hash_b)}"
        )
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def _popcount(values: np.ndarray) -> np.ndarray:
    """Count the set bits of every element of a uint64 array.

    Uses ``np.bitwise_count`` when the installed NumPy exposes it (NumPy >= 2.0)
    and falls back to a byte-wise ``unpackbits`` implementation otherwise, so
    the module keeps working on the older NumPy shipped with some Colab images.

    Args:
        values: Array of unsigned 64-bit integers.

    Returns:
        Array of bit counts with the same shape as the input.
    """
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values).astype(np.uint8)

    contiguous: np.ndarray = np.ascontiguousarray(values)
    byte_view: np.ndarray = contiguous.view(np.uint8).reshape(*contiguous.shape, 8)
    return np.unpackbits(byte_view, axis=-1).sum(axis=-1).astype(np.uint8)


def find_visual_near_duplicates(
    image_paths: Sequence[Path],
    max_distance: int = DEFAULT_DHASH_MAX_DISTANCE,
    hash_size: int = DEFAULT_DHASH_SIZE,
) -> List[Tuple[Path, Path, int]]:
    """Find visually near-duplicate image pairs using perceptual hashing.

    The comparison is vectorised with NumPy bit operations, so the cost for
    a few thousand images stays below a second.

    Args:
        image_paths: Images to compare.
        max_distance: Maximum Hamming distance still considered a duplicate.
        hash_size: Size of the perceptual hash.

    Returns:
        List of (path_a, path_b, distance) triples, sorted by distance.
    """
    if len(image_paths) < 2:
        return []

    hashes: List[str] = []
    kept: List[Path] = []
    for path in image_paths:
        image: Optional[np.ndarray] = cv2.imread(str(path))
        if image is None:
            continue
        hashes.append(compute_dhash(image, hash_size))
        kept.append(path)

    if len(hashes) < 2:
        return []

    values: np.ndarray = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    xor_matrix: np.ndarray = np.bitwise_xor(values[:, None], values[None, :])
    distances: np.ndarray = _popcount(xor_matrix)

    pairs: List[Tuple[Path, Path, int]] = []
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            distance: int = int(distances[i, j])
            if distance <= max_distance:
                pairs.append((kept[i], kept[j], distance))

    pairs.sort(key=lambda item: item[2])
    return pairs


# ----------------------------------------------------------------------------
# Leakage Analysis
# ----------------------------------------------------------------------------
def detect_cross_split_leakage(
    dataset_root: Path,
    splits: Sequence[str] = ("train", "valid", "test"),
) -> SplitLeakageReport:
    """Detect source scenes that appear in more than one split.

    Args:
        dataset_root: Dataset root with ``<split>/images`` folders.
        splits: Split names to inspect.

    Returns:
        A SplitLeakageReport describing the overlap.
    """
    scene_to_splits: Dict[str, List[str]] = defaultdict(list)
    per_split_total: Dict[str, int] = {}
    per_split_overlap: Dict[str, int] = {split: 0 for split in splits}
    per_split_overlap_files: Dict[str, int] = {split: 0 for split in splits}
    split_scene_sets: Dict[str, Set[str]] = {}

    for split in splits:
        paths: List[Path] = collect_images(dataset_root, split)
        per_split_total[split] = len(paths)
        scenes_in_split: Set[str] = {extract_scene_id(path) for path in paths}
        split_scene_sets[split] = scenes_in_split
        for scene in scenes_in_split:
            scene_to_splits[scene].append(split)

    # A scene leaks when it appears in more than one split.
    overlapping: List[str] = []
    for scene, found in scene_to_splits.items():
        if len(found) > 1:
            overlapping.append(scene)
            for split in found:
                if split in per_split_overlap:
                    per_split_overlap[split] += 1

    # File-level impact: how many images of each split are affected.
    for split, scenes_in_split in split_scene_sets.items():
        affected: int = sum(
            1
            for path in collect_images(dataset_root, split)
            if len(scene_to_splits[extract_scene_id(path)]) > 1
        )
        per_split_overlap_files[split] = affected

    return SplitLeakageReport(
        scene_to_splits={scene: sorted(splits_found) for scene, splits_found in scene_to_splits.items()},
        overlapping_scenes=sorted(overlapping),
        per_split_overlap=per_split_overlap,
        per_split_overlap_files=per_split_overlap_files,
        per_split_total=per_split_total,
        total_scenes=len(scene_to_splits),
    )


def scene_has_class(
    labels_dir: Path,
    image_stem: str,
    class_id: int,
) -> bool:
    """Check whether a label file contains at least one box of a given class.

    Args:
        labels_dir: Directory holding the YOLO label files.
        image_stem: File stem of the image (without extension).
        class_id: Target class identifier.

    Returns:
        True when the annotation contains the class, False otherwise.
    """
    label_path: Path = labels_dir / f"{image_stem}.txt"
    if not label_path.exists():
        return False
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            tokens: List[str] = line.strip().split()
            if tokens and tokens[0].isdigit() and int(tokens[0]) == class_id:
                return True
    return False


# ----------------------------------------------------------------------------
# Grouped Split Construction
# ----------------------------------------------------------------------------
def _allocate_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """Distribute ``total`` items over splits proportionally to ``ratios``.

    Args:
        total: Number of items to distribute.
        ratios: Mapping of split name to target ratio.

    Returns:
        Mapping of split name to integer count; the counts sum to ``total``.
    """
    names: List[str] = list(ratios.keys())
    raw: List[float] = [total * ratios[name] for name in names]
    counts: List[int] = [int(value) for value in raw]

    # Hand out the remainder to the splits with the largest fractional part.
    remainders: List[Tuple[float, int]] = sorted(
        ((raw[i] - counts[i], i) for i in range(len(names))),
        key=lambda item: item[0],
        reverse=True,
    )
    leftover: int = total - sum(counts)
    for _, index in remainders[:leftover]:
        counts[index] += 1

    return {name: counts[index] for index, name in enumerate(names)}


def assign_scenes_to_splits(
    scene_ids: Sequence[str],
    ratios: Dict[str, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign source scenes to splits deterministically.

    The assignment is proportional to ``ratios`` and fully reproducible: the
    same ``seed`` always yields the same split for every scene.

    Args:
        scene_ids: Scene identifiers to distribute.
        ratios: Target ratio per split.
        seed: Random seed controlling the shuffle.

    Returns:
        Mapping of scene identifier to split name.
    """
    if not scene_ids:
        return {}

    names: List[str] = list(ratios.keys())
    counts: Dict[str, int] = _allocate_counts(len(scene_ids), ratios)

    # Sort first so that the input order cannot influence the outcome.
    ordered: List[str] = sorted(scene_ids)
    rng = np.random.default_rng(seed)
    shuffled: List[str] = [ordered[index] for index in rng.permutation(len(ordered))]

    assignment: Dict[str, str] = {}
    cursor: int = 0
    for name in names:
        take: int = counts[name]
        for scene in shuffled[cursor : cursor + take]:
            assignment[scene] = name
        cursor += take
    return assignment


def assign_scenes_stratified(
    scenes: Dict[str, bool],
    ratios: Dict[str, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign scenes to splits while stratifying by person-presence.

    Scenes are split into two strata (with and without pedestrians) and each
    stratum is distributed independently. This keeps the pedestrian signal
    balanced across train/valid/test, which matters because the pedestrian
    class is the rarest and therefore the most fragile to under-sample.

    Args:
        scenes: Mapping of scene identifier to a boolean "contains pedestrian".
        ratios: Target ratio per split.
        seed: Random seed controlling the shuffle.

    Returns:
        Mapping of scene identifier to split name.
    """
    with_person: List[str] = [scene for scene, flag in scenes.items() if flag]
    without_person: List[str] = [scene for scene, flag in scenes.items() if not flag]

    assignment: Dict[str, str] = {}
    assignment.update(assign_scenes_to_splits(with_person, ratios, seed))
    # Offset the seed so the two strata are not correlated permutations.
    assignment.update(assign_scenes_to_splits(without_person, ratios, seed + 1))
    return assignment


if __name__ == "__main__":
    from config import DATASET_DIR

    report: SplitLeakageReport = detect_cross_split_leakage(DATASET_DIR)
    print(f"Total source scenes      : {report.total_scenes}")
    print(f"Scenes spanning >1 split : {len(report.overlapping_scenes)}")
    for split, count in report.per_split_total.items():
        overlap: int = report.per_split_overlap.get(split, 0)
        share: float = (overlap / count * 100) if count else 0.0
        print(f"  {split:5s}: {overlap:3d}/{count:4d} scene bocor ({share:.1f}%)")
