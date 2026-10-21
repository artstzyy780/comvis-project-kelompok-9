"""Central configuration layer for the K09 Low-light Pedestrian Detection project.

This module is the single source of truth for paths, class taxonomy, evaluation
thresholds, experiment registry, and success criteria. Every other module in
``src/`` imports from here so that no experiment parameter is ever hard-coded
in more than one place.

Design goals:
- Reproducibility: one place to change the random seed or the evaluation split.
- Auditability: the experiment registry is data (YAML), not code branches.
- CP04 evidence: success criteria are declared explicitly, not implied.

Example:
    >>> from config import load_experiments, load_success_criteria
    >>> experiments = load_experiments()
    >>> criteria = load_success_criteria()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import yaml


# ----------------------------------------------------------------------------
# Project Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset"
DEDUP_DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset_dedup"
CONFIGS_DIR: Final[Path] = PROJECT_ROOT / "configs"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
AUDIT_DIR: Final[Path] = RESULTS_DIR / "audit"
EXPERIMENTS_DIR: Final[Path] = RESULTS_DIR / "experiments"
PROPOSAL_DIR: Final[Path] = RESULTS_DIR / "proposal"

EXPERIMENTS_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "experiments.yaml"
SUCCESS_CRITERIA_PATH: Final[Path] = CONFIGS_DIR / "success_criteria.yaml"


# ----------------------------------------------------------------------------
# Reproducibility & Training Constants
# ----------------------------------------------------------------------------
DEFAULT_SEED: Final[int] = 42
DEFAULT_EPOCHS: Final[int] = 50
DEFAULT_IMGSZ: Final[int] = 640
DEFAULT_BATCH: Final[int] = 16
DEFAULT_LR0: Final[float] = 0.01


# ----------------------------------------------------------------------------
# Class Taxonomy (mirrors dataset/data.yaml)
# ----------------------------------------------------------------------------
VEHICLE_CLASS_ID: Final[int] = 0
PERSON_CLASS_ID: Final[int] = 1
CLASS_NAMES: Final[Dict[int, str]] = {
    VEHICLE_CLASS_ID: "Vehicle",
    PERSON_CLASS_ID: "person",
}
CLASS_COLORS: Final[Dict[int, Tuple[int, int, int]]] = {
    VEHICLE_CLASS_ID: (255, 128, 0),  # BGR orange-blue for Vehicle
    PERSON_CLASS_ID: (0, 255, 0),  # BGR green for Person
}
NUM_CLASSES: Final[int] = len(CLASS_NAMES)


# ----------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationConfig:
    """Thresholds and runtime settings used during quantitative evaluation.

    Attributes:
        conf_threshold: Minimum confidence for a detection to be counted.
        iou_threshold: IoU value above which a prediction matches a ground truth.
        nms_iou: IoU threshold used by the Ultralytics NMS implementation.
        iou_thresholds: Sweep of IoU values used to approximate mAP@50:95.
        realtime_fps_target: Frame-per-second threshold for the real-time claim.
        warmup_iterations: Leading iterations excluded from latency statistics.
        img_size: Inference resolution (square, in pixels).
        bootstrap_samples: Number of bootstrap resamples for confidence intervals.
        confidence_level: Two-sided confidence level for confidence intervals.
    """

    conf_threshold: float = 0.25
    iou_threshold: float = 0.50
    nms_iou: float = 0.45
    iou_thresholds: Tuple[float, ...] = tuple(0.5 + 0.05 * i for i in range(10))
    realtime_fps_target: float = 20.0
    warmup_iterations: int = 5
    img_size: int = 640
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95


@dataclass(frozen=True)
class ExperimentSpec:
    """Declarative description of one controlled experiment.

    Attributes:
        experiment_id: Short identifier used in file names and reports.
        display_name: Human readable name for slides and reports.
        dataset_variant: Training data treatment ('raw', 'clahe', or 'gamma').
        description: One sentence summary of the experiment.
        independent_variable: The single factor that differs from Exp-A.
        evaluation_split: Split used for the headline metrics.
        hyperparams: Training hyper-parameters shared by all fine-tuned runs.
    """

    experiment_id: str
    display_name: str
    dataset_variant: str
    description: str
    independent_variable: str
    evaluation_split: str
    is_zero_shot: bool
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory backing this experiment variant."""
        if self.dataset_variant == "raw":
            return DATASET_DIR
        return PROJECT_ROOT / f"dataset_{self.dataset_variant}"


@dataclass(frozen=True)
class SuccessCriterion:
    """A measurable success criterion declared for the project (CP04).

    Attributes:
        metric_id: Machine readable identifier.
        metric_name: Human readable metric name.
        class_id: Optional class the criterion applies to (None = macro).
        operator: Comparison operator ('>=', '<=', '>').
        target: Numeric threshold value.
        unit: Unit of the target value (ratio, percent, fps).
        rationale: Why this target matters for the project.
    """

    metric_id: str
    metric_name: str
    class_id: Optional[int]
    operator: str
    target: float
    unit: str
    rationale: str

    def is_satisfied(self, value: float) -> bool:
        """Evaluate whether a measured value satisfies this criterion.

        Args:
            value: Measured metric value.

        Returns:
            True when the comparison holds, False otherwise.
        """
        if self.operator == ">=":
            return value >= self.target
        if self.operator == "<=":
            return value <= self.target
        if self.operator == ">":
            return value > self.target
        if self.operator == "<":
            return value < self.target
        raise ValueError(f"Unsupported operator: {self.operator}")


@dataclass(frozen=True)
class SplitLeakageReport:
    """Summary of scene-level leakage between dataset splits (CP03).

    Attributes:
        scene_to_splits: Mapping of scene identifier to the splits it appears in.
        overlapping_scenes: Scene identifiers present in more than one split.
        per_split_overlap: Number of scenes per split that also appear elsewhere.
        per_split_overlap_files: Number of image files per split whose scene
            also appears in another split.
        per_split_total: Number of image files per split.
        total_scenes: Number of distinct source scenes in the dataset.
    """

    scene_to_splits: Dict[str, List[str]]
    overlapping_scenes: List[str]
    per_split_overlap: Dict[str, int]
    per_split_overlap_files: Dict[str, int]
    per_split_total: Dict[str, int]
    total_scenes: int

    @property
    def has_leakage(self) -> bool:
        """Return True when at least one scene spans more than one split."""
        return len(self.overlapping_scenes) > 0


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML document from disk.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed mapping; empty dict when the file contains only comments.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def load_evaluation_config() -> EvaluationConfig:
    """Build the evaluation configuration from YAML with sane fallbacks.

    Returns:
        An EvaluationConfig instance.
    """
    raw: Dict[str, Any] = load_yaml(CONFIGS_DIR / "evaluation.yaml").get("evaluation", {})
    defaults = EvaluationConfig()
    iou_raw = raw.get("iou_thresholds")
    iou_values: Tuple[float, ...] = (
        tuple(float(v) for v in iou_raw) if iou_raw else defaults.iou_thresholds
    )
    return EvaluationConfig(
        conf_threshold=float(raw.get("conf_threshold", defaults.conf_threshold)),
        iou_threshold=float(raw.get("iou_threshold", defaults.iou_threshold)),
        nms_iou=float(raw.get("nms_iou", defaults.nms_iou)),
        iou_thresholds=iou_values,
        realtime_fps_target=float(raw.get("realtime_fps_target", defaults.realtime_fps_target)),
        warmup_iterations=int(raw.get("warmup_iterations", defaults.warmup_iterations)),
        img_size=int(raw.get("img_size", defaults.img_size)),
        bootstrap_samples=int(raw.get("bootstrap_samples", defaults.bootstrap_samples)),
        confidence_level=float(raw.get("confidence_level", defaults.confidence_level)),
    )


def load_experiments() -> List[ExperimentSpec]:
    """Load the controlled-experiment registry from ``configs/experiments.yaml``.

    Returns:
        List of ExperimentSpec objects in declaration order.

    Raises:
        ValueError: If the registry is empty or a fine-tuned entry has no dataset.
    """
    raw: Dict[str, Any] = load_yaml(EXPERIMENTS_CONFIG_PATH)
    entries: List[Dict[str, Any]] = raw.get("experiments", [])
    if not entries:
        raise ValueError(f"No experiments defined in {EXPERIMENTS_CONFIG_PATH}")

    specs: List[ExperimentSpec] = []
    for entry in entries:
        specs.append(
            ExperimentSpec(
                experiment_id=str(entry["id"]),
                display_name=str(entry["display_name"]),
                dataset_variant=str(entry["dataset_variant"]),
                description=str(entry.get("description", "")),
                independent_variable=str(entry.get("independent_variable", "")),
                evaluation_split=str(entry.get("evaluation_split", "test")),
                is_zero_shot=bool(entry.get("is_zero_shot", False)),
                hyperparams=dict(entry.get("hyperparams", {})),
            )
        )
    return specs


def get_experiment(experiment_id: str) -> ExperimentSpec:
    """Retrieve a single experiment by its identifier.

    Args:
        experiment_id: Identifier such as 'Exp_A_Standard'.

    Returns:
        The matching ExperimentSpec.

    Raises:
        KeyError: If the identifier is not registered.
    """
    for spec in load_experiments():
        if spec.experiment_id == experiment_id:
            return spec
    raise KeyError(f"Unknown experiment id: {experiment_id}")


def load_success_criteria() -> List[SuccessCriterion]:
    """Load the declared success criteria for the project (CP04).

    Returns:
        List of SuccessCriterion objects.
    """
    raw: Dict[str, Any] = load_yaml(SUCCESS_CRITERIA_PATH)
    criteria: List[SuccessCriterion] = []
    for entry in raw.get("criteria", []):
        class_id = entry.get("class_id")
        criteria.append(
            SuccessCriterion(
                metric_id=str(entry["id"]),
                metric_name=str(entry["metric_name"]),
                class_id=int(class_id) if class_id is not None else None,
                operator=str(entry.get("operator", ">=")),
                target=float(entry["target"]),
                unit=str(entry.get("unit", "ratio")),
                rationale=str(entry.get("rationale", "")),
            )
        )
    return criteria


def evaluate_criteria(
    criteria: List[SuccessCriterion],
    measured: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Check measured metric values against the declared success criteria.

    Args:
        criteria: Declared success criteria.
        measured: Mapping of metric_id to measured value.

    Returns:
        One dictionary per criterion with the verdict and the gap to target.
    """
    results: List[Dict[str, Any]] = []
    for criterion in criteria:
        value: Optional[float] = measured.get(criterion.metric_id)
        satisfied: Optional[bool] = (
            criterion.is_satisfied(value) if value is not None else None
        )
        gap: Optional[float] = (
            value - criterion.target if value is not None else None
        )
        results.append(
            {
                "metric_id": criterion.metric_id,
                "metric_name": criterion.metric_name,
                "operator": criterion.operator,
                "target": criterion.target,
                "measured": value,
                "satisfied": satisfied,
                "gap": gap,
                "rationale": criterion.rationale,
            }
        )
    return results


if __name__ == "__main__":
    for spec in load_experiments():
        print(f"{spec.experiment_id:20s} | variant={spec.dataset_variant:6s} | {spec.description}")
    print()
    for criterion in load_success_criteria():
        print(f"{criterion.metric_id:28s} {criterion.operator} {criterion.target} ({criterion.unit})")
