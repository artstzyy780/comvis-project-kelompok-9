"""Training entry point for the controlled experiments (Checkpoint CP08, CP09).

The module trains one experiment defined in ``configs/experiments.yaml`` and
records everything the examiner may ask for afterwards:

- the exact hyper-parameters and the resolved dataset variant,
- the Ultralytics artefacts (``args.yaml``, ``results.csv``, confusion matrix),
- a run manifest with per-class validation metrics, the random seed, the
  library versions, and the wall-clock duration,
- the git commit that produced the checkpoint, so a result can always be traced
  back to a code version.

The Ultralytics import is lazy so that the module can be inspected and unit
tested on a machine without PyTorch installed.

Usage:
    python src/train.py --experiment Exp_A_Standard
    python src/train.py --all
"""

import argparse
import json
import logging
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    DATASET_DIR,
    DEFAULT_SEED,
    EXPERIMENTS_DIR,
    PROJECT_ROOT,
    ExperimentSpec,
    load_experiments,
)
from metrics import (
    macro_average,
    per_class_metrics_from_ultralytics,
    ultralytics_map_metrics,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("Train")

RUNS_DIR: Path = EXPERIMENTS_DIR / "runs"
DEFAULT_PROJECT_NAME: str = "runs_comvis"


@dataclass
class RunManifest:
    """Everything needed to reproduce and audit one training run.

    Attributes:
        experiment_id: Identifier of the trained experiment.
        display_name: Human readable experiment name.
        dataset_variant: Training data treatment used.
        dataset_yaml: Absolute path of the dataset configuration.
        hyperparams: Hyper-parameters handed to Ultralytics.
        seed: Random seed used for the run.
        device: Compute device used for training.
        started_at: ISO timestamp of the run start.
        finished_at: ISO timestamp of the run end.
        duration_seconds: Wall-clock duration of the training call.
        weights_path: Path of the best checkpoint.
        validation_metrics: Per-class metrics measured on the evaluation split.
        map50: mAP@50 of the evaluation split in percent.
        map50_95: mAP@50:95 of the evaluation split in percent.
        macro_precision: Macro-average precision in percent.
        macro_recall: Macro-average recall in percent.
        macro_f1: Macro-average F1 in percent.
        environment: Interpreter, platform, and library versions.
        git_commit: Commit hash of the working tree, when available.
    """

    experiment_id: str
    display_name: str
    dataset_variant: str
    dataset_yaml: str
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    seed: int = DEFAULT_SEED
    device: str = "0"
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    weights_path: str = ""
    validation_metrics: List[Dict[str, Any]] = field(default_factory=list)
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""


def _git_commit() -> str:
    """Return the current git commit hash, or an empty string when unavailable.

    Returns:
        The short commit hash of the working tree.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _library_versions() -> Dict[str, str]:
    """Collect the versions of the libraries that influence the results.

    Returns:
        Mapping of library name to version string.
    """
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name in ("ultralytics", "torch", "numpy", "opencv"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional dependency
            versions[module_name] = "not installed"
    return versions


def ensure_dataset_variant(spec: ExperimentSpec, dataset_root: Path) -> Path:
    """Materialise the dataset variant required by an experiment.

    For the ``raw`` variant the original dataset is used. For ``clahe`` and
    ``gamma`` the enhanced training split is generated on demand by
    :mod:`prepare_experiments`, which applies the transformation to the
    training images only and leaves the evaluation splits untouched.

    Args:
        spec: Experiment specification.
        dataset_root: Root of the original dataset.

    Returns:
        Path of the ``data.yaml`` that must be handed to Ultralytics.
    """
    if spec.dataset_variant == "raw":
        return dataset_root / "data.yaml"

    from prepare_experiments import build_augmented_split

    variant_dir: Path = PROJECT_ROOT / f"dataset_{spec.dataset_variant}"
    build_augmented_split(dataset_root, variant_dir, method=spec.dataset_variant)
    return variant_dir / f"data_{spec.dataset_variant}.yaml"


def train_experiment(
    spec: ExperimentSpec,
    dataset_root: Path,
    device: str,
    epochs_override: Optional[int] = None,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> RunManifest:
    """Train a single experiment and persist its audit trail.

    Args:
        spec: Experiment specification from the registry.
        dataset_root: Root of the original dataset.
        device: Device string understood by Ultralytics ('0', 'cpu', ...).
        epochs_override: Optional epoch count that overrides the registry.
        project_name: Ultralytics project folder for the run artefacts.

    Returns:
        The RunManifest describing the finished run.

    Raises:
        ImportError: When Ultralytics is not installed in the environment.
    """
    from ultralytics import YOLO  # Imported lazily: PyTorch is only needed to train.

    manifest: RunManifest = RunManifest(
        experiment_id=spec.experiment_id,
        display_name=spec.display_name,
        dataset_variant=spec.dataset_variant,
        dataset_yaml="",
        seed=int(spec.hyperparams.get("seed", DEFAULT_SEED)),
        device=device,
        environment=_library_versions(),
        git_commit=_git_commit(),
    )

    data_yaml: Path = ensure_dataset_variant(spec, dataset_root)
    manifest.dataset_yaml = str(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    hyperparams: Dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(epochs_override or spec.hyperparams.get("epochs", 50)),
        "imgsz": int(spec.hyperparams.get("imgsz", 640)),
        "batch": int(spec.hyperparams.get("batch", 16)),
        "seed": manifest.seed,
        "optimizer": spec.hyperparams.get("optimizer", "auto"),
        "lr0": float(spec.hyperparams.get("lr0", 0.01)),
        "cos_lr": bool(spec.hyperparams.get("cos_lr", True)),
        "patience": int(spec.hyperparams.get("patience", 100)),
        "project": str(RUNS_DIR),
        "name": spec.experiment_id,
        "device": device,
        "exist_ok": True,
        "verbose": True,
    }

    # Pass fine-tuning transfer learning parameters (anti-catastrophic forgetting)
    for opt_param in ("freeze", "scale", "mosaic", "mixup", "lrf", "warmup_epochs"):
        if opt_param in spec.hyperparams:
            hyperparams[opt_param] = spec.hyperparams[opt_param]

    manifest.hyperparams = hyperparams

    logger.info("Melatih %s (%s)", spec.experiment_id, spec.display_name)
    logger.info("Dataset : %s", data_yaml)
    logger.info("Hyper   : %s", hyperparams)

    model = YOLO(spec.hyperparams.get("pretrained", "yolov8n.pt"))
    started: float = time.perf_counter()
    manifest.started_at = datetime.now(timezone.utc).isoformat()

    model.train(**hyperparams)

    manifest.duration_seconds = round(time.perf_counter() - started, 2)
    manifest.finished_at = datetime.now(timezone.utc).isoformat()

    best_path: Path = RUNS_DIR / spec.experiment_id / "weights" / "best.pt"
    manifest.weights_path = str(best_path) if best_path.exists() else ""

    # Validate on the split declared by the registry and store per-class metrics.
    evaluation_split: str = spec.evaluation_split
    logger.info("Evaluasi %s pada split '%s'", spec.experiment_id, evaluation_split)
    validation = model.val(data=str(data_yaml), split=evaluation_split, device=device)

    class_metrics = per_class_metrics_from_ultralytics(validation)
    manifest.validation_metrics = [asdict(metric) for metric in class_metrics]
    manifest.map50, manifest.map50_95 = ultralytics_map_metrics(validation)
    if class_metrics:
        precision, recall, f1 = macro_average(class_metrics)
        manifest.macro_precision = precision
        manifest.macro_recall = recall
        manifest.macro_f1 = f1

    manifest_dir: Path = RUNS_DIR / spec.experiment_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path: Path = manifest_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2, ensure_ascii=False)
    logger.info("Manifest: %s", manifest_path)

    return manifest


def run_all(
    dataset_root: Path,
    device: str,
    only: Optional[List[str]] = None,
    epochs_override: Optional[int] = None,
) -> List[RunManifest]:
    """Train every registered experiment in a deterministic order.

    Args:
        dataset_root: Root of the original dataset.
        device: Device string understood by Ultralytics.
        only: Optional subset of experiment identifiers to run.
        epochs_override: Optional epoch count applied to every run.

    Returns:
        List of RunManifest objects for the executed runs.
    """
    manifests: List[RunManifest] = []
    for spec in load_experiments():
        if spec.is_zero_shot:
            logger.info("Melewati %s (zero-shot, tanpa pelatihan)", spec.experiment_id)
            continue
        if only and spec.experiment_id not in only:
            continue
        manifests.append(
            train_experiment(
                spec,
                dataset_root=dataset_root,
                device=device,
                epochs_override=epochs_override,
            )
        )
    return manifests


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Latih eksperimen terkontrol K09.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="ID eksperimen tunggal, mis. Exp_A_Standard",
    )
    parser.add_argument("--all", action="store_true", help="Jalankan semua eksperimen fine-tuned")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--device", type=str, default="0", help="Device Ultralytics ('0' atau 'cpu')")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch")
    args = parser.parse_args()

    if not args.experiment and not args.all:
        parser.error("Pilih --experiment <ID> atau --all.")

    if args.all:
        run_all(
            dataset_root=args.dataset_root,
            device=args.device,
            epochs_override=args.epochs,
        )
        return

    specs: List[ExperimentSpec] = load_experiments()
    selected = next((s for s in specs if s.experiment_id == args.experiment), None)
    if selected is None:
        raise SystemExit(f"Unknown experiment: {args.experiment}")

    train_experiment(
        selected,
        dataset_root=args.dataset_root,
        device=args.device,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
