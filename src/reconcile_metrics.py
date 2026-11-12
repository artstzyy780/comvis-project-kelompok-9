"""Reconcile every reported metric into one auditable Markdown table.

Why this module exists
----------------------
The first version of the result report mixed two different metrics in a single
table:

- ``B0`` used per-class ``person`` recall from :mod:`evaluate_baseline`.
- ``Exp-A/B/C`` used Ultralytics ``metrics/recall(B)``, which is the **mean
  recall over all classes** (macro), not the recall of the ``person`` class.

:func:`metrics.per_class_metrics_from_ultralytics` already documents this
pitfall. This module reads the raw JSON artefacts, keeps the original numbers,
and renders one document that labels each metric explicitly. It never re-runs a
model, so it works without ``torch``/``ultralytics`` installed.

Usage:
    python src/reconcile_metrics.py
    python src/reconcile_metrics.py --output results/experiments/METRICS_RECONCILIATION.md
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import EXPERIMENTS_DIR, PERSON_CLASS_ID


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("ReconcileMetrics")

BENCHMARK_JSON: Path = EXPERIMENTS_DIR / "benchmark_latency_metrics.json"
BASELINE_JSON: Path = EXPERIMENTS_DIR / "baseline_b0_metrics.json"
COLAB_JSON: Path = EXPERIMENTS_DIR / "colab_training_benchmark_results.json"
DEFAULT_OUTPUT: Path = EXPERIMENTS_DIR / "METRICS_RECONCILIATION.md"

REALTIME_TARGET_FPS: float = 20.0


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON artefact.

    Args:
        path: Location of the JSON file.

    Returns:
        The parsed JSON document.

    Raises:
        FileNotFoundError: When the artefact is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Artefact not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def class_entry(entry: Dict[str, Any], class_id: int) -> Optional[Dict[str, Any]]:
    """Return the per-class metrics block of one class.

    Args:
        entry: One benchmark entry from ``benchmark_latency_metrics.json``.
        class_id: Class identifier to look up.

    Returns:
        The matching class metrics, or ``None`` when the class is absent.
    """
    for item in entry.get("class_metrics", []):
        if int(item.get("class_id", -1)) == class_id:
            return item
    return None


def pct(value: Optional[float], digits: int = 2) -> str:
    """Format a numeric percentage.

    Args:
        value: Percentage value, or ``None`` when unavailable.
        digits: Number of decimal digits.

    Returns:
        A string such as ``66.67%`` or ``-``.
    """
    return "-" if value is None else f"{value:.{digits}f}%"


def pp(value: Optional[float], digits: int = 2) -> str:
    """Format a delta in percentage points.

    Args:
        value: Delta value in percentage points, or ``None``.
        digits: Number of decimal digits.

    Returns:
        A signed string such as ``+46.16 pp`` or ``-``.
    """
    if value is None:
        return "-"
    return f"{value:+.{digits}f} pp"


def benchmark_index(benchmarks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index benchmark entries by model identifier."""
    return {str(entry["model_id"]): entry for entry in benchmarks}


def build_primary_rows(benchmarks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the per-class ``person`` table for the test split.

    Args:
        benchmarks: The ``benchmarks`` list of the latency artefact.

    Returns:
        Rows with model, split, image count, ground truth, person P/R/F1 and the
        recall delta against B0 in percentage points.
    """
    index: Dict[str, Dict[str, Any]] = benchmark_index(benchmarks)
    baseline_person: Optional[Dict[str, Any]] = class_entry(
        index.get("B0_ZeroShot", {}), PERSON_CLASS_ID
    )
    baseline_recall: Optional[float] = (
        float(baseline_person["recall"]) if baseline_person else None
    )

    rows: List[Dict[str, Any]] = []
    for model_id in ("B0_ZeroShot", "Exp_A_Standard", "Exp_B_CLAHE", "Exp_C_Gamma"):
        entry: Optional[Dict[str, Any]] = index.get(model_id)
        if entry is None:
            continue
        person: Optional[Dict[str, Any]] = class_entry(entry, PERSON_CLASS_ID)
        if person is None:
            continue
        recall: float = float(person["recall"])
        if baseline_recall is None and model_id == "B0_ZeroShot":
            baseline_recall = recall
        delta: Optional[float] = (
            None
            if model_id == "B0_ZeroShot"
            else (recall - baseline_recall if baseline_recall is not None else None)
        )
        rows.append(
            {
                "model_id": model_id,
                "name": entry.get("display_name", model_id),
                "split": entry.get("split", "-"),
                "images": int(entry.get("image_count", 0)),
                "ground_truth": int(person.get("ground_truth", 0)),
                "recall": recall,
                "precision": float(person.get("precision", 0.0)),
                "f1": float(person.get("f1_score", 0.0)),
                "delta_vs_b0": delta,
            }
        )
    return rows


def build_macro_rows(
    baseline: Dict[str, Any], colab: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Build the validation table with explicit macro/per-class separation.

    Args:
        baseline: Parsed ``baseline_b0_metrics.json``.
        colab: Parsed ``colab_training_benchmark_results.json``.

    Returns:
        Rows carrying macro recall, all-class mAP@50, and person recall when the
        artefact actually provides it.
    """
    baseline_person: Optional[Dict[str, Any]] = None
    for item in baseline.get("per_class_metrics", []):
        if int(item.get("class_id", -1)) == PERSON_CLASS_ID:
            baseline_person = item
            break

    rows: List[Dict[str, Any]] = [
        {
            "model_id": "B0",
            "name": "Zero-Shot Baseline (COCO)",
            # baseline_b0_metrics.json stores fractions (0.4228), not percentages.
            "macro_recall": float(baseline["mean_recall"]) * 100.0,
            "map50": None,
            "person_recall": (
                float(baseline_person["recall"]) * 100.0 if baseline_person else None
            ),
            "source": "evaluate_baseline.py (valid, conf 0.25)",
        }
    ]

    for item in colab.get("results", []):
        model_id: str = str(item.get("experiment_id", "?"))
        if model_id == "B0":
            continue
        rows.append(
            {
                "model_id": model_id,
                "name": item.get("name", model_id),
                # The legacy JSON field is named recall_person but stores the
                # Ultralytics macro recall; keep the value, fix the label.
                "macro_recall": float(item.get("recall_person", 0.0)) * 100.0,
                "map50": (
                    float(item["map50"]) * 100.0 if item.get("map50") is not None else None
                ),
                "person_recall": None,
                "source": "metrics/recall(B) (valid, max-F1)",
            }
        )
    return rows


def build_latency_rows(benchmarks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the latency/throughput table for the test split."""
    rows: List[Dict[str, Any]] = []
    for entry in benchmarks:
        latency: Dict[str, Any] = entry.get("latency", {})
        rows.append(
            {
                "model_id": entry.get("model_id", "?"),
                "name": entry.get("display_name", "?"),
                "mean_ms": float(latency.get("total_mean_ms", 0.0)),
                "p50_ms": float(latency.get("p50_ms", 0.0)),
                "p95_ms": float(latency.get("p95_ms", 0.0)),
                "fps": float(latency.get("fps", 0.0)),
                "meets_realtime": bool(latency.get("meets_realtime", False)),
            }
        )
    return rows


def render_markdown(
    benchmarks: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    colab: Dict[str, Any],
) -> str:
    """Render the reconciliation document.

    Args:
        benchmarks: The ``benchmarks`` list of the latency artefact.
        baseline: Parsed ``baseline_b0_metrics.json``.
        colab: Parsed ``colab_training_benchmark_results.json``.

    Returns:
        Markdown text.
    """
    primary: List[Dict[str, Any]] = build_primary_rows(benchmarks)
    macro: List[Dict[str, Any]] = build_macro_rows(baseline, colab)
    latency: List[Dict[str, Any]] = build_latency_rows(benchmarks)

    lines: List[str] = [
        "# Rekonsiliasi Metrik — Satu Sumber Kebenaran",
        "",
        "> Dihasilkan otomatis oleh `python src/reconcile_metrics.py`.",
        "> Modul ini **tidak** menjalankan model; ia hanya membaca artefak JSON yang sudah ada.",
        "",
        "## 1. Masalah yang diperbaiki",
        "",
        "- `metrics/recall(B)` dari Ultralytics adalah **rata-rata recall semua kelas** (macro),",
        "  bukan recall kelas `person`.",
        "- Tabel hasil sebelumnya mencampur `B0` (person recall) dengan `Exp-A/B/C` (macro recall),",
        "  sehingga selisih seperti `+34,18%` tidak apple-to-apple.",
        "- `src/metrics.py::per_class_metrics_from_ultralytics` sudah menulis peringatan ini;",
        "  dokumen ini memakai angka per-kelas yang memang tersedia.",
        "",
        "## 2. Tabel utama — Test split, metrik per-kelas `person`",
        "",
        f"Sumber: `{BENCHMARK_JSON.name}` (conf 0.25; IoU 0.5; split test).",
        "",
        "| Model | Skenario | n citra | GT `person` | Recall `person` | Precision `person` | F1 `person` | Delta Recall vs B0 |",
        "|:---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| **{model}** | {name} | {images} | {gt} | **{recall}** | {precision} | {f1} | {delta} |".format(
                model=row["model_id"],
                name=row["name"],
                images=row["images"],
                gt=row["ground_truth"],
                recall=pct(row["recall"]),
                precision=pct(row["precision"]),
                f1=pct(row["f1"]),
                delta=pp(row["delta_vs_b0"]),
            )
        )

    lines.extend(
        [
            "",
            "**Bacaan:** fine-tuning adalah pendorong utama kenaikan recall.",
            "Gamma (Exp-C) hampir setara dengan fine-tuning standar (Exp-A), sedangkan CLAHE (Exp-B)",
            "pada sampel test ini lebih rendah meski precision-nya lebih tinggi.",
            "Selisih antar-varian belum diuji signifikansinya (lihat Keterbatasan).",
            "",
            "## 3. Tabel pembanding — Validation split (macro vs per-kelas)",
            "",
            "| Model | Macro Recall (semua kelas) | mAP@50 (semua kelas) | Recall `person` | Sumber metrik |",
            "|:---:|---:|---:|---:|:---|",
        ]
    )
    for row in macro:
        macro_value: str = pct(row["macro_recall"])
        if row["model_id"] == "B0":
            macro_value += "*"
        lines.append(
            "| **{model}** | {macro} | {map50} | {person} | {source} |".format(
                model=row["model_id"],
                macro=macro_value,
                map50=pct(row["map50"]),
                person=pct(row["person_recall"]),
                source=row["source"],
            )
        )

    lines.extend(
        [
            "",
            "> \\*B0 dihitung pada conf 0.25 oleh `evaluate_baseline.py`, sedangkan",
            "> Exp-A/B/C memakai operating point max-F1. Baris B0 hanya konteks,",
            "> bukan pembanding langsung.",
            "",
            "> **Peringatan:** kolom `Exp-A/B/C` di tabel ini adalah `metrics/recall(B)`",
            "> (macro, operating point max-F1), bukan recall `person` pada conf 0.25.",
            "> Jangan membandingkannya langsung dengan Tabel 2. Recall `person` validasi",
            "> untuk Exp-A/B/C harus dihitung ulang dengan",
            "> `src/metrics.py::per_class_metrics_from_ultralytics` atau `src/evaluate_experiments.py`.",
            "",
            "## 4. Latensi & throughput — Test split (CPU)",
            "",
            f"Target proyek: >= {REALTIME_TARGET_FPS:.0f} FPS.",
            "",
            "| Model | Rata-rata | P50 | P95 | FPS | Memenuhi target |",
            "|:---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in latency:
        lines.append(
            "| **{model}** | {mean:.2f} ms | {p50:.2f} ms | {p95:.2f} ms | {fps:.2f} | {status} |".format(
                model=row["model_id"],
                mean=row["mean_ms"],
                p50=row["p50_ms"],
                p95=row["p95_ms"],
                fps=row["fps"],
                status="Ya" if row["meets_realtime"] else "Tidak",
            )
        )

    lines.extend(
        [
            "",
            "> Kolom `preprocess_mean_ms` pada laporan benchmark adalah preprocessing internal",
            "> Ultralytics (resize/normalize), **bukan** CLAHE/Gamma. Biaya CLAHE/Gamma",
            "> end-to-end belum diukur.",
            "",
            "## 5. Keterbatasan yang wajib disebut",
            "",
            "1. **Split test masih bocor level scene.** Audit menemukan 55% file test berasal dari",
            "   scene yang juga ada di train; split `dataset_dedup/` sudah dibangun tetapi belum",
            "   dipakai untuk melatih atau mengevaluasi model final.",
            "2. **Sampel kecil.** Tabel test hanya berisi 39 box `person` (18 citra), dan satu seed (42),",
            "   sehingga selisih beberapa poin persen belum tentu signifikan. Interval Wilson/bootstrap",
            "   sudah tersedia di `src/metrics.py` tetapi belum dilampirkan pada tabel hasil.",
            "3. **Operating point berbeda.** Benchmark memakai conf 0.25; Ultralytics validation",
            "   melaporkan max-F1. Keduanya tidak boleh dicampur.",
            "4. **Tidak ada bukti run pelatihan.** `results/experiments/runs/*/run_manifest.json`",
            "   dan `results/experiments/unified_results.md` belum ada.",
            "5. **Freeze artefak perlu diperbarui.** README dan laporan benchmark diedit setelah",
            "   pembekuan 2026-10-20, sehingga `python src/freeze_artifacts.py --check` akan",
            "   melaporkan digest berbeda. Jalankan ulang freeze setelah laporan final selesai.",
            "",
            "## 6. Cara menghitung ulang secara konsisten",
            "",
            "```bash",
            "# Evaluasi per-kelas pada satu split, dengan interval keyakinan:",
            "python src/evaluate_experiments.py --split test --dataset-root dataset_dedup",
            "",
            "# Benchmark latensi + metrik per-kelas:",
            "python src/benchmark_inference.py",
            "```",
            "",
            "Setelah `unified_results.json` tersedia, jalankan ulang modul ini agar Tabel 3",
            "memakai recall `person` yang sebenarnya.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Rekonsiliasi metrik macro vs per-kelas menjadi satu laporan."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    benchmarks: List[Dict[str, Any]] = load_json(BENCHMARK_JSON).get("benchmarks", [])
    baseline: Dict[str, Any] = load_json(BASELINE_JSON)
    colab: Dict[str, Any] = load_json(COLAB_JSON)

    markdown: str = render_markdown(benchmarks, baseline, colab)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    logger.info("Laporan rekonsiliasi ditulis ke: %s", args.output)


if __name__ == "__main__":
    main()
