"""Unit tests for the metric reconciliation report.

These tests pin down the exact defect that motivated the module: the old report
mixed per-class ``person`` recall (B0) with Ultralytics macro recall
(``metrics/recall(B)``) for Exp-A/B/C. The reconciliation must keep the two
tables separate and must convert fraction-valued baseline artefacts to percent.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reconcile_metrics import (  # noqa: E402
    build_macro_rows,
    build_primary_rows,
    pct,
    pp,
    render_markdown,
)


def _benchmark(model_id: str, recall: float, precision: float = 50.0) -> dict:
    """Build a minimal benchmark entry with one ``person`` class block."""
    return {
        "model_id": model_id,
        "display_name": model_id,
        "split": "test",
        "image_count": 100,
        "latency": {
            "total_mean_ms": 50.0,
            "p50_ms": 49.0,
            "p95_ms": 60.0,
            "fps": 19.0,
            "meets_realtime": False,
        },
        "class_metrics": [
            {
                "class_id": 1,
                "class_name": "person",
                "ground_truth": 39,
                "precision": precision,
                "recall": recall,
                "f1_score": 50.0,
            }
        ],
    }


def _baseline() -> dict:
    """Build a baseline artefact with fraction-valued aggregates."""
    return {
        "mean_recall": 0.4228,
        "mean_precision": 0.3996,
        "mean_f1": 0.3981,
        "per_class_metrics": [
            {"class_id": 0, "class_name": "Vehicle", "recall": 0.4544},
            {"class_id": 1, "class_name": "person", "recall": 0.3913},
        ],
    }


def _colab() -> dict:
    """Build a Colab artefact whose legacy field stores macro recall."""
    return {
        "results": [
            {"experiment_id": "B0", "recall_person": 0.391, "map50": 0.38},
            {"experiment_id": "Exp-A", "recall_person": 0.732823, "map50": 0.78124},
            {"experiment_id": "Exp-B", "recall_person": 0.69267, "map50": 0.777252},
            {"experiment_id": "Exp-C", "recall_person": 0.734092, "map50": 0.741687},
        ]
    }


class TestFormatting:
    """Percentage and percentage-point formatting."""

    def test_pct(self) -> None:
        assert pct(66.666) == "66.67%"
        assert pct(None) == "-"

    def test_pp(self) -> None:
        assert pp(46.156) == "+46.16 pp"
        assert pp(None) == "-"


class TestPrimaryTable:
    """Per-class table uses only the benchmark artefact."""

    def test_delta_is_measured_against_b0(self) -> None:
        rows = build_primary_rows(
            [
                _benchmark("B0_ZeroShot", 20.51),
                _benchmark("Exp_A_Standard", 66.67),
                _benchmark("Exp_B_CLAHE", 48.72),
                _benchmark("Exp_C_Gamma", 64.10),
            ]
        )
        assert [row["model_id"] for row in rows] == [
            "B0_ZeroShot",
            "Exp_A_Standard",
            "Exp_B_CLAHE",
            "Exp_C_Gamma",
        ]
        assert rows[0]["delta_vs_b0"] is None
        assert rows[1]["delta_vs_b0"] == pytest.approx(46.16)
        assert rows[2]["delta_vs_b0"] == pytest.approx(28.21)
        assert rows[3]["delta_vs_b0"] == pytest.approx(43.59)


class TestMacroTable:
    """Macro and per-class values must never be silently merged."""

    def test_baseline_fractions_are_converted_to_percent(self) -> None:
        rows = build_macro_rows(_baseline(), _colab())
        b0 = rows[0]
        assert b0["macro_recall"] == pytest.approx(42.28)
        assert b0["person_recall"] == pytest.approx(39.13)

    def test_exp_rows_are_labelled_as_macro_without_person_recall(self) -> None:
        rows = build_macro_rows(_baseline(), _colab())
        exp_a = next(row for row in rows if row["model_id"] == "Exp-A")
        assert exp_a["macro_recall"] == pytest.approx(73.2823)
        assert exp_a["map50"] == pytest.approx(78.124)
        assert exp_a["person_recall"] is None
        assert "metrics/recall(B)" in exp_a["source"]


class TestRender:
    """The rendered document must carry the warning and both tables."""

    def test_warning_and_sections_present(self) -> None:
        markdown = render_markdown(
            [
                _benchmark("B0_ZeroShot", 20.51),
                _benchmark("Exp_A_Standard", 66.67),
            ],
            _baseline(),
            _colab(),
        )
        assert "macro recall" in markdown.lower()
        assert "Tabel utama" in markdown
        assert "Peringatan" in markdown
        assert "66.67%" in markdown
        assert "39.13%" in markdown
