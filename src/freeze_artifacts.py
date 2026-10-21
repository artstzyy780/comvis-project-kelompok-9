"""Freeze the academic artefacts and detect any later drift (CP06, CP13).

The rubric asks for a *beku* (frozen) proposal and a *tabel hasil beku*: the
version that was defended must be identifiable afterwards, and any later edit
must be visible. This module therefore records a SHA-256 fingerprint of every
deliverable together with the git commit that produced it.

Two modes are provided:

- ``python src/freeze_artifacts.py`` writes the manifest and the Markdown
  report into ``results/freeze/``.
- ``python src/freeze_artifacts.py --check`` recomputes the fingerprints and
  reports which artefacts changed since the freeze.

Usage:
    python src/freeze_artifacts.py
    python src/freeze_artifacts.py --check
"""

import argparse
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import PROJECT_ROOT, RESULTS_DIR


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("FreezeArtifacts")

FREEZE_DIR: Path = RESULTS_DIR / "freeze"
MANIFEST_PATH: Path = FREEZE_DIR / "artifact_manifest.json"
REPORT_PATH: Path = FREEZE_DIR / "frozen_artifacts.md"

# Artefacts that must exist before the project can be defended, with the
# checkpoint they satisfy.
REQUIRED_ARTIFACTS: Tuple[Tuple[str, str], ...] = (
    ("K09_DeteksiPedestrianLowLight_Proposal_v02.pdf", "CP06 - Proposal UTS (versi beku)"),
    ("docs/proposal/K09_Sidang_Proposal_DeteksiPedestrianLowLight.pptx", "CP06 - Slide Sidang UTS"),
    ("docs/proposal/Naskah_Presentasi_Sidang_Proposal_K09.md", "CP06 - Naskah presentasi"),
    ("results/audit/audit_report.md", "CP03 - Laporan audit dataset"),
    ("results/audit/dedup_split_report.md", "CP07 - Split bebas-leakage (scene level)"),
    ("results/experiments/unified_results.md", "CP10 - Tabel hasil terpadu (satu split)"),
    ("results/experiments/benchmark_latency_report.md", "CP12 - Benchmark latensi & FPS"),
    ("README.md", "CP13 - README final"),
)


@dataclass
class ArtifactRecord:
    """Fingerprint of one frozen artefact.

    Attributes:
        relative_path: Path relative to the project root.
        purpose: Checkpoint or deliverable the artefact satisfies.
        exists: Whether the file is present.
        size_bytes: File size in bytes.
        sha256: SHA-256 digest of the file content.
    """

    relative_path: str
    purpose: str
    exists: bool
    size_bytes: int
    sha256: str


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hexadecimal digest, or an empty string when the file is missing.
    """
    if not path.exists() or not path.is_file():
        return ""
    digest: "hashlib._Hash" = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    """Return the short hash of the current git commit.

    Returns:
        The commit hash, or an empty string when git is unavailable.
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


def collect_artifacts(
    artifacts: Optional[Tuple[Tuple[str, str], ...]] = None,
) -> List[ArtifactRecord]:
    """Fingerprint every artefact of the freeze list.

    Args:
        artifacts: Optional override of the default artefact list.

    Returns:
        List of ArtifactRecord entries.
    """
    records: List[ArtifactRecord] = []
    for relative_path, purpose in artifacts or REQUIRED_ARTIFACTS:
        absolute: Path = PROJECT_ROOT / relative_path
        records.append(
            ArtifactRecord(
                relative_path=relative_path,
                purpose=purpose,
                exists=absolute.exists(),
                size_bytes=absolute.stat().st_size if absolute.exists() else 0,
                sha256=sha256_of(absolute),
            )
        )
    return records


def write_manifest(records: List[ArtifactRecord]) -> Path:
    """Write the freeze manifest to disk.

    Args:
        records: Fingerprints of the artefacts.

    Returns:
        Path to the written manifest.
    """
    payload: Dict[str, object] = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(),
        "artifact_count": len(records),
        "artifacts": [record.__dict__ for record in records],
    }
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return MANIFEST_PATH


def write_report(records: List[ArtifactRecord], frozen_at: str) -> Path:
    """Write the human readable freeze report.

    Args:
        records: Fingerprints of the artefacts.
        frozen_at: ISO timestamp of the freeze.

    Returns:
        Path to the written report.
    """
    missing: List[ArtifactRecord] = [record for record in records if not record.exists]
    lines: List[str] = [
        "# 🔒 Daftar Artefak Beku (Frozen Deliverables)",
        "",
        f"- **Waktu pembekuan:** {frozen_at}",
        f"- **Commit Git:** `{current_commit() or 'tidak tersedia'}`",
        f"- **Jumlah artefak:** {len(records)}",
        "",
        "| Artefak | Pemenuhan Checkpoint | Ukuran | SHA-256 (12 char) | Status |",
        "|:---|:---|:---:|:---:|:---:|",
    ]
    for record in records:
        status: str = "✅ ada" if record.exists else "❌ belum ada"
        digest: str = record.sha256[:12] if record.sha256 else "-"
        size: str = f"{record.size_bytes / 1024:.1f} KB" if record.exists else "-"
        lines.append(
            f"| `{record.relative_path}` | {record.purpose} | {size} | `{digest}` | {status} |"
        )

    lines.extend([
        "",
        "## Verifikasi Integritas",
        "",
        "Jalankan `python src/freeze_artifacts.py --check` untuk memastikan artefak tidak berubah "
        "setelah pembekuan. Perbedaan digest menandakan berkas diedit setelah freeze.",
        "",
    ])
    if missing:
        lines.extend([
            f"> **Peringatan:** {len(missing)} artefak belum tersedia:",
            "",
        ])
        for record in missing:
            lines.append(f"> - `{record.relative_path}`")
        lines.append("")

    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return REPORT_PATH


def check_drift(records: List[ArtifactRecord]) -> List[Tuple[str, str, str]]:
    """Compare the current fingerprints with the stored manifest.

    Args:
        records: Fingerprints recomputed from the working tree.

    Returns:
        List of (relative_path, status, detail) tuples for every deviation.
    """
    if not MANIFEST_PATH.exists():
        return [("<manifest>", "tidak ada", "Belum pernah dilakukan freeze.")]

    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        manifest: Dict[str, object] = json.load(handle)

    stored: Dict[str, str] = {
        entry["relative_path"]: entry["sha256"]
        for entry in manifest.get("artifacts", [])  # type: ignore[union-attr]
    }

    deviations: List[Tuple[str, str, str]] = []
    for record in records:
        previous: Optional[str] = stored.get(record.relative_path)
        if previous is None:
            deviations.append((record.relative_path, "baru", "Artefak baru setelah freeze."))
        elif previous != record.sha256:
            deviations.append(
                (record.relative_path, "berubah", f"{previous[:12]} -> {record.sha256[:12]}")
            )
    for relative_path in stored:
        if relative_path not in {record.relative_path for record in records}:
            deviations.append((relative_path, "hilang", "Artefak tidak ada lagi."))
    return deviations


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Bekukan artefak akademik dan deteksi perubahan setelahnya."
    )
    parser.add_argument("--check", action="store_true", help="Periksa perubahan sejak freeze")
    args = parser.parse_args()

    records: List[ArtifactRecord] = collect_artifacts()

    if args.check:
        deviations: List[Tuple[str, str, str]] = check_drift(records)
        if not deviations:
            logger.info("Tidak ada perubahan: seluruh artefak konsisten dengan manifest.")
            return
        logger.warning("Ditemukan %d artefak yang berubah sejak freeze:", len(deviations))
        for relative_path, status, detail in deviations:
            logger.warning("  [%s] %s (%s)", status, relative_path, detail)
        return

    frozen_at: str = datetime.now(timezone.utc).isoformat()
    manifest_path: Path = write_manifest(records)
    report_path: Path = write_report(records, frozen_at)
    missing: int = sum(1 for record in records if not record.exists)

    logger.info("Manifest : %s", manifest_path)
    logger.info("Laporan  : %s", report_path)
    if missing:
        logger.warning("%d artefak belum tersedia (lihat tabel pada laporan).", missing)


if __name__ == "__main__":
    main()
