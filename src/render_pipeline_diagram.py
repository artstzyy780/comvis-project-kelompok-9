"""Render diagram pipeline deteksi dan kebutuhan komputasi (Checkpoint CP05).

Modul ini menghasilkan tiga luaran untuk Sidang Proposal:

1. ``results/proposal/pipeline_diagram.png``  : diagram pipeline horizontal
   (matplotlib murni, tanpa paket diagram eksternal).
2. ``results/proposal/computation_requirements.md`` : estimasi biaya pelatihan
   (Google Colab T4), biaya inferensi pada CPU laptop, dan tabel kontrol
   perbandingan antar eksperimen.
3. ``results/proposal/experiment_design.md`` : desain eksperimen terkontrol
   B0 / Exp-A / Exp-B / Exp-C beserta variabel bebas tunggal tiap eksperimen.

Diagram dibuat hanya dengan ``matplotlib.patches`` (FancyBboxPatch dan
FancyArrowPatch) agar diagram dapat direproduksi pada lingkungan WITHOUT
dependensi torch atau ultralytics.

Contoh pemakaian:
    python src/render_pipeline_diagram.py
"""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Final, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("RenderPipelineDiagram")


# ----------------------------------------------------------------------------
# Konstanta Paths & Parameter Visual
# ----------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
PROPOSAL_DIR: Final[Path] = PROJECT_ROOT / "results" / "proposal"
DIAGRAM_PNG_PATH: Final[Path] = PROPOSAL_DIR / "pipeline_diagram.png"
COMPUTATION_MD_PATH: Final[Path] = PROPOSAL_DIR / "computation_requirements.md"
EXPERIMENT_MD_PATH: Final[Path] = PROPOSAL_DIR / "experiment_design.md"

FIGURE_SIZE: Final[Tuple[float, float]] = (18.0, 5.0)
FIGURE_DPI: Final[int] = 200
AXIS_X_LIMIT: Final[float] = 100.0
AXIS_Y_LIMIT: Final[float] = 24.0

BOX_WIDTH: Final[float] = 13.5
BOX_HEIGHT: Final[float] = 13.0
BOX_START_X: Final[float] = 1.25
BOX_GAP: Final[float] = 7.5
BOX_BOTTOM_Y: Final[float] = 6.5

TITLE_FONTSIZE: Final[int] = 14
BODY_FONTSIZE: Final[int] = 10
ARROW_LABEL_FONTSIZE: Final[int] = 10
NOTE_FONTSIZE: Final[int] = 10

TEXT_COLOR: Final[str] = "#1f2937"
ARROW_COLOR: Final[str] = "#334155"
BACKGROUND_COLOR: Final[str] = "#f8fafc"

# ----------------------------------------------------------------------------
# Konstanta Teknis Proyek (sumber: Master_Reference_Proyek_CV.md & proposal v02)
# ----------------------------------------------------------------------------
COURSE_NAME: Final[str] = "Visi Komputer (IF0PP006) - Semester Ganjil 2026/2027"
GROUP_NAME: Final[str] = (
    "Kelompok 9 (Krisantus Amka Ginting, Linardi Halim, "
    "Kristofer Ryan Giggs Eka Saputra, Leonard Franscois)"
)
DOCUMENT_DATE: Final[str] = "20 Oktober 2026"

TRAIN_IMAGES: Final[int] = 2100
TRAIN_BATCH: Final[int] = 16
TRAIN_EPOCHS: Final[int] = 50
TRAIN_IMGSZ: Final[int] = 640
TRAIN_SEED: Final[int] = 42
ITERATIONS_PER_EPOCH: Final[int] = 132
SECONDS_PER_ITERATION_T4: Final[float] = 0.90
WARMUP_NOTE: Final[int] = 5
VALIDATION_SECONDS_PER_EPOCH: Final[float] = 12.0
PRETRAINING_SECONDS_PER_EXPERIMENT: Final[float] = 900.0

COLAB_GPU_NAME: Final[str] = "NVIDIA Tesla T4 (Google Colab)"
COLAB_VRAM_GB: Final[int] = 16
EXPECTED_VRAM_GB: Final[str] = "4-6"
EXPECTED_DATASET_SIZE_GB: Final[str] = "1,2"

INFERENCE_LATENCY_MIN_MS: Final[float] = 52.0
INFERENCE_LATENCY_MAX_MS: Final[float] = 55.0
INFERENCE_FPS_MIN: Final[float] = 18.0
INFERENCE_FPS_MAX: Final[float] = 19.0
REALTIME_FPS_TARGET: Final[float] = 20.0
HOST_CPU: Final[str] = "AMD Ryzen 6-core / 12-thread @ 2,3 GHz"
HOST_RAM_GB: Final[str] = "15,4"
HOST_OS: Final[str] = "Windows 11"


# ----------------------------------------------------------------------------
# Struktur Data Tahap Pipeline
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineStage:
    """Satu tahap pada diagram pipeline.

    Attributes:
        title: Judul tahap (Bahasa Indonesia).
        body: Baris penjelasan di dalam kotak.
        facecolor: Warna isi kotak.
        edgecolor: Warna garis tepi kotak.
        note: Catatan kecil di bawah kotak (opsional).
    """

    title: str
    body: str
    facecolor: str
    edgecolor: str
    note: str = ""


def build_pipeline_stages() -> List[PipelineStage]:
    """Menyusun definisi lima tahap pipeline deteksi.

    Returns:
        Daftar PipelineStage sesuai urutan alur data.
    """
    return [
        PipelineStage(
            title="Input",
            body=(
                "Citra malam 640x640\n"
                "(JPEG / PNG / frame video)\n"
                "Roboflow Low-light v2"
            ),
            facecolor="#e0f2fe",
            edgecolor="#0284c7",
        ),
        PipelineStage(
            title="Prapemrosesan",
            body=(
                "(opsional)\n"
                "CLAHE kanal-L (CIE L*a*b*)\n"
                "atau Gamma γ=0,6\n"
                "HANYA pada data latih"
            ),
            facecolor="#fef3c7",
            edgecolor="#d97706",
            note="anti data leakage",
        ),
        PipelineStage(
            title="Model YOLOv8n",
            body=(
                "3,2 juta parameter\n"
                "Head anchor-free\n"
                "Loss: CIoU + DFL + BCE"
            ),
            facecolor="#dcfce7",
            edgecolor="#16a34a",
        ),
        PipelineStage(
            title="Post-processing",
            body=(
                "NMS (Non-Max Suppression)\n"
                "conf 0,25 | IoU 0,50\n"
                "Filter ambang deteksi"
            ),
            facecolor="#ede9fe",
            edgecolor="#7c3aed",
        ),
        PipelineStage(
            title="Output",
            body=(
                "Bounding box + label\n"
                "+ confidence score\n"
                "+ FPS (target >= 20)"
            ),
            facecolor="#fee2e2",
            edgecolor="#dc2626",
        ),
    ]


# ----------------------------------------------------------------------------
# Render Diagram
# ----------------------------------------------------------------------------
def _draw_stage_box(
    axes: Axes,
    stage: PipelineStage,
    x_position: float,
) -> None:
    """Menggambar satu kotak tahap beserta judul dan isinya.

    Args:
        axes: Objek axes matplotlib tempat digambar.
        stage: Definisi tahap yang akan digambar.
        x_position: Posisi horizontal kiri kotak.
    """
    box = FancyBboxPatch(
        (x_position, BOX_BOTTOM_Y),
        BOX_WIDTH,
        BOX_HEIGHT,
        boxstyle="round,pad=0.7,rounding_size=1.4",
        linewidth=2.2,
        edgecolor=stage.edgecolor,
        facecolor=stage.facecolor,
        zorder=2,
    )
    axes.add_patch(box)

    axes.text(
        x_position + BOX_WIDTH / 2.0,
        BOX_BOTTOM_Y + BOX_HEIGHT - 1.6,
        stage.title,
        ha="center",
        va="top",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        color=stage.edgecolor,
        zorder=3,
    )
    axes.text(
        x_position + BOX_WIDTH / 2.0,
        BOX_BOTTOM_Y + BOX_HEIGHT - 5.0,
        stage.body,
        ha="center",
        va="top",
        fontsize=BODY_FONTSIZE,
        linespacing=1.6,
        color=TEXT_COLOR,
        zorder=3,
    )

    if stage.note:
        axes.text(
            x_position + BOX_WIDTH / 2.0,
            BOX_BOTTOM_Y - 1.1,
            stage.note,
            ha="center",
            va="top",
            fontsize=NOTE_FONTSIZE,
            style="italic",
            color=stage.edgecolor,
            zorder=3,
        )


def _draw_arrow(
    axes: Axes,
    x_start: float,
    x_end: float,
    label: str,
) -> None:
    """Menggambar panah antar tahap beserta label singkatnya.

    Args:
        axes: Objek axes matplotlib tempat digambar.
        x_start: Koordinat horizontal awal panah.
        x_end: Koordinat horizontal akhir panah.
        label: Teks label pada panah.
    """
    y_center: float = BOX_BOTTOM_Y + BOX_HEIGHT / 2.0
    arrow = FancyArrowPatch(
        (x_start, y_center),
        (x_end, y_center),
        arrowstyle="-|>",
        mutation_scale=24,
        linewidth=2.4,
        color=ARROW_COLOR,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    axes.add_patch(arrow)

    axes.text(
        (x_start + x_end) / 2.0,
        y_center + 1.1,
        label,
        ha="center",
        va="bottom",
        fontsize=ARROW_LABEL_FONTSIZE,
        style="italic",
        color=ARROW_COLOR,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=BACKGROUND_COLOR, edgecolor="none"),
        zorder=3,
    )


def render_pipeline_diagram(output_path: Path) -> None:
    """Menggambar dan menyimpan diagram pipeline deteksi.

    Args:
        output_path: Lokasi berkas PNG keluaran.
    """
    stages: List[PipelineStage] = build_pipeline_stages()
    arrow_labels: Tuple[str, ...] = ("frame", "pencerahan", "inferensi", "NMS", "render")

    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axes.set_facecolor(BACKGROUND_COLOR)

    for index, stage in enumerate(stages):
        x_position: float = BOX_START_X + index * (BOX_WIDTH + BOX_GAP)
        _draw_stage_box(axes, stage, x_position)

    for index in range(len(stages) - 1):
        x_start: float = BOX_START_X + index * (BOX_WIDTH + BOX_GAP) + BOX_WIDTH
        x_end: float = BOX_START_X + (index + 1) * (BOX_WIDTH + BOX_GAP)
        _draw_arrow(axes, x_start, x_end, arrow_labels[index])

    axes.set_xlim(0, AXIS_X_LIMIT)
    axes.set_ylim(0, AXIS_Y_LIMIT)
    axes.axis("off")

    figure.suptitle(
        "Pipeline Deteksi Pejalan Kaki pada Kondisi Minim Cahaya (YOLOv8n)",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.99,
    )
    axes.text(
        AXIS_X_LIMIT / 2.0,
        1.4,
        (
            "Catatan: prapemrosesan pencahayaan hanya diterapkan pada split latih "
            "(anti data leakage); split validasi dan uji tetap memakai citra malam asli."
        ),
        ha="center",
        va="center",
        fontsize=NOTE_FONTSIZE,
        color=ARROW_COLOR,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    logger.info("Diagram pipeline tersimpan di: %s", output_path)


# ----------------------------------------------------------------------------
# Render Markdown Kebutuhan Komputasi
# ----------------------------------------------------------------------------
def format_thousands(value: int) -> str:
    """Memformat bilangan bulat dengan pemisah ribuan gaya Indonesia.

    Args:
        value: Bilangan bulat yang akan diformat.

    Returns:
        String berformat, misal "2.100".
    """
    return f"{value:,}".replace(",", ".")


def format_decimal(value: float, decimals: int = 1) -> str:
    """Memformat angka desimal dengan pemisah koma gaya Indonesia.

    Args:
        value: Angka yang akan diformat.
        decimals: Jumlah angka di belakang koma.

    Returns:
        String berformat, misal "2,1".
    """
    return f"{value:.{decimals}f}".replace(".", ",")


def compute_training_estimates() -> List[Tuple[str, float, float]]:
    """Menghitung perkiraan durasi pelatihan tiap eksperimen.

    Returns:
        Daftar tuple (nama eksperimen, durasi epoch dalam detik, durasi total
        dalam detik) untuk ketiga eksperimen fine-tuning.
    """
    seconds_per_epoch: float = ITERATIONS_PER_EPOCH * SECONDS_PER_ITERATION_T4
    training_seconds: float = seconds_per_epoch * TRAIN_EPOCHS
    total_seconds: float = (
        training_seconds
        + VALIDATION_SECONDS_PER_EPOCH * TRAIN_EPOCHS
        + PRETRAINING_SECONDS_PER_EXPERIMENT
    )

    estimates: List[Tuple[str, float, float]] = []
    for experiment_name in ("Exp-A", "Exp-B", "Exp-C"):
        estimates.append((experiment_name, seconds_per_epoch, total_seconds))
    return estimates


def render_computation_requirements(output_path: Path) -> None:
    """Menyusun dokumen estimasi kebutuhan komputasi proyek.

    Args:
        output_path: Lokasi berkas Markdown keluaran.
    """
    estimates: List[Tuple[str, float, float]] = compute_training_estimates()
    epoch_minutes: float = estimates[0][1] / 60.0
    experiment_hours: float = estimates[0][2] / 3600.0
    total_hours: float = experiment_hours * len(estimates)

    lines: List[str] = [
        "# Kebutuhan Komputasi Proyek - Deteksi Pejalan Kaki Minim Cahaya",
        "",
        "> **Checkpoint:** CP05 - Diagram pipeline dan kebutuhan komputasi (7 XP)  ",
        f"> **Mata Kuliah:** {COURSE_NAME}  ",
        f"> **Kelompok:** {GROUP_NAME}  ",
        f"> **Tanggal:** {DOCUMENT_DATE}  ",
        "> **Referensi angka:** `Master_Reference_Proyek_CV.md`, "
        "`results/experiments/benchmark_latency_report.md`, dan "
        "`results/experiments/colab_training_benchmark_report.md`.",
        "",
        "---",
        "",
        "## 1. Ringkasan Lingkungan Eksekusi",
        "",
        "| Komponen | Spesifikasi |",
        "|---|---|",
        f"| Platform pelatihan | {COLAB_GPU_NAME}, {COLAB_VRAM_GB} GB VRAM (gratis) |",
        f"| Platform inferensi | CPU laptop lokal ({HOST_CPU}), RAM {HOST_RAM_GB} GB |",
        f"| Sistem operasi host | {HOST_OS} |",
        f"| Resolusi kerja | {TRAIN_IMGSZ}x{TRAIN_IMGSZ} piksel |",
        "| Bobot model | YOLOv8n, sekitar 3,2 juta parameter (sekitar 6,5 MB) |",
        "",
        "## 2. Estimasi Biaya Pelatihan (Training Cost)",
        "",
        "Perhitungan berdasarkan konfigurasi terkunci pada `configs/experiments.yaml`: "
        f"{format_thousands(TRAIN_IMAGES)} citra latih, batch {TRAIN_BATCH}, imgsz "
        f"{TRAIN_IMGSZ}, "
        f"{TRAIN_EPOCHS} epoch, seed {TRAIN_SEED}. Dengan "
        f"{ITERATIONS_PER_EPOCH} iterasi per epoch dan sekitar "
        f"{format_decimal(SECONDS_PER_ITERATION_T4, 2)} detik per iterasi pada GPU T4.",
        "",
        "| Eksperimen | Jumlah Citra Latih | Iterasi/Epoch | Durasi per Epoch | Total Durasi (50 epoch) |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]

    for experiment_name, seconds_epoch, seconds_total in estimates:
        lines.append(
            f"| **{experiment_name}** | {format_thousands(TRAIN_IMAGES)} | "
            f"{ITERATIONS_PER_EPOCH} | "
            f"{format_decimal(seconds_epoch / 60.0)} menit | "
            f"sekitar {format_decimal(seconds_total / 3600.0)} jam |"
        )

    lines.extend([
        f"| **Total 3 eksperimen** | {format_thousands(TRAIN_IMAGES)} x 3 | "
        f"{ITERATIONS_PER_EPOCH} x 3 | {format_decimal(epoch_minutes * 3)} menit | "
        f"**sekitar {format_decimal(total_hours)} jam** |",
        "",
        "### Catatan Tambahan Biaya Pelatihan",
        "",
        "- **Anggaran waktu aman:** Dengan rentang 2-4 jam per eksperimen seperti yang "
        "dokumentasikan pada Master Reference, total biaya GPU berada pada rentang "
        "6-12 jam untuk ketiga eksperimen (estimasi terhitung di atas adalah durasi "
        "pelatihan murni tanpa jeda sesi Colab gratis).",
        f"- **Kebutuhan VRAM:** sekitar {EXPECTED_VRAM_GB} GB, masih jauh di bawah "
        f"kapasitas {COLAB_VRAM_GB} GB milik T4, sehingga batch size {TRAIN_BATCH} "
        "aman tanpa penyesuaian.",
        f"- **Kebutuhan penyimpanan:** sekitar {EXPECTED_DATASET_SIZE_GB} GB per varian "
        "dataset (mentah, CLAHE, dan Gamma), sehingga total sekitar 3,6 GB di Google Drive.",
        "- **Ketidakdeterminan GPU:** performa T4 di Colab gratis dapat menurun bila "
        "sesi terputus, sehingga hasil terbaik disimpan dari checkpoint validasi, "
        "bukan dari epoch terakhir.",
        "",
        "## 3. Estimasi Biaya Inferensi (Inference Cost)",
        "",
        f"Diukur pada host lokal dengan {WARMUP_NOTE} iterasi warm-up sebelum "
        "pengukuran, batch size 1, dan resolusi 640x640 (sumber: "
        "`results/experiments/benchmark_latency_report.md`).",
        "",
        "| Parameter | Nilai Terukur |",
        "|:---|:---|",
        f"| Prosesor | {HOST_CPU} |",
        f"| Memori RAM | {HOST_RAM_GB} GB |",
        "| Perangkat inferensi | CPU (tanpa GPU) |",
        "| Resolusi input | 640x640 piksel, batch size 1 |",
        "| Model | YOLOv8n (3,2 juta parameter) |",
        f"| Latensi rata-rata | {INFERENCE_LATENCY_MIN_MS:.0f}-{INFERENCE_LATENCY_MAX_MS:.0f} ms per citra |",
        f"| Throughput | {INFERENCE_FPS_MIN:.0f}-{INFERENCE_FPS_MAX:.0f} FPS |",
        f"| Ambang target proyek | {REALTIME_FPS_TARGET:.0f} FPS |",
        "| Status ambang target | **BORDERLINE** - sekitar 1-2 FPS di bawah ambang |",
        "",
        "### Rincian Latensi per Tahap (ms)",
        "",
        "| Tahap | Rentang Terukur |",
        "|:---|:---|",
        "| Prapemrosesan (resize + normalisasi) | 1,9 - 2,7 ms |",
        "| Model forward (inferensi YOLOv8n) | 47,9 - 51,0 ms |",
        "| Post-processing (NMS) | 0,8 - 1,0 ms |",
        f"| **Total** | **{INFERENCE_LATENCY_MIN_MS:.0f}-{INFERENCE_LATENCY_MAX_MS:.0f} ms** |",
        "",
        "### Mitigasi terhadap ambang 20 FPS",
        "",
        "1. **Ekspor ONNX dengan kuantisasi INT8** yang dapat menurunkan bobot model "
        "sekitar 4 kali dan mempercepat inferensi pada CPU.",
        "2. **Multithreading OpenCV** untuk memisahkan pembacaan frame, prapemrosesan, "
        "dan inferensi ke thread berbeda.",
        "3. **Penggunaan TFLite atau ONNX Runtime** sebagai runtime inferensi pengganti "
        "PyTorch pada skrip demo.",
        "4. **Pernyataan ambang secara eksplisit** pada proposal agar klaim real-time "
        "dapat diverifikasi dosen dengan angka yang sama.",
        "",
        "## 4. Kontrol Perbandingan Antar Eksperimen",
        "",
        "Agar kesimpulan ilmiah valid, seluruh parameter berikut dikunci (tidak "
        "berubah) dan hanya perlakuan data latih yang menjadi variabel bebas.",
        "",
        "| Parameter | B0 | Exp-A | Exp-B | Exp-C | Status Kontrol |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
        "| Arsitektur model | YOLOv8n | YOLOv8n | YOLOv8n | YOLOv8n | **Dikunci** |",
        "| Bobot awal | yolov8n.pt (COCO) | yolov8n.pt (COCO) | yolov8n.pt (COCO) | "
        "yolov8n.pt (COCO) | **Dikunci** |",
        f"| Epoch | 0 (tanpa pelatihan) | {TRAIN_EPOCHS} | {TRAIN_EPOCHS} | "
        f"{TRAIN_EPOCHS} | **Dikunci** (Exp-A/B/C) |",
        f"| Batch size | - | {TRAIN_BATCH} | {TRAIN_BATCH} | {TRAIN_BATCH} | **Dikunci** |",
        f"| Ukuran citra (imgsz) | {TRAIN_IMGSZ} | {TRAIN_IMGSZ} | {TRAIN_IMGSZ} | "
        f"{TRAIN_IMGSZ} | **Dikunci** |",
        f"| Random seed | {TRAIN_SEED} | {TRAIN_SEED} | {TRAIN_SEED} | {TRAIN_SEED} | "
        "**Dikunci** |",
        "| Optimizer / lr0 | - | auto / 0,01 | auto / 0,01 | auto / 0,01 | **Dikunci** |",
        "| Split validasi | valid (200) | valid (200) | valid (200) | valid (200) | **Dikunci** |",
        "| Split uji | test (100) | test (100) | test (100) | test (100) | **Dikunci** |",
        "| Split latih | - | 2.100 | 2.100 | 2.100 | **Dikunci** |",
        "| Augmentasi Roboflow | flip, crop, brightness | flip, crop, brightness | "
        "flip, crop, brightness | flip, crop, brightness | **Dikunci** |",
        "| **Perlakuan data latih** | **tanpa pelatihan (zero-shot)** | **citra malam asli** | "
        "**citra + CLAHE kanal-L** | **citra + Gamma 0,6** | "
        "**VARIABEL BEBAS SATU-SATUNYA** |",
        "",
        "Aturan evaluasi yang berlaku: pembagian split data dilakukan **sebelum** augmentasi, "
        "split uji hanya dievaluasi satu kali pada tahap akhir, dan metrik utama "
        "adalah Recall pejalan kaki pada IoU 0,50.",
        "",
        "## 5. Reproduksi",
        "",
        "```bash",
        "python src/render_pipeline_diagram.py",
        "```",
        "",
        "---",
        "",
        "*(Dokumen ini dihasilkan secara otomatis oleh "
        "`src/render_pipeline_diagram.py` untuk pemenuhan Checkpoint CP05.)*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    logger.info("Dokumen kebutuhan komputasi tersimpan di: %s", output_path)


def render_experiment_design(output_path: Path) -> None:
    """Menyusun dokumen desain eksperimen terkontrol.

    Args:
        output_path: Lokasi berkas Markdown keluaran.
    """
    lines: List[str] = [
        "# Desain Eksperimen Terkontrol - B0, Exp-A, Exp-B, Exp-C",
        "",
        "> **Checkpoint:** CP05 - Desain eksperimen dan diagram pipeline (7 XP)  ",
        f"> **Mata Kuliah:** {COURSE_NAME}  ",
        f"> **Kelompok:** {GROUP_NAME}  ",
        f"> **Tanggal:** {DOCUMENT_DATE}  ",
        "> **Sumber konfigurasi:** `configs/experiments.yaml` (sumber kebenaran tunggal "
        "untuk registry eksperimen).",
        "",
        "---",
        "",
        "## 1. Tabel Desain Eksperimen",
        "",
        "| ID | Model | Perlakuan Data Latih | Split Evaluasi | Fokus Metrik | "
        "Hipotesis yang Diharapkan | Variabel Bebas Tunggal |",
        "|:---:|---|---|:---:|---|---|---|",
        "| **B0** | YOLOv8n (COCO pretrained) | Tanpa pelatihan (zero-shot); citra malam "
        "asli apa adanya | Test (100 citra) | Baseline Recall & mAP@50 | Detektor "
        "siang hari gagal pada citra malam sehingga Recall pejalan kaki sangat rendah "
        "(< 50%) | Tidak ada pelatihan sama sekali (acuan pembanding) |",
        "| **Exp-A** | YOLOv8n fine-tuned | Fine-tuning pada 2.100 citra malam asli "
        "(tanpa pencerahan) | Test (100 citra) | Recall domain fine-tuning | "
        "Fine-tuning domain menaikkan Recall pejalan kaki secara signifikan dibanding "
        "B0 | Fine-tuning tanpa augmentasi pencahayaan (kontrol) |",
        "| **Exp-B** | YOLOv8n fine-tuned | Fine-tuning pada 2.100 citra yang diperkuat "
        "CLAHE kanal-L (clip 2,0; tile 8x8) pada ruang CIE L*a*b* | Test (100 citra) | "
        "Recall pejalan kaki target (+10%) & mAP@50 | CLAHE kontras lokal meninggikan "
        "siluet pejalan kaki sehingga Recall naik tanpa menurunkan Precision secara "
        "besar | Pra-pemrosesan CLAHE pada data latih saja |",
        "| **Exp-C** | YOLOv8n fine-tuned | Fine-tuning pada 2.100 citra yang dikoreksi "
        "Gamma non-linear (gamma = 0,6) | Test (100 citra) | Recall pejalan kaki pada "
        "kecerahan sedang & kontras global | Gamma correction global dapat mengangkat bayangan "
        "gelap tanpa memperbesar derau frekuensi tinggi sehingga Recall "
        "pejalan kaki paling tinggi | Pra-pemrosesan Gamma Correction pada data latih saja |",
        "",
        "## 2. Aturan Eksperimen Terkontrol",
        "",
        "1. **Satu variabel bebas per eksperimen.** Arsitektur, bobot awal, jumlah "
        "epoch, batch size, ukuran citra, dan random seed dikunci; hanya perlakuan data "
        f"latih yang berubah (seed {TRAIN_SEED}, epoch {TRAIN_EPOCHS}, batch "
        f"{TRAIN_BATCH}, imgsz {TRAIN_IMGSZ}).",
        "2. **Split data sebelum augmentasi.** Prapemrosesan CLAHE dan Gamma hanya "
        "diterapkan pada split latih; split validasi dan uji tetap citra malam asli "
        "sehingga tidak terjadi kebocoran data.",
        "3. **Split uji digunakan satu kali.** Evaluasi pada 100 citra test dilakukan "
        "hanya setelah seluruh eksperimen selesai dan seluruh parameter terkunci.",
        "4. **Metrik utama Recall pejalan kaki pada IoU 0,50.** Precision, mAP@50, "
        "mAP@50:95, dan FPS dilaporkan sebagai metrik pendukung.",
        "5. **Warm-up sebelum pengukuran FPS.** Lima iterasi warm-up tidak dihitung "
        "dalam statistik latensi.",
        "",
        "## 3. Kriteria Keberhasilan (Kriteria Sukses)",
        "",
        "| Kriteria | Ambang |",
        "|:---|:---:|",
        "| Peningkatan Recall pejalan kaki dibanding B0 | > +10 poin persentase |",
        "| Penurunan Precision pejalan kaki akibat augmentasi | < 10 poin persentase |",
        "| Throughput inferensi pada CPU laptop | >= 20 FPS |",
        "| Kepatuhan aturan anti data leakage | 100% (audit otomatis pada split) |",
        "",
        "## 4. Reproduksi",
        "",
        "```bash",
        "python src/render_pipeline_diagram.py",
        "```",
        "",
        "---",
        "",
        "*(Dokumen ini dihasilkan secara otomatis oleh "
        "`src/render_pipeline_diagram.py` untuk pemenuhan Checkpoint CP05.)*",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    logger.info("Dokumen desain eksperimen tersimpan di: %s", output_path)


# ----------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------
def main() -> None:
    """Menjalankan render diagram pipeline dan dokumen kebutuhan komputasi."""
    parser = argparse.ArgumentParser(
        description="Render diagram pipeline dan kebutuhan komputasi (CP05)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROPOSAL_DIR,
        help="Direktori keluaran untuk diagram dan dokumen Markdown.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    render_pipeline_diagram(output_dir / DIAGRAM_PNG_PATH.name)
    render_computation_requirements(output_dir / COMPUTATION_MD_PATH.name)
    render_experiment_design(output_dir / EXPERIMENT_MD_PATH.name)

    logger.info("Seluruh luaran CP05 berhasil dibuat di: %s", output_dir)


if __name__ == "__main__":
    main()
