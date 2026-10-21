"""Matriks keputusan topik untuk proyek Visi Komputer (Checkpoint CP01).

Modul ini membangun bukti tertulis "Matriks Keputusan Topik" untuk proyek
Visi Komputer Kelompok 9: Deteksi Pejalan Kaki pada Kondisi Minim Cahaya
Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif.

Data kandidat dan kriteria penilaian diambil dari ``Master_Reference_Proyek_CV.md``
(Tiga Kandidat Topik dan Rubrik Penilaian) sehingga matriks yang dihasilkan
merupakan representasi terukur dari keputusan yang telah diambil, bukan
penilaian ulang subjektif.

Keluaran:
- ``results/proposal/candidate_matrix.md``   : matriks keputusan siap Sidang UTS
- ``results/proposal/candidate_matrix.json`` : struktur terhitung untuk mesin

Contoh pemakaian:
    python src/build_candidate_matrix.py
"""

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("BuildCandidateMatrix")


# ----------------------------------------------------------------------------
# Konstanta Paths & Parameter Perhitungan
# ----------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
PROPOSAL_DIR: Final[Path] = PROJECT_ROOT / "results" / "proposal"
MATRIX_MD_PATH: Final[Path] = PROPOSAL_DIR / "candidate_matrix.md"
MATRIX_JSON_PATH: Final[Path] = PROPOSAL_DIR / "candidate_matrix.json"

MAX_SCORE: Final[int] = 5
MIN_SCORE: Final[int] = 1
WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9
DOCUMENT_DATE: Final[str] = "20 Oktober 2026"
COURSE_NAME: Final[str] = "Visi Komputer (IF0PP006) - Semester Ganjil 2026/2027"
GROUP_NAME: Final[str] = (
    "Kelompok 9 (Krisantus Amka Ginting, Linardi Halim, "
    "Kristofer Ryan Giggs Eka Saputra, Leonard Franscois)"
)
SOURCE_DOCUMENT: Final[str] = "Master_Reference_Proyek_CV.md"


# ----------------------------------------------------------------------------
# Struktur Data
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Criterion:
    """Satu kriteria penilaian beserta bobotnya terhadap keputusan topik.

    Attributes:
        criterion_id: Kode ringkas kriteria (contoh: 'K1').
        name: Nama kriteria dalam Bahasa Indonesia.
        weight: Bobot relatif kriteria; jumlah seluruh bobot harus 1.0.
        description: Penjelasan singkat mengenai apa yang diukur kriteria ini.
    """

    criterion_id: str
    name: str
    weight: float
    description: str


@dataclass(frozen=True)
class Candidate:
    """Profil lengkap satu kandidat topik beserta skor per kriteria.

    Attributes:
        candidate_id: Kode kandidat (C1 sampai C3).
        title: Judul topik kandidat.
        task_type: Jenis tugas computer vision yang ditangani.
        main_problem: Masalah utama yang hendak diselesaikan.
        input_spec: Bentuk masukan data yang digunakan sistem.
        output_spec: Bentuk keluaran yang dihasilkan sistem.
        data_availability: Ketersediaan dataset publik beserta statusnya.
        baseline_availability: Ketersediaan metode baseline yang realistis.
        metric_feasibility: Kelayakan metrik evaluasi standar.
        compute_cost: Estimasi biaya komputasi ('Low', 'Medium', atau 'High').
        novelty_risk: Risiko orisinalitas dan ruang kontribusi topik.
        rationale: Alasan ringkas penilaian terhadap kandidat ini.
        scores: Skor 1-5 tiap kriteria dengan key ``Criterion.criterion_id``.
    """

    candidate_id: str
    title: str
    task_type: str
    main_problem: str
    input_spec: str
    output_spec: str
    data_availability: str
    baseline_availability: str
    metric_feasibility: str
    compute_cost: str
    novelty_risk: str
    rationale: str
    scores: Dict[str, int]


@dataclass(frozen=True)
class RankedCandidate:
    """Hasil perhitungan tertimbang untuk satu kandidat.

    Attributes:
        candidate: Profil kandidat asal.
        weighted_score: Skor tertimbang (rentang 0.0 sampai 5.0).
        percentage: Skor tertimbang dalam bentuk persen dari skor maksimum.
        rank: Peringkat 1 berarti tertinggi.
    """

    candidate: Candidate
    weighted_score: float
    percentage: float
    rank: int


# ----------------------------------------------------------------------------
# Data Kandidat & Kriteria
# ----------------------------------------------------------------------------
def build_criteria() -> List[Criterion]:
    """Membangun daftar kriteria penilaian beserta bobotnya.

    Bobot diturunkan dari rubrik Panduan Proyek Akhir: kelayakan dataset,
    baseline, eksperimen terkontrol, dan evaluasi kuantitatif diberi bobot
    lebih besar daripada orisinalitas murni.

    Returns:
        Daftar kriteria terurut sesuai urutan presentasi pada laporan.
    """
    return [
        Criterion(
            criterion_id="K1",
            name="Kesesuaian tugas CV",
            weight=0.20,
            description=(
                "Seberapa dekat topik dengan inti mata kuliah Computer Vision "
                "(deteksi objek, segmentasi, tracking) dan relevansinya dengan "
                "permasalahan nyata di lapangan."
            ),
        ),
        Criterion(
            criterion_id="K2",
            name="Ketersediaan dataset",
            weight=0.20,
            description=(
                "Ketersediaan dataset publik/legal yang sudah terunduh, "
                "teranotasi, dan siap pakai tanpa pengumpulan data tambahan."
            ),
        ),
        Criterion(
            criterion_id="K3",
            name="Baseline yang realistis",
            weight=0.15,
            description=(
                "Keberadaan metode pembanding sederhana yang wajar sehingga "
                "kontribusi tiap eksperimen dapat dinilai terhadap sebuah acuan."
            ),
        ),
        Criterion(
            criterion_id="K4",
            name="Kesesuaian metrik",
            weight=0.15,
            description=(
                "Ketersediaan metrik evaluasi standar dan tidak ambigu yang "
                "menjadi ukuran keberhasilan eksperimen."
            ),
        ),
        Criterion(
            criterion_id="K5",
            name="Biaya komputasi",
            weight=0.10,
            description=(
                "Kebutuhan GPU/CPU dan waktu pelatihan, serta kecocokannya "
                "dengan ketersediaan GPU gratis dan laptop mahasiswa."
            ),
        ),
        Criterion(
            criterion_id="K6",
            name="Kontribusi orisinal",
            weight=0.10,
            description=(
                "Ruang kebaruan yang dapat ditawarkan dari topik yang umumnya "
                "sudah banyak dikaji oleh peneliti lain."
            ),
        ),
        Criterion(
            criterion_id="K7",
            name="Feasibility dalam 1 semester",
            weight=0.10,
            description=(
                "Kesesuaian ruang lingkup dengan jadwal UTS-UAS satu semester, "
                "termasuk waktu pengujian dan penyusunan laporan."
            ),
        ),
    ]


def build_candidates() -> List[Candidate]:
    """Membangun profil tiga kandidat topik beserta skor per kriteria.

    Returns:
        Daftar kandidat C1 sampai C3 sesuai urutan pada dokumen referensi.
    """
    return [
        Candidate(
            candidate_id="C1",
            title="Deteksi Pejalan Kaki di Malam Hari (Night-time Pedestrian Detection)",
            task_type="Object Detection",
            main_problem=(
                "Model deteksi standar yang dilatih pada data siang hari "
                "mengalami penurunan drastis pada kondisi malam karena kontras "
                "rendah, derau sensor, dan silau lampu kendaraan sehingga "
                "pejalan kaki lolos terdeteksi (false negative fatal)."
            ),
            input_spec=(
                "Foto atau frame video kamera jalanan malam hari "
                "(JPEG/PNG/MP4), resolusi 640x640 piksel"
            ),
            output_spec=(
                "Bounding box pejalan kaki + label 'Pedestrian' + confidence score + FPS"
            ),
            data_availability=(
                "Tersedia: Roboflow Low-light v2 (2.400 citra, 7.777 anotasi, "
                "lisensi CC BY 4.0), sudah diunduh dan diaudit pada src/audit_dataset.py"
            ),
            baseline_availability=(
                "Tersedia: YOLOv8n pretrained COCO sebagai baseline zero-shot (B0) "
                "dan YOLOv8n fine-tuned sebagai pembanding (Exp-A)"
            ),
            metric_feasibility=(
                "Sangat layak: mAP@50, mAP@50:95, Precision, Recall, dan FPS "
                "merupakan metrik standar deteksi objek"
            ),
            compute_cost="Low",
            novelty_risk=(
                "Sedang: topik deteksi malam sudah banyak dikaji, namun "
                "kontribusi spesifik augmentasi pencahayaan adaptif (CLAHE kanal-L "
                "vs Gamma) untuk Recall pejalan kaki masih relatif baru"
            ),
            rationale=(
                "Dataset legal dan siap pakai, baseline melimpah dan metrik "
                "standar, biaya komputasi rendah, serta memiliki dampak "
                "keselamatan yang jelas sehingga mudah dikaitkan dengan ADAS."
            ),
            scores={
                "K1": 5,  # Kesesuaian tugas CV
                "K2": 5,  # Ketersediaan dataset
                "K3": 5,  # Baseline yang realistis
                "K4": 5,  # Kesesuaian metrik
                "K5": 4,  # Biaya komputasi
                "K6": 4,  # Kontribusi orisinal
                "K7": 5,  # Feasibility dalam 1 semester
            },
        ),
        Candidate(
            candidate_id="C2",
            title="Segmentasi Kerusakan Bodi Mobil (Goresan & Penyok)",
            task_type="Semantic Segmentation",
            main_problem=(
                "Penilaian kerusakan mobil untuk klaim asuransi masih "
                "dilakukan manual, subjektif, dan lambat sehingga menimbulkan "
                "ketidaksepakatan antar penilai yang tinggi."
            ),
            input_spec="Foto bodi kendaraan rusak (goresan, penyok, cat terkelupas)",
            output_spec="Mask piksel area kerusakan + persentase luas kerusakan per citra",
            data_availability=(
                "Belum tersedia: tidak ada dataset anotasi mask yang siap pakai; "
                "butuh pengumpulan citra dan pelabelan piksel secara manual"
            ),
            baseline_availability=(
                "Ada secara teori (U-Net/segmentasi semantik), namun harus "
                "dibangun dari nol karena label mask belum tersedia"
            ),
            metric_feasibility=(
                "Cukup: mIoU dan F1 per kelas tersedia, tetapi metrik bisnis "
                "'persentase luas kerusakan' memerlukan validasi ulang"
            ),
            compute_cost="Medium",
            novelty_risk=(
                "Sedang: banyak riset segmentasi kerusakan kendaraan yang sudah "
                "publik, sehingga temuan orisinal sulit dibedakan"
            ),
            rationale=(
                "Nilai bisnisnya tinggi, tetapi biaya pengumpulan dan pelabelan "
                "data berada di luar jangkauan satu semester dan ruang lingkup "
                "menjadi terlalu luas untuk eksperimen terkontrol yang bersih."
            ),
            scores={
                "K1": 4,  # Kesesuaian tugas CV
                "K2": 2,  # Ketersediaan dataset
                "K3": 3,  # Baseline yang realistis
                "K4": 4,  # Kesesuaian metrik
                "K5": 3,  # Biaya komputasi
                "K6": 3,  # Kontribusi orisinal
                "K7": 2,  # Feasibility dalam 1 semester
            },
        ),
        Candidate(
            candidate_id="C3",
            title="Pelacakan Multi-Objek Bawah Air (Multi-Object Tracking saat Oklusi)",
            task_type="Multi-Object Tracking (MOT)",
            main_problem=(
                "Sistem pelacakan sering keliru menukar identitas objek "
                "(ID-Switch) ketika objek saling menutupi di dalam air dengan "
                "visibilitas yang sangat buruk."
            ),
            input_spec="Klip video pendek rekaman bawah air (multi-objek, oklusi tinggi)",
            output_spec="ID unik konsisten per objek + garis trajektori + laporan ID-Switch",
            data_availability=(
                "Belum tersedia: dataset MOT bawah air yang legal dan siap pakai "
                "tidak ada di repositori proyek; perekaman sendiri berisiko lisensi"
            ),
            baseline_availability=(
                "Ada (SORT dan ByteTrack), namun harus digabungkan dengan detektor "
                "sehingga pipeline menjadi dua tahap"
            ),
            metric_feasibility=(
                "Cukup: IDF1, HOTA, dan MOTA tersedia, namun metrik ini kurang "
                "khas untuk argumen keselamatan yang dituju proyek"
            ),
            compute_cost="High",
            novelty_risk=(
                "Tinggi: topik pelacakan bawah air sudah memiliki banyak karya "
                "terbit dengan dataset video sendiri, sehingga pembedaan "
                "kontribusi menjadi sulit"
            ),
            rationale=(
                "Butuh koleksi video, pemrosesan frame yang berat, dan metrik yang "
                "kurang terhubung dengan rumusan masalah keselamatan, sehingga "
                "merupakan pilihan paling berisiko dari sisi kelayakan satu semester."
            ),
            scores={
                "K1": 4,  # Kesesuaian tugas CV
                "K2": 1,  # Ketersediaan dataset
                "K3": 3,  # Baseline yang realistis
                "K4": 4,  # Kesesuaian metrik
                "K5": 2,  # Biaya komputasi
                "K6": 3,  # Kontribusi orisinal
                "K7": 2,  # Feasibility dalam 1 semester
            },
        ),
    ]


# ----------------------------------------------------------------------------
# Validasi & Perhitungan
# ----------------------------------------------------------------------------
def validate_criteria(criteria: List[Criterion]) -> None:
    """Memastikan bobot kriteria valid dan berjumlah tepat 1.0.

    Args:
        criteria: Daftar kriteria yang akan divalidasi.

    Raises:
        ValueError: Jika jumlah bobot tidak sama dengan 1.0 atau suatu bobot
            berada di luar rentang (0, 1].
    """
    if not criteria:
        raise ValueError("Daftar kriteria tidak boleh kosong.")

    total_weight: float = sum(criterion.weight for criterion in criteria)
    if abs(total_weight - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"Jumlah bobot kriteria harus tepat 1.0, terdeteksi {total_weight:.6f}."
        )

    for criterion in criteria:
        if not 0.0 < criterion.weight <= 1.0:
            raise ValueError(
                f"Bobot kriteria '{criterion.criterion_id}' di luar rentang (0, 1]: "
                f"{criterion.weight}."
            )


def validate_scores(candidates: List[Candidate], criteria: List[Criterion]) -> None:
    """Memastikan setiap kandidat memiliki skor valid untuk seluruh kriteria.

    Args:
        candidates: Daftar kandidat yang akan divalidasi.
        criteria: Daftar kriteria acuan.

    Raises:
        ValueError: Jika ada skor yang hilang atau berada di luar rentang 1-5.
    """
    criterion_ids: List[str] = [criterion.criterion_id for criterion in criteria]

    for candidate in candidates:
        missing: List[str] = [
            criterion_id
            for criterion_id in criterion_ids
            if criterion_id not in candidate.scores
        ]
        if missing:
            raise ValueError(
                f"Kandidat '{candidate.candidate_id}' kehilangan skor untuk: {missing}."
            )

        for criterion_id, score in candidate.scores.items():
            if not MIN_SCORE <= score <= MAX_SCORE:
                raise ValueError(
                    f"Skor kandidat '{candidate.candidate_id}' pada '{criterion_id}' "
                    f"harus di rentang {MIN_SCORE}-{MAX_SCORE}, terdeteksi {score}."
                )


def compute_weighted_score(candidate: Candidate, criteria: List[Criterion]) -> float:
    """Menghitung skor tertimbang kandidat sebagai jumlah (skor x bobot).

    Args:
        candidate: Kandidat yang akan dihitung.
        criteria: Daftar kriteria beserta bobotnya.

    Returns:
        Skor tertimbang dengan presisi dua angka desimal.
    """
    weighted: float = sum(
        candidate.scores[criterion.criterion_id] * criterion.weight
        for criterion in criteria
    )
    return round(weighted, 2)


def compute_percentage(weighted_score: float) -> float:
    """Mengubah skor tertimbang menjadi persentase dari skor maksimum.

    Args:
        weighted_score: Skor tertimbang (maksimum 5.0).

    Returns:
        Persentase pembulatan satu angka desimal dari 5.0.
    """
    max_achievable: float = float(MAX_SCORE)
    return round(weighted_score / max_achievable * 100.0, 1)


def rank_candidates(
    candidates: List[Candidate],
    criteria: List[Criterion],
) -> List[RankedCandidate]:
    """Mengurutkan kandidat dari skor tertimbang tertinggi ke terendah.

    Args:
        candidates: Daftar kandidat sumber.
        criteria: Daftar kriteria beserta bobotnya.

    Returns:
        Daftar RankedCandidate terurut menurun dengan peringkat 1 = tertinggi.
    """
    scored: List[RankedCandidate] = []
    for candidate in candidates:
        weighted: float = compute_weighted_score(candidate, criteria)
        scored.append(
            RankedCandidate(
                candidate=candidate,
                weighted_score=weighted,
                percentage=compute_percentage(weighted),
                rank=0,
            )
        )

    # Pengurutan stabil: skor sama dipecah berdasarkan urutan deklarasi kandidat.
    scored.sort(key=lambda item: -item.weighted_score)
    return [
        RankedCandidate(
            candidate=item.candidate,
            weighted_score=item.weighted_score,
            percentage=item.percentage,
            rank=position,
        )
        for position, item in enumerate(scored, start=1)
    ]


# ----------------------------------------------------------------------------
# Render Laporan
# ----------------------------------------------------------------------------
def render_markdown(
    criteria: List[Criterion],
    ranked: List[RankedCandidate],
) -> str:
    """Menyusun isi Markdown matriks keputusan topik.

    Args:
        criteria: Daftar kriteria beserta bobotnya.
        ranked: Hasil ranking kandidat.

    Returns:
        Teks Markdown lengkap untuk berkas ``candidate_matrix.md``.
    """
    total_weight: float = sum(criterion.weight for criterion in criteria)
    winner: RankedCandidate = ranked[0]
    runner_up: RankedCandidate = ranked[1]

    lines: List[str] = [
        "# Matriks Keputusan Topik - Proyek Visi Komputer",
        "",
        "> **Checkpoint:** CP01 - Bukti kelayakan & matriks keputusan topik (4 XP)  ",
        f"> **Mata Kuliah:** {COURSE_NAME}  ",
        f"> **Kelompok:** {GROUP_NAME}  ",
        f"> **Tanggal:** {DOCUMENT_DATE}  ",
        f"> **Sumber data:** `{SOURCE_DOCUMENT}` (bagian Tiga Kandidat Topik dan Rubrik Penilaian)",
        "",
        "---",
        "",
        "## 1. Ringkasan Kandidat Topik",
        "",
        "Tiga kandidat topik yang telah dievaluasi oleh kelompok, beserta status kelulusannya:",
        "",
        "| ID | Kandidat Topik | Tugas CV | Status |",
        "|:---:|---|---|:---:|",
    ]

    for item in ranked:
        status: str = "**TERPILIH**" if item.rank == 1 else "Tidak dipilih"
        lines.append(
            f"| **{item.candidate.candidate_id}** | {item.candidate.title} | "
            f"{item.candidate.task_type} | {status} |"
        )

    lines.extend([
        "",
        "## 2. Profil Rinci Setiap Kandidat",
        "",
        "| Aspek | Kandidat C1 | Kandidat C2 | Kandidat C3 |",
        "|---|---|---|---|",
    ])

    profile_rows: List[Tuple[str, str]] = [
        ("Masalah utama", "main_problem"),
        ("Input", "input_spec"),
        ("Output", "output_spec"),
        ("Ketersediaan dataset", "data_availability"),
        ("Ketersediaan baseline", "baseline_availability"),
        ("Kelayakan metrik", "metric_feasibility"),
        ("Biaya komputasi", "compute_cost"),
        ("Risiko orisinalitas", "novelty_risk"),
        ("Rasional penilaian", "rationale"),
    ]

    for label, attribute in profile_rows:
        values: List[str] = [str(getattr(item.candidate, attribute)) for item in ranked]
        lines.append(f"| **{label}** | {values[0]} | {values[1]} | {values[2]} |")

    lines.extend([
        "",
        "## 3. Kriteria Evaluasi dan Bobot",
        "",
        "Bobot ditentukan dari rubrik Panduan Proyek Akhir: kelayakan dataset, "
        "baseline, eksperimen terkontrol, dan evaluasi kuantitatif diberi bobot "
        "lebih besar daripada orisinalitas murni. Skala skor 1 (sangat lemah) "
        "hingga 5 (sangat kuat).",
        "",
        "| Kode | Kriteria | Bobot | Penjelasan |",
        "|:---:|---|:---:|---|",
    ])

    for criterion in criteria:
        lines.append(
            f"| **{criterion.criterion_id}** | {criterion.name} | "
            f"{criterion.weight:.2f} | {criterion.description} |"
        )

    lines.append(
        f"| | **Total bobot** | **{total_weight:.2f}** | "
        "Jumlah bobot diverifikasi secara otomatis bernilai tepat 1,00 |"
    )

    lines.extend([
        "",
        "## 4. Matriks Skor Per Kandidat (1-5)",
        "",
        "| Kode | Kriteria | Bobot | C1 | C2 | C3 |",
        "|:---:|---|:---:|:---:|:---:|:---:|",
    ])

    for criterion in criteria:
        scores: List[int] = [
            item.candidate.scores[criterion.criterion_id] for item in ranked
        ]
        lines.append(
            f"| **{criterion.criterion_id}** | {criterion.name} | {criterion.weight:.2f} | "
            f"{scores[0]} | {scores[1]} | {scores[2]} |"
        )

    lines.extend([
        "",
        "## 5. Ranking Akhir (Skor Tertimbang)",
        "",
        "Skor tertimbang = jumlah (skor kriteria x bobot kriteria), dengan skor "
        "maksimum yang dapat dicapai sebesar 5,00. Persentase dihitung terhadap "
        "skor maksimum tersebut.",
        "",
        "| Peringkat | Kandidat | Skor Tertimbang (maks 5,00) | Persentase |",
        "|:---:|---|:---:|:---:|",
    ])

    for item in ranked:
        lines.append(
            f"| **{item.rank}** | **{item.candidate.candidate_id}** - {item.candidate.title} | "
            f"**{item.weighted_score:.2f}** | **{item.percentage:.1f}%** |"
        )

    margin: float = round(winner.weighted_score - runner_up.weighted_score, 2)
    margin_percentage: float = round(winner.percentage - runner_up.percentage, 1)
    lines.extend([
        "",
        f"> Selisih skor peringkat 1 terhadap peringkat 2 adalah **{margin:.2f} poin** "
        f"({margin_percentage:.1f} poin persentase).",
        "",
        "## 6. Keputusan",
        "",
        f"**Topik yang dipilih adalah Kandidat C1 - {winner.candidate.title}.**",
        "",
        "Alasan keputusan berdasarkan hasil matriks tertimbang:",
        "",
        f"1. **Skor tertinggi dengan selisih signifikan.** C1 memperoleh skor tertimbang "
        f"**{winner.weighted_score:.2f} / 5,00** atau **{winner.percentage:.1f}%**, "
        f"melebihi kandidat peringkat kedua ({runner_up.candidate.candidate_id}, "
        f"{runner_up.weighted_score:.2f} / 5,00).",
        "2. **Dataset legal dan siap pakai.** Roboflow Low-light v2 (2.400 citra, "
        "7.777 anotasi bounding box, lisensi CC BY 4.0) telah diunduh dan diaudit "
        "sehingga tidak ada waktu tunggu pengumpulan data.",
        "3. **Baseline yang ada dan realistis.** Bobot COCO YOLOv8n (zero-shot) dan "
        "YOLOv8n fine-tuned menyediakan dua titik pembanding yang sah bagi eksperimen terkontrol.",
        "4. **Metrik evaluasi tidak ambigu.** Recall, Precision, mAP@50, mAP@50:95, "
        "dan FPS adalah standar industri deteksi objek, sehingga hasil dapat "
        "diperbandingkan secara terbuka.",
        "5. **Biaya komputasi rendah.** Arsitektur YOLOv8n hanya memiliki sekitar 3,2 juta "
        "parameter sehingga seluruh eksperimen dapat diselesaikan pada satu GPU T4 gratis.",
        "6. **Relevansi keselamatan yang kuat.** Fokus pada Recall pejalan kaki pada "
        "malam hari menghubungkan proyek ini dengan Intelligent Transportation Systems "
        "dan Advance Driver Assistance Systems.",
        "",
        "## 7. Risiko dan Alternatif",
        "",
        "Meskipun hanya ada satu jalur utama, proyek tetap memerlukan jalur cadangan "
        "apabila sewaktu-waktu ada hambatan teknis:",
        "",
        "- **Risiko 1 - Ketidakseimbangan kelas yang ekstrem.** Objek `person` hanya "
        "11,6% dari total anotasi (rasio 7,58 : 1) sehingga nilai mAP dapat menutupi "
        "kualitas deteksi pejalan kaki. **Mitigasi:** Recall per kelas menjadi metrik "
        "utama dan confusion matrix selalu dilampirkan.",
        "- **Risiko 2 - Peningkatan Recall berbanding Precision.** Augmentasi pencahayaan "
        "mencerahkan noise sensor sehingga false positive dapat meningkat. "
        "**Mitigasi:** pelaporan Recall dan Precision berdampingan serta analisis kasus gagal.",
        "- **Risiko 3 - Throughput CPU di bawah ambang real-time.** Latensi terukur "
        "sekitar 52-55 ms (18-19 FPS) masih sedikit di bawah ambang 20 FPS. "
        "**Mitigasi:** ekspor ONNX INT8 dan multithreading OpenCV; ambang FPS "
        "dinyatakan secara eksplisit pada proposal.",
        "- **Risiko 4 - Ketergantungan pada satu partisi data.** **Mitigasi:** sliced "
        "cross-validation pada split validasi 200 citra agar kesimpulan tidak "
        "bergantung pada satu partisi.",
        "- **Alternatif 1 - Kandidat C2 (segmentasi kerusakan kendaraan)** bila "
        "annotasi mask piksel siap dikumpulkan; memerlukan estimasi tambahan 4-6 minggu "
        "untuk pelabelan data.",
        "- **Alternatif 2 - Kandidat C3 (pelacakan bawah air)** bila akses dataset MOT "
        "bawah air legal diperoleh; pipeline menjadi dua tahap (deteksi + tracking) "
        "sehingga biaya komputasi naik ke kategori High.",
        "- **Alternatif 3 - Tetap pada C1 namun memperluas kontribusi** dengan "
        "eksperimen multi-slicing brightness, kuantifikasi INT8, dan uji pada video "
        "berjalan tanpa perlu mengganti topik.",
        "",
        "## 8. Reproduksi",
        "",
        "Matriks ini dihasilkan ulang secara deterministik oleh skrip berikut:",
        "",
        "```bash",
        "python src/build_candidate_matrix.py",
        "```",
        "",
        "Struktur terhitung versi mesin disimpan pada "
        "`results/proposal/candidate_matrix.json`.",
        "",
        "---",
        "",
        "*(Dokumen ini dihasilkan secara otomatis oleh `src/build_candidate_matrix.py` "
        "untuk pemenuhan Checkpoint CP01.)*",
    ])

    return "\n".join(lines) + "\n"


def build_payload(
    criteria: List[Criterion],
    ranked: List[RankedCandidate],
) -> Dict[str, Any]:
    """Menyusun struktur JSON lengkap dari matriks yang telah dihitung.

    Args:
        criteria: Daftar kriteria beserta bobotnya.
        ranked: Hasil ranking kandidat.

    Returns:
        Dictionary JSON-serializable berisi seluruh hasil perhitungan.
    """
    winner: RankedCandidate = ranked[0]
    return {
        "metadata": {
            "checkpoint": "CP01",
            "course": COURSE_NAME,
            "group": GROUP_NAME,
            "document_date": DOCUMENT_DATE,
            "source_document": SOURCE_DOCUMENT,
            "generator": "src/build_candidate_matrix.py",
        },
        "scoring": {
            "min_score": MIN_SCORE,
            "max_score": MAX_SCORE,
            "weights_sum": round(sum(criterion.weight for criterion in criteria), 6),
            "weights_sum_valid": True,
        },
        "criteria": [
            {
                "id": criterion.criterion_id,
                "name": criterion.name,
                "weight": criterion.weight,
                "description": criterion.description,
            }
            for criterion in criteria
        ],
        "candidates": [
            {
                "id": item.candidate.candidate_id,
                "title": item.candidate.title,
                "task_type": item.candidate.task_type,
                "main_problem": item.candidate.main_problem,
                "input": item.candidate.input_spec,
                "output": item.candidate.output_spec,
                "data_availability": item.candidate.data_availability,
                "baseline_availability": item.candidate.baseline_availability,
                "metric_feasibility": item.candidate.metric_feasibility,
                "compute_cost": item.candidate.compute_cost,
                "novelty_risk": item.candidate.novelty_risk,
                "rationale": item.candidate.rationale,
                "scores": dict(item.candidate.scores),
                "weighted_score": item.weighted_score,
                "percentage": item.percentage,
                "rank": item.rank,
            }
            for item in ranked
        ],
        "decision": {
            "selected_candidate_id": winner.candidate.candidate_id,
            "selected_title": winner.candidate.title,
            "weighted_score": winner.weighted_score,
            "percentage": winner.percentage,
            "margin_over_runner_up": round(
                winner.weighted_score - ranked[1].weighted_score, 2
            ),
        },
    }


def print_summary_table(ranked: List[RankedCandidate]) -> None:
    """Menampilkan tabel ringkasan matriks ke standard output.

    Args:
        ranked: Hasil ranking kandidat.
    """
    title_width: int = max(len(item.candidate.title) for item in ranked)
    header: str = (
        f"{'Peringkat':<9} {'Kode':<6} {'Skor Tertimbang':<17} "
        f"{'Persentase':<12} Judul"
    )
    print("\n" + "=" * (len(header) + title_width))
    print("RINGKASAN MATRIKS KEPUTUSAN TOPIK (Kelompok 9)")
    print("=" * (len(header) + title_width))
    print(header)
    print("-" * (len(header) + title_width))
    for item in ranked:
        print(
            f"{item.rank:<9} {item.candidate.candidate_id:<6} "
            f"{item.weighted_score:<17.2f} {item.percentage:<12.1f} {item.candidate.title}"
        )
    print("-" * (len(header) + title_width))
    winner: RankedCandidate = ranked[0]
    print(f"KEPUTUSAN: {winner.candidate.candidate_id} - {winner.candidate.title}")
    print("=" * (len(header) + title_width) + "\n")


# ----------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------
def main() -> None:
    """Menjalankan pipeline pembuatan matriks keputusan topik."""
    parser = argparse.ArgumentParser(
        description="Membangun matriks keputusan topik proyek (CP01)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROPOSAL_DIR,
        help="Direktori keluaran untuk matriks Markdown dan JSON.",
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir

    criteria: List[Criterion] = build_criteria()
    candidates: List[Candidate] = build_candidates()

    logger.info("Memvalidasi bobot kriteria dan matriks skor...")
    validate_criteria(criteria)
    validate_scores(candidates, criteria)

    ranked: List[RankedCandidate] = rank_candidates(candidates, criteria)
    for item in ranked:
        logger.info(
            "Peringkat %d: %s - skor tertimbang %.2f (%.1f%%)",
            item.rank,
            item.candidate.candidate_id,
            item.weighted_score,
            item.percentage,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path: Path = output_dir / MATRIX_MD_PATH.name
    json_path: Path = output_dir / MATRIX_JSON_PATH.name

    markdown_text: str = render_markdown(criteria, ranked)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_text)
    logger.info("Matriks Markdown tersimpan di: %s", markdown_path)

    payload: Dict[str, Any] = build_payload(criteria, ranked)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    logger.info("Matriks JSON tersimpan di: %s", json_path)

    print_summary_table(ranked)


if __name__ == "__main__":
    main()
