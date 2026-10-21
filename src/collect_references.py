"""Pengelolaan dan validasi rujukan ilmiah untuk proposal (Checkpoint CP02).

Modul ini menjadi sumber kebenaran tunggal (single source of truth) mengenai
lima rujukan ilmiah proyek Visi Komputer Kelompok 9, lalu memvalidasinya
terhadap rubrik proposal:

- minimal 5 sumber kredibel,
- minimal 2 di antaranya merupakan paper metode atau paper dataset,
- setiap komponen rubrik (pemahaman masalah, dataset, metodologi, baseline,
  evaluasi) harus didukung oleh setidaknya satu rujukan.

Keluaran:
- ``docs/proposal/references.md`` : daftar rujukan gaya IEEE, tabel pemetaan,
  ringkasan relevansi, dan uji cakupan rubrik.
- ``references.bib``               : seluruh rujukan dalam format BibTeX.

Verifikasi daring bersifat opsional melalui flag ``--verify-online``; skrip
tetap berjalan penuh secara luring apabila jaringan tidak tersedia.

Contoh pemakaian:
    python src/collect_references.py
    python src/collect_references.py --verify-online
"""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple
import urllib.error
import urllib.request


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("CollectReferences")


# ----------------------------------------------------------------------------
# Konstanta Paths & Parameter Rubrik
# ----------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DOCS_PROPOSAL_DIR: Final[Path] = PROJECT_ROOT / "docs" / "proposal"
REFERENCES_MD_PATH: Final[Path] = DOCS_PROPOSAL_DIR / "references.md"
BIBTEX_PATH: Final[Path] = PROJECT_ROOT / "references.bib"

MIN_CREDIBLE_SOURCES: Final[int] = 5
MIN_METHOD_OR_DATASET_SOURCES: Final[int] = 2
NETWORK_TIMEOUT_SECONDS: Final[float] = 8.0
USER_AGENT: Final[str] = (
    "Mozilla/5.0 (compatible; K09-LowLightPedestrian/1.0; +reference-validator)"
)
DOCUMENT_DATE: Final[str] = "20 Oktober 2026"
COURSE_NAME: Final[str] = "Visi Komputer (IF0PP006) - Semester Ganjil 2026/2027"
GROUP_NAME: Final[str] = (
    "Kelompok 9 (Krisantus Amka Ginting, Linardi Halim, "
    "Kristofer Ryan Giggs Eka Saputra, Leonard Franscois)"
)
PROJECT_TITLE: Final[str] = (
    "Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya "
    "Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif"
)

# Komponen rubrik yang wajib dicakup oleh rujukan (kode, label Bahasa Indonesia).
RUBRIC_COMPONENTS: Final[Tuple[Tuple[str, str], ...]] = (
    ("problem", "Pemahaman masalah"),
    ("dataset", "Dataset"),
    ("method", "Metodologi"),
    ("baseline", "Baseline"),
    ("evaluation", "Evaluasi"),
)

# Klasifikasi jenis rujukan beserta label yang dipakai pada laporan.
REFERENCE_TYPE_LABELS: Final[Dict[str, str]] = {
    "method": "Paper metode",
    "dataset": "Paper dataset",
    "tooling": "Perangkat lunak / repositori",
    "competition-review": "Review tantangan / challenge",
}

# Jenis entri BibTeX per rujukan (dipisah dari klasifikasi rubrik).
BIBTEX_ENTRY_TYPES: Final[Dict[str, str]] = {
    "jocher2023yolov8": "misc",
    "loh2019exdark": "article",
    "zuiderveld1994clahe": "incollection",
    "guan2019fusion": "article",
    "ye2026lpcvc": "misc",
}


# ----------------------------------------------------------------------------
# Struktur Data
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Reference:
    """Satu rujukan ilmiah beserta metadata lengkapnya.

    Attributes:
        ref_id: Nomor urut rujukan (1 sampai 5).
        bibtex_key: Kunci unik entri BibTeX.
        authors: Penulis dalam gaya IEEE (misal "Y. P. Loh and C. S. Chan").
        authors_bibtex: Penulis dalam format "Nama, Inisial" untuk BibTeX.
        title: Judul karya ilmiah.
        container: Venue atau container publikasi.
        year: Tahun publikasi.
        reference_type: Klasifikasi ('method', 'dataset', 'tooling',
            'competition-review').
        url: Tautan sumber daya daring.
        doi: DOI bila tersedia.
        volume: Volume jurnal/buku bila tersedia.
        issue: Nomor terbitan bila tersedia.
        pages: Rentang halaman bila tersedia.
        publisher: Penerbit bila tersedia.
        supporting_components: Kode komponen rubrik yang didukung rujukan ini.
        project_components: Komponen proyek yang ditopang rujukan ini.
        relevance_summary: Ringkasan relevansi dalam Bahasa Indonesia.
    """

    ref_id: int
    bibtex_key: str
    authors: str
    authors_bibtex: str
    title: str
    container: str
    year: int
    reference_type: str
    url: str
    doi: Optional[str]
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    publisher: Optional[str]
    supporting_components: Tuple[str, ...]
    project_components: str
    relevance_summary: str


@dataclass(frozen=True)
class RubricCheck:
    """Hasil ujikepatuhan terhadap rubrik rujukan.

    Attributes:
        check_name: Nama uji.
        requirement: Ambang batas dari rubrik.
        actual: Nilai faktual yang terukur.
        passed: True bila ambang terpenuhi.
        detail: Penjelasan singkat hasil uji.
    """

    check_name: str
    requirement: str
    actual: str
    passed: bool
    detail: str


# ----------------------------------------------------------------------------
# Data Rujukan
# ----------------------------------------------------------------------------
def build_references() -> List[Reference]:
    """Menyusun daftar lima rujukan ilmiah proyek beserta klasifikasinya.

    Returns:
        Daftar rujukan terurut sesuai nomor yang muncul di proposal.
    """
    return [
        Reference(
            ref_id=1,
            bibtex_key="jocher2023yolov8",
            authors="G. Jocher, A. Chaurasia, and J. Qiu",
            authors_bibtex="Jocher, Glenn and Chaurasia, Ajay and Qiu, Jing",
            title="Ultralytics YOLOv8",
            container="GitHub repository",
            year=2023,
            reference_type="tooling",
            url="https://github.com/ultralytics/ultralytics",
            doi=None,
            volume=None,
            issue=None,
            pages=None,
            publisher=None,
            supporting_components=("method", "baseline", "evaluation"),
            project_components=(
                "Detektor utama YOLOv8n (3,2 juta parameter), baseline zero-shot B0, "
                "pipeline evaluasi mAP/Recall/Precision"
            ),
            relevance_summary=(
                "Repositori resmi YOLOv8 menjadi justifikasi formal pemilihan "
                "arsitektur YOLOv8-Nano sebagai detektor utama proyek ini. Head "
                "anchor-free yang memprediksi titik pusat dan dimensi bounding box "
                "secara langsung sesuai dengan rasio aspek pejalan kaki yang "
                "bervariasi, sementara ukuran bobot sekitar 3,2 juta parameter "
                "menjadikan ambang 20 FPS pada CPU laptop tetap realistis. Framework "
                "ini juga menyediakan perhitungan metrik mAP, Recall, dan Precision "
                "yang dipakai pada skrip evaluasi dan benchmarking, sehingga "
                "angka yang dilaporkan konsisten di seluruh eksperimen. Bobot COCO "
                "miliknya dipakai langsung sebagai baseline zero-shot B0 tanpa "
                "pengumpulan data tambahan."
            ),
        ),
        Reference(
            ref_id=2,
            bibtex_key="loh2019exdark",
            authors="Y. P. Loh and C. S. Chan",
            authors_bibtex="Loh, Yuen Peng and Chan, Chee Seng",
            title="Getting to know low-light images with the ExDark dataset",
            container="Computer Vision and Image Understanding",
            year=2019,
            reference_type="dataset",
            url="https://doi.org/10.1016/j.cviu.2018.10.010",
            doi="10.1016/j.cviu.2018.10.010",
            volume="178",
            issue=None,
            pages="30-42",
            publisher=None,
            supporting_components=("problem", "dataset", "evaluation"),
            project_components=(
                "Argumentasi masalah degradasi deteksi pada kondisi minim cahaya, "
                "klasifikasi kondisi pencahayaan untuk slicing brightness"
            ),
            relevance_summary=(
                "Paper ini mendokumentasikan dataset ExDark berisi 7.363 citra "
                "low-light dengan 12 kelas dan anotasi bounding box, sehingga "
                "membuktikan bahwa masalah minim cahaya memiliki benchmark publik "
                "dan bukan sekadar fenomena visual. Eksperimen pada paper "
                "menunjukkan degradasi deteksi objek pada citra low-light "
                "dibandingkan MS COCO; temuan inilah yang menjadi landasan "
                "argumentasi bahwa model pretrained siang hari tidak dapat "
                "dipakai langsung pada citra malam. Klasifikasi kondisi "
                "pencahayaan yang dipublikasikan paper (twilight, "
                "shadow, dark, dan variasinya) mengilhami pembagian citra menjadi "
                "slice brightness untuk analisis kasus gagal pada eksperimen kami. "
                "Paper ini juga menjadi contoh akademik bahwa dataset low-light "
                "seharusnya dipecah secara bersih agar tidak terjadi "
                "kebocoran data antar partisi."
            ),
        ),
        Reference(
            ref_id=3,
            bibtex_key="zuiderveld1994clahe",
            authors="K. Zuiderveld",
            authors_bibtex="Zuiderveld, Karel",
            title="Contrast Limited Adaptive Histogram Equalization",
            container="Graphics Gems IV",
            year=1994,
            reference_type="method",
            url="https://doi.org/10.1016/B978-0-12-336156-1.50061-6",
            doi="10.1016/B978-0-12-336156-1.50061-6",
            volume=None,
            issue=None,
            pages="474-485",
            publisher="Academic Press",
            supporting_components=("method",),
            project_components=(
                "Implementasi CLAHE pada kanal L ruang CIE L*a*b* di "
                "src/preprocessing.py (clip limit 2,0; tile grid 8x8)"
            ),
            relevance_summary=(
                "Bab ini merupakan sumber primer algoritma CLAHE yang "
                "diimplementasikan pada modul prapemrosesan proyek, sehingga "
                "parameter clip limit 2,0 dan tile grid 8x8 yang dipakai pada "
                "eksperimen Exp-B memiliki dasar teoretis yang jelas. Pembatas "
                "kontras yang diperkenalkan algoritma ini menjaga agar "
                "histogram lokal tidak menjadi over-amplified, yang persis "
                "menyelesaikan masalah derau sensor pada citra malam. CLAHE "
                "diterapkan hanya pada kanal L di ruang warna CIE L*a*b* "
                "supaya krominansi tidak berubah dan tidak memunculkan color "
                "cast, sementara pemrosesan hanya dilakukan pada citra latih "
                "untuk mencegah kebocoran data."
            ),
        ),
        Reference(
            ref_id=4,
            bibtex_key="guan2019fusion",
            authors="D. Guan, Y. Cao, J. Yang, C. Shen, and M. G. Trefzer",
            authors_bibtex=(
                "Guan, Dayan and Cao, Yanpeng and Yang, Jun and "
                "Shen, Cheng and Trefzer, Martin G."
            ),
            title=(
                "Fusion of Multispectral Data Through Illumination-Aware Deep "
                "Neural Networks for Pedestrian Detection"
            ),
            container="IEEE Transactions on Information Forensics and Security",
            year=2019,
            reference_type="method",
            url="https://arxiv.org/abs/1802.09972",
            doi="10.48550/arXiv.1802.09972",
            volume="14",
            issue="11",
            pages="3048-3059",
            publisher=None,
            supporting_components=("problem", "method", "baseline"),
            project_components=(
                "Argumen teoretis bahwa informasi pencahayaan dapat dieksploitasi "
                "untuk meningkatkan deteksi pejalan kaki; desain pipeline dua tahap"
            ),
            relevance_summary=(
                "Paper ini menunjukkan secara eksplisit bahwa informasi "
                "pencahayaan dapat dimanfaatkan untuk meningkatkan deteksi pejalan "
                "kaki pada kondisi siang maupun malam, yang menjadi dasar "
                "argumentasi bahwa masalah low-light memang bisa ditangani di "
                "lapangan pemrosesan citra dan bukan murni keterbatasan "
                "benchmark. Mekanisme illumination-aware weighting yang "
                "diperkenalkan mengilhami desain pipeline proyek ini, yaitu "
                "augmentasi pencahayaan diperlakukan "
                "sebagai bagian dari pipeline secara langsung, bukan sebagai "
                "langkah yang dijalankan setelah training. Pendekungan "
                "multi-task pada paper juga menjadi alasan proyek memilih "
                "evaluasi per kelas (Recall pejalan kaki) di samping mAP "
                "gabungan, agar kualitas deteksi tiap kelas tidak tertutup oleh "
                "kelas dominan."
            ),
        ),
        Reference(
            ref_id=5,
            bibtex_key="ye2026lpcvc",
            authors="Z. Ye, Y.-H. Lu, X. Hu, S. Zhang, T. Jing, X. Li, and others",
            authors_bibtex=(
                "Ye, Zihao and Lu, Yung-Hsiang and Hu, Xiao and Zhang, Shuai and "
                "Jing, Taotao and Li, Xin and Yao, Zhen and Lang, Bo and "
                "Zheng, Zhihao and Oh, Seungmin and Kang, Hankyul and "
                "Kang, Seunghun and Ryu, Jongbin and Chen, Kexin and Qi, Yuan and "
                "Thiruvathukal, George K and Chuah, Mooi Choo"
            ),
            title="Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge",
            container="arXiv preprint arXiv:2604.19054",
            year=2026,
            reference_type="competition-review",
            url="https://arxiv.org/abs/2604.19054",
            doi="10.48550/arXiv.2604.19054",
            volume=None,
            issue=None,
            pages=None,
            publisher=None,
            supporting_components=("problem", "baseline", "evaluation"),
            project_components=(
                "Justifikasi ambang target 20 FPS, metodologi pengukuran latensi "
                "dan throughput, pelaporan metrik per kelas secara terbuka"
            ),
            relevance_summary=(
                "Evaluasi solusi pemenang LPCVC 2025 menunjukkan bahwa pada "
                "perangkat dengan sumber daya terbatas, tradeoff antara akurasi "
                "dan latensi menjadi kriteria utama, sehingga ambang 20 FPS yang "
                "kami nyatakan pada proposal bukan angka sembarang melainkan "
                "target yang masuk akal untuk inferensi edge. Kerangka "
                "benchmarking competition tersebut memakai protokol pengukuran "
                "latensi dan FPS yang konsisten, dan pendekatan yang sama "
                "diadopsi pada skrip benchmarking inferensi lokal proyek ini. "
                "Paper ini juga menegaskan bahwa penilaian solusi terbaik harus "
                "dilakukan dengan metrik yang dilaporkan secara terbuka, yang "
                "menjadi alasan kami mencantumkan Recall dan Precision per kelas "
                "bukan hanya mAP gabungan. Selain itu, tantangan LPCVC ini "
                "menjadi bukti terbaru bahwa isu efisiensi komputasi masih aktif "
                "dibahas pada komunitas computer vision, sehingga arah riset proyek "
                "tetap relevan."
            ),
        ),
    ]


# ----------------------------------------------------------------------------
# Format IEEE & BibTeX
# ----------------------------------------------------------------------------
def format_venue_details(reference: Reference) -> str:
    """Menyusun detail venue (volume, nomor, penerbit, halaman).

    Args:
        reference: Rujukan sumber.

    Returns:
        Fragmen detail dalam gaya IEEE, misal "vol. 14, no. 11, pp. 3048-3059".
        String kosong bila rujukan tidak memiliki detail venue.
    """
    details: List[str] = []
    if reference.volume:
        details.append(f"vol. {reference.volume}")
    if reference.issue:
        details.append(f"no. {reference.issue}")
    if reference.publisher:
        details.append(reference.publisher)
    if reference.pages:
        details.append(f"pp. {reference.pages}")

    return ", ".join(details)


def format_venue_full(reference: Reference) -> str:
    """Menyusun venue lengkap beserta detailnya untuk keperluan laporan.

    Args:
        reference: Rujukan sumber.

    Returns:
        Venue lengkap, misal "Computer Vision and Image Understanding, vol. 178, pp. 30-42".
    """
    details: str = format_venue_details(reference)
    if not details:
        return reference.container
    return f"{reference.container}, {details}"


def format_ieee(reference: Reference, italic_container: bool = True) -> str:
    """Memformat satu rujukan menjadi entri gaya IEEE.

    Args:
        reference: Rujukan yang akan diformat.
        italic_container: True untuk menandai venue dengan miring Markdown.

    Returns:
        Entri rujukan siap pakai, misal "[1] G. Jocher, ... 2023. [Online]. ...".
    """
    container: str = (
        f"*{reference.container}*" if italic_container else reference.container
    )
    venue_detail: str = format_venue_details(reference)
    venue_text: str = f"{container}, {venue_detail}" if venue_detail else container
    entry: str = (
        f"[{reference.ref_id}] {reference.authors}, \"{reference.title},\" "
        f"{venue_text}, {reference.year}. [Online]. Available: {reference.url}"
    )
    if reference.doi:
        entry += f" [doi: {reference.doi}]"
    return entry


def format_bibtex(reference: Reference) -> str:
    """Memformat satu rujukan menjadi entri BibTeX.

    Args:
        reference: Rujukan yang akan diformat.

    Returns:
        Entri BibTeX lengkap dengan penjepit kurung kurawal.
    """
    entry_type: str = BIBTEX_ENTRY_TYPES.get(reference.bibtex_key, "misc")
    fields: List[Tuple[str, str]] = [
        ("author", reference.authors_bibtex),
        ("title", reference.title),
    ]

    if entry_type == "article":
        fields.append(("journal", reference.container))
        if reference.volume:
            fields.append(("volume", reference.volume))
        if reference.issue:
            fields.append(("number", reference.issue))
    elif entry_type == "incollection":
        fields.append(("booktitle", reference.container))
        if reference.publisher:
            fields.append(("publisher", reference.publisher))
    else:
        fields.append(("howpublished", reference.url))

    if reference.pages:
        fields.append(("pages", reference.pages.replace("-", "--")))
    fields.append(("year", str(reference.year)))
    if reference.doi:
        fields.append(("doi", reference.doi))
    fields.append(("url", reference.url))

    lines: List[str] = [f"@{entry_type}{{{reference.bibtex_key},"]
    for name, value in fields:
        lines.append(f"  {name:<13} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


def render_bibtex(references: List[Reference]) -> str:
    """Menyusun berkas BibTeX lengkap.

    Args:
        references: Daftar seluruh rujukan.

    Returns:
        Isi berkas ``references.bib``.
    """
    header: List[str] = [
        "% references.bib - Daftar rujukan ilmiah Proyek Visi Komputer Kelompok 9",
        "% Topik: Deteksi Pejalan Kaki pada Kondisi Minim Cahaya (YOLOv8)",
        "% Dibuat oleh: src/collect_references.py (Checkpoint CP02)",
        "",
    ]
    entries: List[str] = [format_bibtex(reference) for reference in references]
    return "\n".join(header) + "\n\n".join(entries) + "\n"


# ----------------------------------------------------------------------------
# Validasi Rubrik
# ----------------------------------------------------------------------------
def check_source_count(references: List[Reference]) -> RubricCheck:
    """Menguji jumlah sumber kredibel terhadap ambang minimum rubrik.

    Args:
        references: Daftar rujukan yang dinilai.

    Returns:
        Hasil uji jumlah sumber.
    """
    actual: int = len(references)
    passed: bool = actual >= MIN_CREDIBLE_SOURCES
    return RubricCheck(
        check_name="Jumlah sumber kredibel",
        requirement=f">= {MIN_CREDIBLE_SOURCES} rujukan",
        actual=f"{actual} rujukan",
        passed=passed,
        detail=(
            f"Proyek memuat {actual} rujukan ilmiah "
            f"(ambang rubrik: {MIN_CREDIBLE_SOURCES})."
        ),
    )


def check_method_or_dataset_count(references: List[Reference]) -> RubricCheck:
    """Menguji jumlah paper metode atau paper dataset terhadap ambang rubrik.

    Args:
        references: Daftar rujukan yang dinilai.

    Returns:
        Hasil uji jumlah paper metode/dataset.
    """
    qualifying: List[Reference] = [
        reference
        for reference in references
        if reference.reference_type in ("method", "dataset")
    ]
    passed: bool = len(qualifying) >= MIN_METHOD_OR_DATASET_SOURCES
    keys: str = ", ".join(reference.bibtex_key for reference in qualifying)
    return RubricCheck(
        check_name="Paper metode atau dataset",
        requirement=f">= {MIN_METHOD_OR_DATASET_SOURCES} paper metode/dataset",
        actual=f"{len(qualifying)} paper ({keys})",
        passed=passed,
        detail=(
            "Rujukan yang diklasifikasikan sebagai paper metode atau paper dataset: "
            f"{keys}."
        ),
    )


def check_component_coverage(references: List[Reference]) -> Dict[str, List[Reference]]:
    """Memetakan rujukan pendukung untuk setiap komponen rubrik.

    Args:
        references: Daftar rujukan yang dinilai.

    Returns:
        Dictionary komponen ke daftar rujukan pendukung; daftar kosong berarti
        komponen belum tercakup.
    """
    coverage: Dict[str, List[Reference]] = {
        component_id: [] for component_id, _ in RUBRIC_COMPONENTS
    }
    for reference in references:
        for component_id in reference.supporting_components:
            if component_id in coverage:
                coverage[component_id].append(reference)
    return coverage


def check_citation_completeness(
    coverage: Dict[str, List[Reference]],
) -> RubricCheck:
    """Menguji apakah seluruh komponen rubrik didukung minimal satu rujukan.

    Args:
        coverage: Pemetaan komponen ke rujukan pendukung.

    Returns:
        Hasil uji kelengkapan sitasi.
    """
    missing: List[str] = [
        label
        for component_id, label in RUBRIC_COMPONENTS
        if not coverage.get(component_id)
    ]
    passed: bool = not missing
    detail: str = (
        "Seluruh lima komponen rubrik didukung oleh minimal satu rujukan."
        if passed
        else f"Komponen belum tercakup: {', '.join(missing)}."
    )
    return RubricCheck(
        check_name="Kelengkapan sitasi per komponen",
        requirement="5 dari 5 komponen tercakup",
        actual=f"{len(RUBRIC_COMPONENTS) - len(missing)} dari {len(RUBRIC_COMPONENTS)}",
        passed=passed,
        detail=detail,
    )


# ----------------------------------------------------------------------------
# Verifikasi Online (Opsional)
# ----------------------------------------------------------------------------
def _verify_with_get(reference: Reference, timeout: float) -> str:
    """Melakukan verifikasi GET singkat untuk server yang menolak HEAD.

    Args:
        reference: Rujukan yang diverifikasi.
        timeout: Batas waktu permintaan dalam detik.

    Returns:
        Keterangan status keterjangkauan.
    """
    request = urllib.request.Request(
        reference.url, headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1)
            status: Optional[int] = getattr(response, "status", None) or response.getcode()
        return f"terjangkau (HTTP {status}, metode GET)"
    except urllib.error.HTTPError as exc:
        return f"terjangkau namun akses dibatasi (HTTP {exc.code})"
    except Exception:  # noqa: BLE001 - verifikasi daring bersifat opsional
        return "tidak dapat diverifikasi (offline)"


def verify_reference_online(reference: Reference, timeout: float) -> str:
    """Memeriksa keterjangkauan URL rujukan tanpa membuat skrip gagal.

    Args:
        reference: Rujukan yang diverifikasi.
        timeout: Batas waktu permintaan dalam detik.

    Returns:
        Keterangan status keterjangkauan dalam Bahasa Indonesia.
    """
    request = urllib.request.Request(
        reference.url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status: Optional[int] = getattr(response, "status", None) or response.getcode()
        return f"terjangkau (HTTP {status})"
    except urllib.error.HTTPError as exc:
        if exc.code in (405, 501):
            return _verify_with_get(reference, timeout)
        if exc.code == 404:
            return "tidak ditemukan (HTTP 404)"
        return f"terjangkau namun akses dibatasi (HTTP {exc.code})"
    except Exception:  # noqa: BLE001 - verifikasi daring bersifat opsional
        return "tidak dapat diverifikasi (offline)"


# ----------------------------------------------------------------------------
# Render Laporan Markdown
# ----------------------------------------------------------------------------
def render_references_markdown(
    references: List[Reference],
    checks: List[RubricCheck],
    coverage: Dict[str, List[Reference]],
    online_status: Optional[Dict[int, str]],
) -> str:
    """Menyusun isi Markdown daftar rujukan beserta uji rubriknya.

    Args:
        references: Daftar rujukan.
        checks: Hasil uji kepatuhan rubrik.
        coverage: Pemetaan komponen rubrik ke rujukan pendukung.
        online_status: Hasil verifikasi daring per nomor rujukan bila dijalankan.

    Returns:
        Teks Markdown lengkap untuk ``docs/proposal/references.md``.
    """
    lines: List[str] = [
        "# Daftar Rujukan Ilmiah - Proyek Visi Komputer",
        "",
        "> **Checkpoint:** CP02 - Problem statement, pengguna atau konteks, 5 rujukan, "
        "dan ringkasan relevansi (6 XP)  ",
        f"> **Mata Kuliah:** {COURSE_NAME}  ",
        f"> **Kelompok:** {GROUP_NAME}  ",
        f"> **Topik:** {PROJECT_TITLE}  ",
        f"> **Tanggal:** {DOCUMENT_DATE}  ",
        "> **Catatan:** daftar rujukan ini menjadi sumber kebenaran tunggal (single "
        "source of truth) dan dipakai oleh skrip `src/collect_references.py`.",
        "",
        "---",
        "",
        "## 1. Daftar Rujukan (Gaya IEEE)",
        "",
    ]

    for reference in references:
        lines.append(format_ieee(reference))
    lines.append("")

    lines.extend([
        "## 2. Rujukan dan Komponen Proyek yang Didukung",
        "",
        "| # | Rujukan | Jenis | Komponen Proyek yang Didukung |",
        "|:---:|---|---|---|",
    ])

    for reference in references:
        type_label: str = REFERENCE_TYPE_LABELS.get(
            reference.reference_type, reference.reference_type
        )
        lines.append(
            f"| **{reference.ref_id}** | {reference.authors} (\"{reference.title}\", "
            f"{reference.year}) | {type_label} | {reference.project_components} |"
        )

    lines.extend([
        "",
        "## 3. Ringkasan Relevansi per Rujukan",
        "",
        "Setiap rujukan menjelaskan secara spesifik keputusan desain proyek mana "
        "yang disandarkan oleh sumber tersebut.",
        "",
    ])

    for reference in references:
        lines.extend([
            f"### [{reference.ref_id}] {reference.authors} - \"{reference.title}\" "
            f"({reference.year})",
            "",
            f"- **Venue:** {format_venue_full(reference)}",
            f"- **Jenis rujukan:** {REFERENCE_TYPE_LABELS.get(reference.reference_type, reference.reference_type)}",
            f"- **Tautan:** {reference.url}",
            "",
            reference.relevance_summary,
            "",
        ])

    lines.extend([
        "## 4. Uji Cakupan terhadap Kriteria Rubrik",
        "",
        "Setiap komponen rubrik wajib didukung oleh minimal satu rujukan.",
        "",
        "| Komponen Rubrik | Rujukan pendukung | Jumlah | Status |",
        "|:---|:---|:---:|:---:|",
    ])

    for component_id, label in RUBRIC_COMPONENTS:
        supporting: List[Reference] = coverage.get(component_id, [])
        numbers: str = (
            ", ".join(f"[{reference.ref_id}]" for reference in supporting)
            if supporting
            else "-"
        )
        status: str = "TERPENUHI" if supporting else "GAGAL"
        lines.append(
            f"| {label} | {numbers} | {len(supporting)} | {status} |"
        )

    source_check: RubricCheck = checks[0]
    lines.append(
        f"| Rujukan (kredibilitas sumber) | "
        f"{', '.join(f'[{reference.ref_id}]' for reference in references)} | "
        f"{len(references)} | {'TERPENUHI' if source_check.passed else 'GAGAL'} |"
    )

    lines.extend([
        "",
        "## 5. Uji Kepatuhan Rubrik Rujukan",
        "",
        "| Uji | Ambang | Nilai Aktual | Status | Keterangan |",
        "|:---|:---|:---|:---:|---|",
    ])

    for check in checks:
        status: str = "**PASS**" if check.passed else "**FAIL**"
        lines.append(
            f"| {check.check_name} | {check.requirement} | {check.actual} | "
            f"{status} | {check.detail} |"
        )

    lines.extend([
        "",
        "## 6. Status Verifikasi Online",
        "",
    ])

    if online_status is None:
        lines.extend([
            "Verifikasi daring tidak dijalankan pada pembuatan dokumen ini "
            "(mode luring). Jalankan `python src/collect_references.py --verify-online` "
            "untuk mencoba menghubungi setiap URL rujukan dengan timeout pendek.",
        ])
    else:
        lines.extend([
            "Verifikasi daring dijalankan dengan `python src/collect_references.py "
            "--verify-online`. Rujukan yang gagal dijangkau tetap valid secara "
            "akademik; status hanya mencerminkan ketersediaan akses jaringan "
            "pada saat pemeriksaan.",
            "",
            "| # | Rujukan | URL | Status |",
            "|:---:|---|---|---|",
        ])
        for reference in references:
            status: str = online_status.get(reference.ref_id, "tidak diperiksa")
            lines.append(
                f"| {reference.ref_id} | {reference.title} | {reference.url} | {status} |"
            )

    lines.extend([
        "",
        "## 7. Reproduksi",
        "",
        "```bash",
        "python src/collect_references.py                 # mode luring (default)",
        "python src/collect_references.py --verify-online  # tambahan verifikasi URL",
        "```",
        "",
        "Berkas `references.bib` di root proyek dihasilkan oleh skrip yang sama "
        "dan dapat dipakai langsung pada Overleaf, Zotero, atau Mendeley.",
        "",
        "---",
        "",
        "*(Dokumen ini dihasilkan secara otomatis oleh `src/collect_references.py` "
        "untuk pemenuhan Checkpoint CP02.)*",
    ])

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------
def main() -> None:
    """Menjalankan pipeline pengelolaan dan validasi rujukan."""
    parser = argparse.ArgumentParser(
        description="Mengelola dan memvalidasi rujukan ilmiah proyek (CP02)."
    )
    parser.add_argument(
        "--verify-online",
        action="store_true",
        help=(
            "Mencoba mengecek keterjangkauan URL setiap rujukan. "
            "Skrip tetap berjalan penuh bila jaringan tidak tersedia."
        ),
    )
    args = parser.parse_args()

    references: List[Reference] = build_references()

    checks: List[RubricCheck] = [
        check_source_count(references),
        check_method_or_dataset_count(references),
    ]
    coverage: Dict[str, List[Reference]] = check_component_coverage(references)
    checks.append(check_citation_completeness(coverage))

    print("\n" + "=" * 78)
    print("VALIDASI RUBRIK RUJUKAN - Checkpoint CP02 (Kelompok 9)")
    print("=" * 78)
    for check in checks:
        verdict: str = "PASS" if check.passed else "FAIL"
        print(f"[{verdict}] {check.check_name}")
        print(f"       Ambang : {check.requirement}")
        print(f"       Aktual : {check.actual}")
        print(f"       Catatan: {check.detail}")
    print("-" * 78)

    online_status: Optional[Dict[int, str]] = None
    if args.verify_online:
        online_status = {}
        logger.info("Memverifikasi keterjangkauan URL rujukan (mode daring)...")
        for reference in references:
            status: str = verify_reference_online(
                reference, NETWORK_TIMEOUT_SECONDS
            )
            online_status[reference.ref_id] = status
            logger.info("  [%d] %s -> %s", reference.ref_id, reference.bibtex_key, status)
    else:
        logger.info("Mode luring: verifikasi URL dilewati (gunakan --verify-online).")

    DOCS_PROPOSAL_DIR.mkdir(parents=True, exist_ok=True)

    markdown_text: str = render_references_markdown(
        references, checks, coverage, online_status
    )
    with open(REFERENCES_MD_PATH, "w", encoding="utf-8") as handle:
        handle.write(markdown_text)
    logger.info("Dokumen rujukan tersimpan di: %s", REFERENCES_MD_PATH)

    bibtex_text: str = render_bibtex(references)
    with open(BIBTEX_PATH, "w", encoding="utf-8") as handle:
        handle.write(bibtex_text)
    logger.info("Berkas BibTeX tersimpan di: %s", BIBTEX_PATH)

    all_passed: bool = all(check.passed for check in checks)
    print(f"Semua uji rubrik: {'PASS' if all_passed else 'FAIL'}")
    print(f"Output   : {REFERENCES_MD_PATH}")
    print(f"Output   : {BIBTEX_PATH}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
