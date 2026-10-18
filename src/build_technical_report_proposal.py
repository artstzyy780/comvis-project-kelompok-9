"""Script to build the official UTS Technical Report Proposal (v02) in docx, md, and pdf formats.

This module populates the official Technical_Report_Template.docx with Kelompok 9's
proposal content for nighttime pedestrian detection using YOLOv8 and adaptive illumination.
"""

from pathlib import Path
from typing import Final, List, Dict, Tuple
import mistune
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
from playwright.sync_api import sync_playwright

# Constants for file paths
TEMPLATE_DOCX: Final[Path] = Path("Technical_Report_Template.docx")
OUTPUT_DOCX_PROPOSAL: Final[Path] = Path("docs/proposal/K09_DeteksiPedestrianLowLight_Proposal_v02.docx")
ROOT_DOCX_PROPOSAL: Final[Path] = Path("K09_DeteksiPedestrianLowLight_Proposal_v02.docx")

OUTPUT_MD_PROPOSAL: Final[Path] = Path("docs/proposal/K09_DeteksiPedestrianLowLight_Proposal_v02.md")
ROOT_MD_PROPOSAL: Final[Path] = Path("K09_DeteksiPedestrianLowLight_Proposal_v02.md")

OUTPUT_PDF_PROPOSAL: Final[Path] = Path("docs/proposal/K09_DeteksiPedestrianLowLight_Proposal_v02.pdf")
ROOT_PDF_PROPOSAL: Final[Path] = Path("K09_DeteksiPedestrianLowLight_Proposal_v02.pdf")

OBSIDIAN_NOTE: Final[Path] = Path(r"C:\Project Github\OBSIDIAN comvis\projectCOMVIS\2026-10-03_technical_report_proposal_uts.md")


def set_cell_background(cell, fill_hex: str) -> None:
    """Sets background shading color of a docx table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def add_table_borders(table) -> None:
    """Applies subtle borders to a table."""
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
            f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
            f'<w:insideV w:val="none"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)


def build_docx_report() -> None:
    """Populates Technical_Report_Template.docx with Kelompok 9's proposal content."""
    if not TEMPLATE_DOCX.exists():
        raise FileNotFoundError(f"Template docx not found at {TEMPLATE_DOCX}")

    doc = docx.Document(TEMPLATE_DOCX)

    # 1. Title & Subtitle
    doc.paragraphs[0].text = "Computer Vision Final Project: Technical Report / Proposal"
    doc.paragraphs[1].text = "Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif"

    # 2. Authors Table (Table 0)
    author_table = doc.tables[0]
    author_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_names = author_table.cell(0, 0)
    cell_ids = author_table.cell(0, 1)

    cell_names.text = "Krisantus Linardi\nKristofer Leonard"
    cell_ids.text = "Kelompok 9 — Data & Preprocessing Lead\nKelompok 9 — Model & Deployment Lead"

    # 3. Main Text Paragraphs Mapping
    paragraph_updates = {
        3: ("Abstract", None),
        4: (
            "Abstract—This project addresses the critical challenge of nighttime pedestrian detection in Intelligent "
            "Transportation Systems (ITS) and Advanced Driver Assistance Systems (ADAS). Standard deep learning detectors "
            "often suffer severe performance degradation under low-light conditions due to low contrast, sensor noise, and glare "
            "from headlights. We propose a real-time computer vision pipeline utilizing the Ultralytics YOLOv8-Nano detector "
            "coupled with adaptive illumination preprocessing techniques, specifically Contrast Limited Adaptive Histogram Equalization "
            "(CLAHE) on the CIE L*a*b* L-channel and non-linear Gamma Correction. Evaluated on the Roboflow Low-light v2 dataset "
            "comprising 2,400 street images with 7,777 annotated objects (showing a 7.58:1 vehicle-to-person class imbalance), our study "
            "prioritizes pedestrian Recall to minimize life-threatening false negatives. Controlled experiments compare zero-shot baseline, "
            "fine-tuned baseline, and illumination-augmented variants on a strictly isolated test set. The proposed lightweight approach "
            "aims to boost pedestrian Recall by over 10% while maintaining real-time inference throughput (>= 20 FPS) on standard CPU hardware.",
            "Abstract"
        ),
        5: (
            "Abstrak—Proyek ini mengatasi tantangan krusial deteksi pejalan kaki di malam hari pada Sistem Transportasi Cerdas (ITS) "
            "dan Advanced Driver Assistance Systems (ADAS). Model detektor deep learning standar kerap mengalami degradasi performa drastis "
            "pada kondisi minim cahaya akibat kontras rendah, derau sensor, dan silau lampu kendaraan. Kami mengusulkan pipeline visi komputer "
            "real-time berbasis Ultralytics YOLOv8-Nano yang dipadukan dengan teknik prapemrosesan pencahayaan adaptif, yaitu Contrast Limited "
            "Adaptive Histogram Equalization (CLAHE) pada L-channel ruang warna CIE L*a*b* serta Gamma Correction non-linear. Dievaluasi pada dataset "
            "Roboflow Low-light v2 berukuran 2.400 citra jalanan dengan 7.777 objek teranotasi (memiliki ketidakseimbangan kelas kendaraan "
            "terhadap pejalan kaki sebesar 7,58:1), penelitian ini memprioritaskan metrik Recall pejalan kaki untuk meminimalkan kegagalan "
            "deteksi (false negative) yang membahayakan nyawa. Eksperimen terkontrol membandingkan baseline zero-shot, fine-tuned baseline, dan "
            "varian teraugmentasi pencahayaan pada split test yang terisolasi. Pendekatan ringan yang diusulkan menargetkan peningkatan Recall "
            "pejalan kaki sebesar >10% dengan tetap mempertahankan kecepatan inferensi real-time (>= 20 FPS) pada perangkat CPU standar.",
            "Abstrak"
        ),
        6: ("1\t1. Introduction / Pendahuluan", None),
        7: (
            "Pedestrian detection is a cornerstone safety component in smart mobility and autonomous driving applications. Statistically, a "
            "disproportionate number of fatal pedestrian accidents occur during twilight and nighttime hours when human visibility is severely "
            "compromised. Conventional object detection models trained predominantly on daytime datasets (such as MS COCO) struggle in nighttime "
            "environments because low photon counts reduce image contrast, amplify sensor grain, and induce extreme dynamic range from artificial light "
            "sources. Traditional image processing filters often fail to dynamically adapt to varying local darkness, while heavy multi-modal or "
            "generative low-light restoration networks demand excessive compute incompatible with real-time edge processing. Therefore, a deep "
            "learning object detection approach reinforced by adaptive, lightweight illumination enhancement is required to reliably extract spatial "
            "and textural features of pedestrians without introducing prohibitive computational latency.",
            None
        ),
        8: (
            "Deteksi pejalan kaki merupakan komponen keselamatan utama dalam mobilitas cerdas dan sistem kendali otonom. Secara statistik, sebagian "
            "besar kecelakaan fatal pejalan kaki terjadi pada malam hari ketika jarak pandang manusia sangat terbatas. Model deteksi objek konvensional "
            "yang dilatih mayoritas pada data siang hari (seperti MS COCO) mengalami penurunan keandalan di lingkungan malam karena intensitas foton "
            "yang rendah mengurangi kontras citra, mengamplifikasi derau sensor, dan menciptakan rentang dinamis ekstrem dari pencahayaan buatan. "
            "Filter pemrosesan citra tradisional tidak mampu beradaptasi secara dinamis terhadap tingkat kegelapan lokal, sedangkan model restorasi "
            "generatif yang berat membutuhkan daya komputasi tinggi yang tidak cocok untuk inferensi edge real-time. Oleh karena itu, pendekatan "
            "visi komputer deep learning yang diperkuat oleh prapemrosesan pencahayaan adaptif yang ringan sangat diperlukan untuk mengekstraksi "
            "fitur spasial pejalan kaki secara andal tanpa mengorbankan kecepatan inferensi.",
            None
        ),
        9: ("2\t2. Methodology / Metodologi", None),
        10: ("2.1   A. Data Strategy / Strategi Data", None),
        11: (
            "We utilize the Roboflow Low-light v2 dataset, licensed under CC BY 4.0, containing 2,400 nighttime traffic images at 640x640 resolution. "
            "A rigorous audit using our custom script (src/audit_dataset.py) verified 7,777 bounding box annotations partitioned into Train (2,100 "
            "images; 6,816 boxes), Validation (200 images; 650 boxes), and Test (100 images; 311 boxes) splits. The dataset exhibits a severe class "
            "imbalance, with 6,871 Vehicle boxes (88.4%) versus 906 Person boxes (11.6%), yielding a 7.58:1 ratio. Preprocessing involves L-channel "
            "CLAHE (clip limit = 2.0, tile grid = 8x8) in CIE L*a*b* space to selectively enhance local silhouette contrast, as well as Gamma "
            "Correction (gamma = 0.6) to brighten mid-tones without saturating highlights. To strictly prevent data leakage, all histogram fitting "
            "and augmentations are applied solely to the training set, keeping the test split completely unseen.",
            None
        ),
        12: (
            "Kami menggunakan dataset Roboflow Low-light v2 (lisensi CC BY 4.0) yang terdiri dari 2.400 citra lalu lintas malam beresolusi 640x640 "
            "piksel. Hasil audit ketat menggunakan skrip khusus (src/audit_dataset.py) mengonfirmasi 7.777 anotasi bounding box yang terbagi ke "
            "dalam split Train (2.100 citra; 6.816 box), Validasi (200 citra; 650 box), dan Uji (100 citra; 311 box). Dataset mengalami ketidakseimbangan "
            "kelas yang signifikan dengan 6.871 box Vehicle (88,4%) berbanding 906 box Person (11,6%), atau rasio 7,58:1. Prapemrosesan menerapkan "
            "CLAHE pada kanal L (clip limit = 2.0, tile grid = 8x8) dalam ruang warna CIE L*a*b* untuk memperjelas siluet lokal, serta Gamma Correction "
            "(gamma = 0.6) untuk mencerahkan tone sedang tanpa membuat area terang menjadi jenuh. Untuk mencegah kebocoran data (data leakage), seluruh "
            "kalkulasi histogram dan augmentasi hanya diterapkan pada data latih, sementara data uji dijaga murni.",
            None
        ),
        13: ("2.2   B. Model Architecture / Arsitektur Model", None),
        14: (
            "We adopt the lightweight Ultralytics YOLOv8-Nano (yolov8n.pt) architecture (~3.2 million parameters) due to its optimal balance "
            "between detection capability and low computational footprint on edge devices. YOLOv8 employs an anchor-free detection head that directly "
            "predicts object center points and bounding box dimensions, making it highly adaptable to varying pedestrian aspect ratios. Its backbone "
            "and neck utilize Cross Stage Partial C2f modules to effectively aggregate low-level spatial contours with high-level semantic context. "
            "Training is guided by a composite loss function comprising Complete IoU (CIoU) for bounding box spatial alignment, Distribution Focal "
            "Loss (DFL) for fine-grained boundary regression under blurry low-light conditions, and Binary Cross-Entropy (BCE) for class assignment. "
            "Hyperparameters are standardized: batch size = 16, initial learning rate = 0.01 (AdamW/SGD with cosine decay), random seed = 42, trained "
            "for 50 epochs with early stopping.",
            None
        ),
        15: (
            "Kami menggunakan arsitektur Ultralytics YOLOv8-Nano (yolov8n.pt) berukuran ~3,2 juta parameter karena memiliki keseimbangan terbaik "
            "antara akurasi deteksi dan efisiensi komputasi pada perangkat edge. YOLOv8 menggunakan head deteksi anchor-free yang secara langsung "
            "memprediksi titik pusat dan dimensi bounding box, sehingga sangat adaptif terhadap berbagai rasio aspek pejalan kaki. Backbone dan neck "
            "memanfaatkan modul C2f untuk menggabungkan kontur spasial tingkat rendah dengan konteks semantik tingkat tinggi. Pelatihan dipandu "
            "oleh kombinasi fungsi loss Complete IoU (CIoU) untuk akurasi bounding box, Distribution Focal Loss (DFL) untuk regresi batas objek yang "
            "kabur pada kondisi gelap, dan Binary Cross-Entropy (BCE) untuk klasifikasi objek. Hyperparameter distandarisasi: batch size = 16, "
            "learning rate awal = 0,01 (AdamW/SGD dengan cosine decay), random seed = 42, dilatih selama 50 epoch dengan early stopping.",
            None
        ),
        16: ("2.3   C. Deployment & Environment / Lingkungan & Penerapan", None),
        17: (
            "Model training and fine-tuning experiments are conducted using an NVIDIA T4 GPU environment hosted on Google Colab (16 GB VRAM, "
            "PyTorch framework). Evaluation of latencies and throughput benchmarking is performed on a local host machine equipped with an Intel "
            "Core i5/i7 processor and 16 GB RAM to verify real-time feasibility on standard consumer hardware. For deployment, the trained YOLOv8 "
            "model is integrated into a lightweight Python inference pipeline (OpenCV / Streamlit) that ingests video frames or static image streams, "
            "executes adaptive illumination preprocessing, and renders bounding box predictions with confidence scores at >= 20-25 FPS.",
            None
        ),
        18: (
            "Eksperimen pelatihan dan fine-tuning model dijalankan menggunakan perangkat keras GPU NVIDIA T4 pada Google Colab (16 GB VRAM, framework "
            "PyTorch). Pengukuran latensi dan throughput inferensi dilakukan pada perangkat lokal berprosesor Intel Core i5/i7 dan RAM 16 GB untuk "
            "memverifikasi kelayakan real-time pada perangkat keras umum. Untuk penerapan, model YOLOv8 yang telah dilatih diintegrasikan ke dalam "
            "pipeline inferensi Python (OpenCV / Streamlit) yang menerima masukan frame video atau gambar statis, menjalankan prapemrosesan pencahayaan "
            "adaptif, dan menampilkan prediksi bounding box beserta confidence score secara real-time (>= 20-25 FPS).",
            None
        ),
        19: ("3\t3. Results and Discussion / Hasil dan Pembahasan", None),
        20: (
            "Our quantitative evaluation framework prioritizes pedestrian Recall alongside standard object detection metrics (Precision, mAP@50, "
            "mAP@50:95, and FPS). Four controlled experiment variants are established on identical test splits: B0 (Zero-shot YOLOv8n COCO baseline), "
            "Exp-A (Fine-tuned YOLOv8n without illumination augmentation), Exp-B (Fine-tuned YOLOv8n with L-channel CLAHE augmentation), and Exp-C "
            "(Fine-tuned YOLOv8n with Gamma Correction). Dataset audit results (Table I) confirm a 7.58:1 class imbalance, proving that overall mAP "
            "can mask poor pedestrian detection if evaluated uncritically. Preliminary zero-shot baseline testing on nighttime validation images "
            "reveals significant false negatives where dark silhouetted pedestrians are missed entirely. The controlled experiment design aims to "
            "demonstrate that CLAHE and Gamma augmentations significantly improve pedestrian Recall (+10% target) without incurring excessive false "
            "positive rates.",
            None
        ),
        21: (
            "Kerangka evaluasi kuantitatif kami memprioritaskan Recall pejalan kaki di samping metrik deteksi standar (Precision, mAP@50, mAP@50:95, "
            "dan FPS). Empat varian eksperimen terkontrol ditetapkan pada split uji yang identik: B0 (Baseline zero-shot YOLOv8n COCO), Exp-A (Fine-tuned "
            "YOLOv8n tanpa augmentasi pencahayaan), Exp-B (Fine-tuned YOLOv8n dengan augmentasi CLAHE), dan Exp-C (Fine-tuned YOLOv8n dengan Gamma "
            "Correction). Hasil audit dataset (Tabel I) mengonfirmasi ketidakseimbangan kelas 7,58:1, membuktikan bahwa nilai mAP keseluruhan dapat "
            "mengeluhkan rendahnya deteksi pejalan kaki jika tidak dievaluasi per kelas. Pengujian awal baseline zero-shot menunjukkan tingginya "
            "false negative di mana pejalan kaki bersiluet gelap tidak terdeteksi sama sekali. Desain eksperimen terkontrol ini bertujuan membuktikan "
            "bahwa augmentasi CLAHE dan Gamma meningkatkan Recall pejalan kaki secara signifikan (target +10%) tanpa memicu peningkatan false "
            "positive berlebih.",
            None
        ),
        22: ("Visualizing Results / Visualisasi Hasil", None),
        23: (
            "Quantitative data distribution from the dataset audit is summarized in Table I. Controlled experiment configurations are detailed "
            "in Table II. Figure 1 illustrates visual annotations generated during the dataset audit phase (results/audit/sample_annotations.png), "
            "showing green bounding boxes for Person objects and blue boxes for Vehicle objects under realistic nighttime lighting. Qualitative "
            "failure case analysis plans focus on categorizing errors into False Negatives (pedestrians obscured by extreme shadows or heavy distance) "
            "and False Positives (misclassifying ambient structures such as poles or tree shadows due to noise over-amplification).",
            None
        ),
        24: (
            "Distribusi data kuantitatif dari hasil audit dataset dirangkum dalam Tabel I. Konfigurasi eksperimen terkontrol dirinci dalam Tabel II. "
            "Gambar 1 menampilkan visualisasi anotasi yang dihasilkan pada tahap audit data (results/audit/sample_annotations.png), yang memperlihatkan "
            "kotak hijau untuk objek Person dan kotak biru untuk objek Vehicle pada kondisi pencahayaan malam. Rencana analisis kasus gagal (failure "
            "cases) difokuskan pada pengkategorian kesalahan menjadi False Negatives (pejalan kaki terhalang bayangan gelap atau berjarak jauh) "
            "dan False Positives (salah mengklasifikasikan struktur lingkungan seperti tiang atau bayangan pohon akibat amplifikasi derau).",
            None
        ),
        25: ("4\t4. Conclusion / Kesimpulan", None),
        26: (
            "This technical proposal establishes a robust, controlled methodology to enhance nighttime pedestrian detection using YOLOv8-Nano "
            "and adaptive illumination augmentation (CLAHE and Gamma Correction). By addressing low contrast and sensor noise at the data "
            "preprocessing level while keeping model parameters lightweight (~3.2M), the proposed pipeline seeks to achieve a substantial boost in "
            "pedestrian Recall without sacrificing real-time inference throughput (>= 20 FPS). Future work will focus on completing all controlled "
            "training iterations, quantifying precision-recall trade-offs across brightness slices, and exploring low-bit INT8 quantization for "
            "deployment on low-power edge platforms.",
            None
        ),
        27: (
            "Proposal teknis ini menetapkan metodologi terkontrol yang kokoh untuk meningkatkan deteksi pejalan kaki di malam hari menggunakan "
            "YOLOv8-Nano dan augmentasi pencahayaan adaptif (CLAHE dan Gamma Correction). Dengan mengatasi masalah kontras rendah dan derau sensor pada "
            "tingkat prapemrosesan data tanpa menambah beban parameter model (~3,2M), pipeline ini menargetkan peningkatan signifikan pada Recall "
            "pejalan kaki tanpa mengorbankan kecepatan inferensi real-time (>= 20 FPS). Pekerjaan selanjutnya akan berfokus pada penyelesaian seluruh "
            "iterasi pelatihan terkontrol, pengukuran trade-off precision-recall pada berbagai tingkat kecerahan, serta eksplorasi kuantisasi INT8 "
            "untuk penerapan pada perangkat edge berdaya rendah.",
            None
        ),
        28: ("References", None)
    }

    for idx, (text_content, bold_lead) in paragraph_updates.items():
        if idx < len(doc.paragraphs):
            p = doc.paragraphs[idx]
            p.text = ""  # Clear text
            if bold_lead:
                run_lead = p.add_run(bold_lead)
                run_lead.bold = True
                run_rest = p.add_run(text_content[len(bold_lead):])
            else:
                p.add_run(text_content)

    # 4. Handle References Section (P29 and additional reference paragraphs)
    ref_list = [
        '[1]  G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," GitHub repository, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics',
        '[2]  Y. P. Loh and C. S. Chan, "Getting to know low-light images with The ExDark dataset," Computer Vision and Image Understanding, vol. 178, pp. 30-42, 2019.',
        '[3]  K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," Graphics Gems IV, Academic Press, pp. 474-485, 1994.',
        '[4]  D. Guan, Y. Cao, J. Yang, C. Shen, and M. G. Trefzer, "Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection," IEEE Transactions on Information Forensics and Security, vol. 14, no. 11, pp. 3048-3059, 2019.',
        '[5]  H. Ye et al., "Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge," arXiv preprint arXiv:2604.19054, 2026.'
    ]

    # Update paragraph 29
    if len(doc.paragraphs) > 29:
        p29 = doc.paragraphs[29]
        p29.text = ref_list[0]
        # Append remaining references
        for ref_str in ref_list[1:]:
            p_new = doc.add_paragraph(ref_str, style="Report Reference")

    # Save output docx safely
    OUTPUT_DOCX_PROPOSAL.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc.save(OUTPUT_DOCX_PROPOSAL)
        doc.save(ROOT_DOCX_PROPOSAL)
        print(f"[OK] DOCX Proposal generated:\n  - {OUTPUT_DOCX_PROPOSAL}\n  - {ROOT_DOCX_PROPOSAL}")
    except PermissionError:
        fallback_path = Path("K09_DeteksiPedestrianLowLight_Proposal_v02_fixed.docx")
        doc.save(fallback_path)
        print(f"[OK] DOCX Proposal generated (fallback because file is open in Word):\n  - {fallback_path}")


def build_md_report() -> None:
    """Generates the Markdown version of the proposal adhering to Technical Report Template."""
    md_content = r"""# Computer Vision Final Project: Technical Report / Proposal
## Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif

**Kode Berkas:** `K09_DeteksiPedestrianLowLight_Proposal_v02`  
**Mata Kuliah:** Visi Komputer (*Computer Vision* — IF0PP006)  
**Kelas / Semester:** 5PITA / Ganjil 2026/2027  
**Fase:** Ujian Tengah Semester (Sidang Proposal UTS)  

---

### Kelompok 9

| Nama Mahasiswa | Peran & Tanggung Jawab Utama |
|---|---|
| **Krisantus Linardi** | Kelompok 9 — Data & Preprocessing Lead |
| **Kristofer Leonard** | Kelompok 9 — Model & Deployment Lead |

---

### Abstract
Abstract—This project addresses the critical challenge of nighttime pedestrian detection in Intelligent Transportation Systems (ITS) and Advanced Driver Assistance Systems (ADAS). Standard deep learning detectors often suffer severe performance degradation under low-light conditions due to low contrast, sensor noise, and glare from headlights. We propose a real-time computer vision pipeline utilizing the Ultralytics YOLOv8-Nano detector coupled with adaptive illumination preprocessing techniques, specifically Contrast Limited Adaptive Histogram Equalization (CLAHE) on the CIE L\*a\*b\* L-channel and non-linear Gamma Correction. Evaluated on the Roboflow Low-light v2 dataset comprising 2,400 street images with 7,777 annotated objects (showing a 7.58:1 vehicle-to-person class imbalance), our study prioritizes pedestrian Recall to minimize life-threatening false negatives. Controlled experiments compare zero-shot baseline, fine-tuned baseline, and illumination-augmented variants on a strictly isolated test set. The proposed lightweight approach aims to boost pedestrian Recall by over 10% while maintaining real-time inference throughput ($\ge 20$ FPS) on standard CPU hardware.

### Abstrak
Abstrak—Proyek ini mengatasi tantangan krusial deteksi pejalan kaki di malam hari pada Sistem Transportasi Cerdas (ITS) dan Advanced Driver Assistance Systems (ADAS). Model detektor deep learning standar kerap mengalami degradasi performa drastis pada kondisi minim cahaya akibat kontras rendah, derau sensor, dan silau lampu kendaraan. Kami mengusulkan pipeline visi komputer real-time berbasis Ultralytics YOLOv8-Nano yang dipadukan dengan teknik prapemrosesan pencahayaan adaptif, yaitu Contrast Limited Adaptive Histogram Equalization (CLAHE) pada L-channel ruang warna CIE L\*a\*b\* serta Gamma Correction non-linear. Dievaluasi pada dataset Roboflow Low-light v2 berukuran 2.400 citra jalanan dengan 7.777 objek teranotasi (memiliki ketidakseimbangan kelas kendaraan terhadap pejalan kaki sebesar 7,58:1), penelitian ini memprioritaskan metrik Recall pejalan kaki untuk meminimalkan kegagalan deteksi (false negative) yang membahayakan nyawa. Eksperimen terkontrol membandingkan baseline zero-shot, fine-tuned baseline, dan varian teraugmentasi pencahayaan pada split test yang terisolasi. Pendekatan ringan yang diusulkan menargetkan peningkatan Recall pejalan kaki sebesar >10% dengan tetap mempertahankan kecepatan inferensi real-time ($\ge 20$ FPS) pada perangkat CPU standar.

---

## 1. Introduction / Pendahuluan

Pedestrian detection is a cornerstone safety component in smart mobility and autonomous driving applications. Statistically, a disproportionate number of fatal pedestrian accidents occur during twilight and nighttime hours when human visibility is severely compromised. Conventional object detection models trained predominantly on daytime datasets (such as MS COCO) struggle in nighttime environments because low photon counts reduce image contrast, amplify sensor grain, and induce extreme dynamic range from artificial light sources. Traditional image processing filters often fail to dynamically adapt to varying local darkness, while heavy multi-modal or generative low-light restoration networks demand excessive compute incompatible with real-time edge processing. Therefore, a deep learning object detection approach reinforced by adaptive, lightweight illumination enhancement is required to reliably extract spatial and textural features of pedestrians without introducing prohibitive computational latency.

Deteksi pejalan kaki merupakan komponen keselamatan utama dalam mobilitas cerdas dan sistem kendali otonom. Secara statistik, sebagian besar kecelakaan fatal pejalan kaki terjadi pada malam hari ketika jarak pandang manusia sangat terbatas. Model deteksi objek konvensional yang dilatih mayoritas pada data siang hari (seperti MS COCO) mengalami penurunan keandalan di lingkungan malam karena intensitas foton yang rendah mengurangi kontras citra, mengamplifikasi derau sensor, dan menciptakan rentang dinamis ekstrem dari pencahayaan buatan. Filter pemrosesan citra tradisional tidak mampu beradaptasi secara dinamis terhadap tingkat kegelapan lokal, sedangkan model restorasi generatif yang berat membutuhkan daya komputasi tinggi yang tidak cocok untuk inferensi edge real-time. Oleh karena itu, pendekatan visi komputer deep learning yang diperkuat oleh prapemrosesan pencahayaan adaptif yang ringan sangat diperlukan untuk mengekstraksi fitur spasial pejalan kaki secara andal tanpa mengorbankan kecepatan inferensi.

---

## 2. Methodology / Metodologi

### 2.1 A. Data Strategy / Strategi Data

We utilize the Roboflow Low-light v2 dataset, licensed under CC BY 4.0, containing 2,400 nighttime traffic images at 640x640 resolution. A rigorous audit using our custom script (`src/audit_dataset.py`) verified 7,777 bounding box annotations partitioned into Train (2,100 images; 6,816 boxes), Validation (200 images; 650 boxes), and Test (100 images; 311 boxes) splits. The dataset exhibits a severe class imbalance, with 6,871 Vehicle boxes (88.4%) versus 906 Person boxes (11.6%), yielding a 7.58:1 ratio. Preprocessing involves L-channel CLAHE (clip limit = 2.0, tile grid = 8x8) in CIE L\*a\*b\* space to selectively enhance local silhouette contrast, as well as Gamma Correction (gamma = 0.6) to brighten mid-tones without saturating highlights. To strictly prevent data leakage, all histogram fitting and augmentations are applied solely to the training set, keeping the test split completely unseen.

Kami menggunakan dataset Roboflow Low-light v2 (lisensi CC BY 4.0) yang terdiri dari 2.400 citra lalu lintas malam beresolusi 640x640 piksel. Hasil audit ketat menggunakan skrip khusus (`src/audit_dataset.py`) mengonfirmasi 7.777 anotasi bounding box yang terbagi ke dalam split Train (2.100 citra; 6.816 box), Validasi (200 citra; 650 box), dan Uji (100 citra; 311 box). Dataset mengalami ketidakseimbangan kelas yang signifikan dengan 6.871 box Vehicle (88,4%) berbanding 906 box Person (11,6%), atau rasio 7,58:1. Prapemrosesan menerapkan CLAHE pada kanal L (clip limit = 2.0, tile grid = 8x8) dalam ruang warna CIE L\*a\*b\* untuk memperjelas siluet lokal, serta Gamma Correction (gamma = 0.6) untuk mencerahkan tone sedang tanpa membuat area terang menjadi jenuh. Untuk mencegah kebocoran data (data leakage), seluruh kalkulasi histogram dan augmentasi hanya diterapkan pada data latih, sementara data uji dijaga murni.

### 2.2 B. Model Architecture / Arsitektur Model

We adopt the lightweight Ultralytics YOLOv8-Nano (`yolov8n.pt`) architecture (~3.2 million parameters) due to its optimal balance between detection capability and low computational footprint on edge devices. YOLOv8 employs an anchor-free detection head that directly predicts object center points and bounding box dimensions, making it highly adaptable to varying pedestrian aspect ratios. Its backbone and neck utilize Cross Stage Partial C2f modules to effectively aggregate low-level spatial contours with high-level semantic context. Training is guided by a composite loss function comprising Complete IoU (CIoU) for bounding box spatial alignment, Distribution Focal Loss (DFL) for fine-grained boundary regression under blurry low-light conditions, and Binary Cross-Entropy (BCE) for class assignment. Hyperparameters are standardized: batch size = 16, initial learning rate = 0.01 (AdamW/SGD with cosine decay), random seed = 42, trained for 50 epochs with early stopping.

Kami menggunakan arsitektur Ultralytics YOLOv8-Nano (`yolov8n.pt`) berukuran ~3,2 juta parameter karena memiliki keseimbangan terbaik antara akurasi deteksi dan efisiensi komputasi pada perangkat edge. YOLOv8 menggunakan head deteksi anchor-free yang secara langsung memprediksi titik pusat dan dimensi bounding box, sehingga sangat adaptif terhadap berbagai rasio aspek pejalan kaki. Backbone dan neck memanfaatkan modul C2f untuk menggabungkan kontur spasial tingkat rendah dengan konteks semantik tingkat tinggi. Pelatihan dipandu oleh kombinasi fungsi loss Complete IoU (CIoU) untuk akurasi bounding box, Distribution Focal Loss (DFL) untuk regresi batas objek yang kabur pada kondisi gelap, dan Binary Cross-Entropy (BCE) untuk klasifikasi objek. Hyperparameter distandarisasi: batch size = 16, learning rate awal = 0,01 (AdamW/SGD dengan cosine decay), random seed = 42, dilatih selama 50 epoch dengan early stopping.

### 2.3 C. Deployment & Environment / Lingkungan & Penerapan

Model training and fine-tuning experiments are conducted using an NVIDIA T4 GPU environment hosted on Google Colab (16 GB VRAM, PyTorch framework). Evaluation of latencies and throughput benchmarking is performed on a local host machine equipped with an Intel Core i5/i7 processor and 16 GB RAM to verify real-time feasibility on standard consumer hardware. For deployment, the trained YOLOv8 model is integrated into a lightweight Python inference pipeline (OpenCV / Streamlit) that ingests video frames or static image streams, executes adaptive illumination preprocessing, and renders bounding box predictions with confidence scores at $\ge 20$–$25$ FPS.

Eksperimen pelatihan dan fine-tuning model dijalankan menggunakan perangkat keras GPU NVIDIA T4 pada Google Colab (16 GB VRAM, framework PyTorch). Pengukuran latensi dan throughput inferensi dilakukan pada perangkat lokal berprosesor Intel Core i5/i7 dan RAM 16 GB untuk memverifikasi kelayakan real-time pada perangkat keras umum. Untuk penerapan, model YOLOv8 yang telah dilatih diintegrasikan ke dalam pipeline inferensi Python (OpenCV / Streamlit) yang menerima masukan frame video atau gambar statis, menjalankan prapemrosesan pencahayaan adaptif, dan menampilkan prediksi bounding box beserta confidence score secara real-time ($\ge 20$–$25$ FPS).

---

## 3. Results and Discussion / Hasil dan Pembahasan

Our quantitative evaluation framework prioritizes pedestrian Recall alongside standard object detection metrics (Precision, mAP@50, mAP@50:95, and FPS). Four controlled experiment variants are established on identical test splits: B0 (Zero-shot YOLOv8n COCO baseline), Exp-A (Fine-tuned YOLOv8n without illumination augmentation), Exp-B (Fine-tuned YOLOv8n with L-channel CLAHE augmentation), and Exp-C (Fine-tuned YOLOv8n with Gamma Correction). Dataset audit results (Table I) confirm a 7.58:1 class imbalance, proving that overall mAP can mask poor pedestrian detection if evaluated uncritically. Preliminary zero-shot baseline testing on nighttime validation images reveals significant false negatives where dark silhouetted pedestrians are missed entirely. The controlled experiment design aims to demonstrate that CLAHE and Gamma augmentations significantly improve pedestrian Recall (+10% target) without incurring excessive false positive rates.

Kerangka evaluasi kuantitatif kami memprioritaskan Recall pejalan kaki di samping metrik deteksi standar (Precision, mAP@50, mAP@50:95, dan FPS). Empat varian eksperimen terkontrol ditetapkan pada split uji yang identik: B0 (Baseline zero-shot YOLOv8n COCO), Exp-A (Fine-tuned YOLOv8n tanpa augmentasi pencahayaan), Exp-B (Fine-tuned YOLOv8n dengan augmentasi CLAHE), dan Exp-C (Fine-tuned YOLOv8n dengan Gamma Correction). Hasil audit dataset (Tabel I) mengonfirmasi ketidakseimbangan kelas 7,58:1, membuktikan bahwa nilai mAP keseluruhan dapat mengeluhkan rendahnya deteksi pejalan kaki jika tidak dievaluasi per kelas. Pengujian awal baseline zero-shot menunjukkan tingginya false negative di mana pejalan kaki bersiluet gelap tidak terdeteksi sama sekali. Desain eksperimen terkontrol ini bertujuan membuktikan bahwa augmentasi CLAHE dan Gamma meningkatkan Recall pejalan kaki secara signifikan (target +10%) tanpa memicu peningkatan false positive berlebih.

### Visualizing Results / Visualisasi Hasil

Quantitative data distribution from the dataset audit is summarized in Table I. Controlled experiment configurations are detailed in Table II. Figure 1 illustrates visual annotations generated during the dataset audit phase (`results/audit/sample_annotations.png`), showing green bounding boxes for Person objects and blue boxes for Vehicle objects under realistic nighttime lighting. Qualitative failure case analysis plans focus on categorizing errors into False Negatives (pedestrians obscured by extreme shadows or heavy distance) and False Positives (misclassifying ambient structures such as poles or tree shadows due to noise over-amplification).

Distribusi data kuantitatif dari hasil audit dataset dirangkum dalam Tabel I. Konfigurasi eksperimen terkontrol dirinci dalam Tabel II. Gambar 1 menampilkan visualisasi anotasi yang dihasilkan pada tahap audit data (`results/audit/sample_annotations.png`), yang memperlihatkan kotak hijau untuk objek Person dan kotak biru untuk objek Vehicle pada kondisi pencahayaan malam. Rencana analisis kasus gagal (failure cases) difokuskan pada pengkategorian kesalahan menjadi False Negatives (pejalan kaki terhalang bayangan gelap atau berjarak jauh) dan False Positives (salah mengklasifikasikan struktur lingkungan seperti tiang atau bayangan pohon akibat amplifikasi derau).

**Table I: Dataset Audit & Bounding Box Distribution per Split**

| Split Data | Jumlah Citra | Total Bounding Box | Box `Vehicle` | Box `person` | Rasio Imbalance |
|---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 2.100 | 6.816 | 6.018 (88,3%) | 798 (11,7%) | 7,54 : 1 |
| **Valid** | 200 | 650 | 581 (89,4%) | 69 (10,6%) | 8,42 : 1 |
| **Test** | 100 | 311 | 272 (87,5%) | 39 (12,5%) | 6,97 : 1 |
| **Total** | **2.400** | **7.777** | **6.871 (88,4%)** | **906 (11,6%)** | **7,58 : 1** |

**Table II: Controlled Experiment Design & Evaluation Framework**

| ID Eksperimen | Strategi Model | Augmentasi / Prapemrosesan | Split Evaluasi | Fokus Metrik Utama |
|---|---|---|---|---|
| **B0** | YOLOv8n (COCO) | None (Zero-shot baseline) | Test (100 citra) | Baseline Recall & mAP50 |
| **Exp-A** | Fine-tuned YOLOv8n | Augmentasi standar (tanpa pencerahan) | Test (100 citra) | Recall Domain Fine-Tuning |
| **Exp-B** | Fine-tuned YOLOv8n | L-channel CLAHE (clip=2.0, tile=8x8) | Test (100 citra) | Target Recall (+10%) & mAP |
| **Exp-C** | Fine-tuned YOLOv8n | Non-linear Gamma Correction ($\gamma=0.6$) | Test (100 citra) | Mid-tone Contrast Recall |

---

## 4. Conclusion / Kesimpulan

This technical proposal establishes a robust, controlled methodology to enhance nighttime pedestrian detection using YOLOv8-Nano and adaptive illumination augmentation (CLAHE and Gamma Correction). By addressing low contrast and sensor noise at the data preprocessing level while keeping model parameters lightweight (~3.2M), the proposed pipeline seeks to achieve a substantial boost in pedestrian Recall without sacrificing real-time inference throughput ($\ge 20$ FPS). Future work will focus on completing all controlled training iterations, quantifying precision-recall trade-offs across brightness slices, and exploring low-bit INT8 quantization for deployment on low-power edge platforms.

Proposal teknis ini menetapkan metodologi terkontrol yang kokoh untuk meningkatkan deteksi pejalan kaki di malam hari menggunakan YOLOv8-Nano dan augmentasi pencahayaan adaptif (CLAHE dan Gamma Correction). Dengan mengatasi masalah kontras rendah dan derau sensor pada tingkat prapemrosesan data tanpa menambah beban parameter model (~3,2M), pipeline ini menargetkan peningkatan signifikan pada Recall pejalan kaki tanpa mengorbankan kecepatan inferensi real-time ($\ge 20$ FPS). Pekerjaan selanjutnya akan berfokus pada penyelesaian seluruh iterasi pelatihan terkontrol, pengukuran trade-off precision-recall pada berbagai tingkat kecerahan, serta eksplorasi kuantisasi INT8 untuk penerapan pada perangkat edge berdaya rendah.

---

## References / Referensi

[1] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," GitHub repository, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics  
[2] Y. P. Loh and C. S. Chan, "Getting to know low-light images with The ExDark dataset," *Computer Vision and Image Understanding*, vol. 178, pp. 30–42, 2019.  
[3] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," *Graphics Gems IV*, Academic Press, pp. 474–485, 1994.  
[4] D. Guan, Y. Cao, J. Yang, C. Shen, and M. G. Trefzer, "Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection," *IEEE Transactions on Information Forensics and Security*, vol. 14, no. 11, pp. 3048–3059, 2019.  
[5] H. Ye et al., "Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge," *arXiv preprint arXiv:2604.19054*, 2026.
"""
    OUTPUT_MD_PROPOSAL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PROPOSAL.write_text(md_content, encoding="utf-8")
    ROOT_MD_PROPOSAL.write_text(md_content, encoding="utf-8")
    print(f"[OK] MD Proposal generated:\n  - {OUTPUT_MD_PROPOSAL}\n  - {ROOT_MD_PROPOSAL}")


def build_pdf_report() -> None:
    """Renders HTML template into IEEE 2-column styled PDF using Playwright."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Computer Vision Technical Report Proposal</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ]
        });"></script>
    <style>
        @page {
            size: A4;
            margin: 16mm 14mm 16mm 14mm;
        }

        body {
            font-family: 'Times New Roman', Times, serif;
            font-size: 9pt;
            line-height: 1.35;
            color: #000;
            margin: 0;
            padding: 0;
            text-align: justify;
        }

        /* Title Block (1-Column) */
        .header-title-block {
            text-align: center;
            margin-bottom: 12px;
            border-bottom: 1px solid #000;
            padding-bottom: 8px;
        }

        .header-title-block h1 {
            font-size: 16pt;
            font-weight: bold;
            color: #17365D;
            margin: 0 0 4px 0;
            text-transform: uppercase;
        }

        .header-title-block h2 {
            font-size: 11pt;
            font-weight: normal;
            font-style: italic;
            color: #005B97;
            margin: 0 0 8px 0;
        }

        .author-table {
            width: 80%;
            margin: 0 auto;
            border-collapse: collapse;
            font-size: 8.5pt;
        }

        .author-table td {
            text-align: center;
            padding: 4px;
            vertical-align: top;
            border: none;
        }

        /* 2-Column Section Layout */
        .two-column-wrapper {
            column-count: 2;
            column-gap: 16px;
            column-fill: balance;
        }

        h2.section-head {
            font-size: 10pt;
            font-weight: bold;
            color: #005B97;
            text-transform: uppercase;
            border-bottom: 0.5pt solid #005B97;
            padding-bottom: 2px;
            margin-top: 10px;
            margin-bottom: 4px;
            break-after: avoid;
        }

        h3.subsection-head {
            font-size: 9.5pt;
            font-weight: bold;
            color: #000;
            margin-top: 8px;
            margin-bottom: 3px;
            break-after: avoid;
        }

        p {
            margin: 0 0 6px 0;
            text-indent: 12px;
        }

        p.no-indent {
            text-indent: 0;
        }

        .lead-bold {
            font-weight: bold;
        }

        table.data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 8px 0;
            font-size: 7.5pt;
            break-inside: avoid;
        }

        table.data-table th, table.data-table td {
            border: 0.5pt solid #64748b;
            padding: 3px 4px;
            text-align: center;
        }

        table.data-table th {
            background-color: #f1f5f9;
            font-weight: bold;
        }

        .table-caption {
            font-size: 8pt;
            font-weight: bold;
            text-align: center;
            margin-top: 6px;
            margin-bottom: 3px;
        }

        ul, ol {
            margin: 0 0 6px 0;
            padding-left: 16px;
        }

        .reference-item {
            font-size: 8pt;
            text-indent: -14px;
            padding-left: 14px;
            margin-bottom: 4px;
        }
    </style>
</head>
<body>

<div class="header-title-block">
    <h1>Computer Vision Final Project: Technical Report / Proposal</h1>
    <h2>Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif</h2>
    
    <table class="author-table">
        <tr>
            <td><strong>Krisantus Linardi</strong><br>Kelompok 9 — Data & Preprocessing Lead</td>
            <td><strong>Kristofer Leonard</strong><br>Kelompok 9 — Model & Deployment Lead</td>
        </tr>
    </table>
</div>

<div class="two-column-wrapper">

    <h2 class="section-head">Abstract</h2>
    <p class="no-indent"><span class="lead-bold">Abstract—</span>This project addresses the critical challenge of nighttime pedestrian detection in Intelligent Transportation Systems (ITS) and Advanced Driver Assistance Systems (ADAS). Standard deep learning detectors often suffer severe performance degradation under low-light conditions due to low contrast, sensor noise, and glare from headlights. We propose a real-time computer vision pipeline utilizing the Ultralytics YOLOv8-Nano detector coupled with adaptive illumination preprocessing techniques, specifically Contrast Limited Adaptive Histogram Equalization (CLAHE) on the CIE L*a*b* L-channel and non-linear Gamma Correction. Evaluated on the Roboflow Low-light v2 dataset comprising 2,400 street images with 7,777 annotated objects (showing a 7.58:1 vehicle-to-person class imbalance), our study prioritizes pedestrian Recall to minimize life-threatening false negatives. Controlled experiments compare zero-shot baseline, fine-tuned baseline, and illumination-augmented variants on a strictly isolated test set. The proposed lightweight approach aims to boost pedestrian Recall by over 10% while maintaining real-time inference throughput (&ge; 20 FPS) on standard CPU hardware.</p>

    <p class="no-indent"><span class="lead-bold">Abstrak—</span>Proyek ini mengatasi tantangan krusial deteksi pejalan kaki di malam hari pada Sistem Transportasi Cerdas (ITS) dan Advanced Driver Assistance Systems (ADAS). Model detektor deep learning standar kerap mengalami degradasi performa drastis pada kondisi minim cahaya akibat kontras rendah, derau sensor, dan silau lampu kendaraan. Kami mengusulkan pipeline visi komputer real-time berbasis Ultralytics YOLOv8-Nano yang dipadukan dengan teknik prapemrosesan pencahayaan adaptif, yaitu Contrast Limited Adaptive Histogram Equalization (CLAHE) pada L-channel ruang warna CIE L*a*b* serta Gamma Correction non-linear. Dievaluasi pada dataset Roboflow Low-light v2 berukuran 2.400 citra jalanan dengan 7.777 objek teranotasi (memiliki ketidakseimbangan kelas kendaraan terhadap pejalan kaki sebesar 7,58:1), penelitian ini memprioritaskan metrik Recall pejalan kaki untuk meminimalkan kegagalan deteksi (false negative) yang membahayakan nyawa. Eksperimen terkontrol membandingkan baseline zero-shot, fine-tuned baseline, dan varian teraugmentasi pencahayaan pada split test yang terisolasi. Pendekatan ringan yang diusulkan menargetkan peningkatan Recall pejalan kaki sebesar >10% dengan tetap mempertahankan kecepatan inferensi real-time (&ge; 20 FPS) pada perangkat CPU standar.</p>

    <h2 class="section-head">1. Introduction / Pendahuluan</h2>
    <p>Pedestrian detection is a cornerstone safety component in smart mobility and autonomous driving applications. Statistically, a disproportionate number of fatal pedestrian accidents occur during twilight and nighttime hours when human visibility is severely compromised. Conventional object detection models trained predominantly on daytime datasets (such as MS COCO) struggle in nighttime environments because low photon counts reduce image contrast, amplify sensor grain, and induce extreme dynamic range from artificial light sources. Traditional image processing filters often fail to dynamically adapt to varying local darkness, while heavy multi-modal or generative low-light restoration networks demand excessive compute incompatible with real-time edge processing. Therefore, a deep learning object detection approach reinforced by adaptive, lightweight illumination enhancement is required to reliably extract spatial and textural features of pedestrians without introducing prohibitive computational latency.</p>

    <p>Deteksi pejalan kaki merupakan komponen keselamatan utama dalam mobilitas cerdas dan sistem kendali otonom. Secara statistik, sebagian besar kecelakaan fatal pejalan kaki terjadi pada malam hari ketika jarak pandang manusia sangat terbatas. Model deteksi objek konvensional yang dilatih mayoritas pada data siang hari (seperti MS COCO) mengalami penurunan keandalan di lingkungan malam karena intensitas foton yang rendah mengurangi kontras citra, mengamplifikasi derau sensor, dan menciptakan rentang dinamis ekstrem dari pencahayaan buatan. Filter pemrosesan citra tradisional tidak mampu beradaptasi secara dinamis terhadap tingkat kegelapan lokal, sedangkan model restorasi generatif yang berat membutuhkan daya komputasi tinggi yang tidak cocok untuk inferensi edge real-time. Oleh karena itu, pendekatan visi komputer deep learning yang diperkuat oleh prapemrosesan pencahayaan adaptif yang ringan sangat diperlukan untuk mengekstraksi fitur spasial pejalan kaki secara andal tanpa mengorbankan kecepatan inferensi.</p>

    <h2 class="section-head">2. Methodology / Metodologi</h2>
    
    <h3 class="subsection-head">2.1 A. Data Strategy / Strategi Data</h3>
    <p>We utilize the Roboflow Low-light v2 dataset, licensed under CC BY 4.0, containing 2,400 nighttime traffic images at 640x640 resolution. A rigorous audit using our custom script (src/audit_dataset.py) verified 7,777 bounding box annotations partitioned into Train (2,100 images; 6,816 boxes), Validation (200 images; 650 boxes), and Test (100 images; 311 boxes) splits. The dataset exhibits a severe class imbalance, with 6,871 Vehicle boxes (88.4%) versus 906 Person boxes (11.6%), yielding a 7.58:1 ratio. Preprocessing involves L-channel CLAHE (clip limit = 2.0, tile grid = 8x8) in CIE L*a*b* space to selectively enhance local silhouette contrast, as well as Gamma Correction (gamma = 0.6) to brighten mid-tones without saturating highlights. To strictly prevent data leakage, all histogram fitting and augmentations are applied solely to the training set, keeping the test split completely unseen.</p>

    <p>Kami menggunakan dataset Roboflow Low-light v2 (lisensi CC BY 4.0) yang terdiri dari 2.400 citra lalu lintas malam beresolusi 640x640 piksel. Hasil audit ketat menggunakan skrip khusus (src/audit_dataset.py) mengonfirmasi 7.777 anotasi bounding box yang terbagi ke dalam split Train (2.100 citra; 6.816 box), Validasi (200 citra; 650 box), dan Uji (100 citra; 311 box). Dataset mengalami ketidakseimbangan kelas yang signifikan dengan 6.871 box Vehicle (88,4%) berbanding 906 box Person (11,6%), atau rasio 7,58:1. Prapemrosesan menerapkan CLAHE pada kanal L (clip limit = 2.0, tile grid = 8x8) dalam ruang warna CIE L*a*b* untuk memperjelas siluet lokal, serta Gamma Correction (gamma = 0.6) untuk mencerahkan tone sedang tanpa membuat area terang menjadi jenuh. Untuk mencegah kebocoran data (data leakage), seluruh kalkulasi histogram dan augmentasi hanya diterapkan pada data latih, sementara data uji dijaga murni.</p>

    <h3 class="subsection-head">2.2 B. Model Architecture / Arsitektur Model</h3>
    <p>We adopt the lightweight Ultralytics YOLOv8-Nano (yolov8n.pt) architecture (~3.2 million parameters) due to its optimal balance between detection capability and low computational footprint on edge devices. YOLOv8 employs an anchor-free detection head that directly predicts object center points and bounding box dimensions, making it highly adaptable to varying pedestrian aspect ratios. Its backbone and neck utilize Cross Stage Partial C2f modules to effectively aggregate low-level spatial contours with high-level semantic context. Training is guided by a composite loss function comprising Complete IoU (CIoU) for bounding box spatial alignment, Distribution Focal Loss (DFL) for fine-grained boundary regression under blurry low-light conditions, and Binary Cross-Entropy (BCE) for class assignment. Hyperparameters are standardized: batch size = 16, initial learning rate = 0.01 (AdamW/SGD with cosine decay), random seed = 42, trained for 50 epochs with early stopping.</p>

    <p>Kami menggunakan arsitektur Ultralytics YOLOv8-Nano (yolov8n.pt) berukuran ~3,2 juta parameter karena memiliki keseimbangan terbaik antara akurasi deteksi dan efisiensi komputasi pada perangkat edge. YOLOv8 menggunakan head deteksi anchor-free yang secara langsung memprediksi titik pusat dan dimensi bounding box, sehingga sangat adaptif terhadap berbagai rasio aspek pejalan kaki. Backbone dan neck memanfaatkan modul C2f untuk menggabungkan kontur spasial tingkat rendah dengan konteks semantik tingkat tinggi. Pelatihan dipandu oleh kombinasi fungsi loss Complete IoU (CIoU) untuk akurasi bounding box, Distribution Focal Loss (DFL) untuk regresi batas objek yang kabur pada kondisi gelap, dan Binary Cross-Entropy (BCE) untuk klasifikasi objek. Hyperparameter distandarisasi: batch size = 16, learning rate awal = 0,01 (AdamW/SGD dengan cosine decay), random seed = 42, dilatih selama 50 epoch dengan early stopping.</p>

    <h3 class="subsection-head">2.3 C. Deployment & Environment / Lingkungan & Penerapan</h3>
    <p>Model training and fine-tuning experiments are conducted using an NVIDIA T4 GPU environment hosted on Google Colab (16 GB VRAM, PyTorch framework). Evaluation of latencies and throughput benchmarking is performed on a local host machine equipped with an Intel Core i5/i7 processor and 16 GB RAM to verify real-time feasibility on standard consumer hardware. For deployment, the trained YOLOv8 model is integrated into a lightweight Python inference pipeline (OpenCV / Streamlit) that ingests video frames or static image streams, executes adaptive illumination preprocessing, and renders bounding box predictions with confidence scores at &ge; 20-25 FPS.</p>

    <p>Eksperimen pelatihan dan fine-tuning model dijalankan menggunakan perangkat keras GPU NVIDIA T4 pada Google Colab (16 GB VRAM, framework PyTorch). Pengukuran latensi dan throughput inferensi dilakukan pada perangkat lokal berprosesor Intel Core i5/i7 dan RAM 16 GB untuk memverifikasi kelayakan real-time pada perangkat keras umum. Untuk penerapan, model YOLOv8 yang telah dilatih diintegrasikan ke dalam pipeline inferensi Python (OpenCV / Streamlit) yang menerima masukan frame video atau gambar statis, menjalankan prapemrosesan pencahayaan adaptif, dan menampilkan prediksi bounding box beserta confidence score secara real-time (&ge; 20-25 FPS).</p>

    <h2 class="section-head">3. Results and Discussion / Hasil dan Pembahasan</h2>
    <p>Our quantitative evaluation framework prioritizes pedestrian Recall alongside standard object detection metrics (Precision, mAP@50, mAP@50:95, and FPS). Four controlled experiment variants are established on identical test splits: B0 (Zero-shot YOLOv8n COCO baseline), Exp-A (Fine-tuned YOLOv8n without illumination augmentation), Exp-B (Fine-tuned YOLOv8n with L-channel CLAHE augmentation), and Exp-C (Fine-tuned YOLOv8n with Gamma Correction). Dataset audit results (Table I) confirm a 7.58:1 class imbalance, proving that overall mAP can mask poor pedestrian detection if evaluated uncritically. Preliminary zero-shot baseline testing on nighttime validation images reveals significant false negatives where dark silhouetted pedestrians are missed entirely. The controlled experiment design aims to demonstrate that CLAHE and Gamma augmentations significantly improve pedestrian Recall (+10% target) without incurring excessive false positive rates.</p>

    <p>Kerangka evaluasi kuantitatif kami memprioritaskan Recall pejalan kaki di samping metrik deteksi standar (Precision, mAP@50, mAP@50:95, dan FPS). Empat varian eksperimen terkontrol ditetapkan pada split uji yang identik: B0 (Baseline zero-shot YOLOv8n COCO), Exp-A (Fine-tuned YOLOv8n tanpa augmentasi pencahayaan), Exp-B (Fine-tuned YOLOv8n dengan augmentasi CLAHE), dan Exp-C (Fine-tuned YOLOv8n dengan Gamma Correction). Hasil audit dataset (Tabel I) mengonfirmasi ketidakseimbangan kelas 7,58:1, membuktikan bahwa nilai mAP keseluruhan dapat mengeluhkan rendahnya deteksi pejalan kaki jika tidak dievaluasi per kelas. Pengujian awal baseline zero-shot menunjukkan tingginya false negative di mana pejalan kaki bersiluet gelap tidak terdeteksi sama sekali. Desain eksperimen terkontrol ini bertujuan membuktikan bahwa augmentasi CLAHE dan Gamma meningkatkan Recall pejalan kaki secara signifikan (target +10%) tanpa memicu peningkatan false positive berlebih.</p>

    <h3 class="subsection-head">Visualizing Results / Visualisasi Hasil</h3>
    <p>Quantitative data distribution from the dataset audit is summarized in Table I. Controlled experiment configurations are detailed in Table II. Figure 1 illustrates visual annotations generated during the dataset audit phase (results/audit/sample_annotations.png), showing green bounding boxes for Person objects and blue boxes for Vehicle objects under realistic nighttime lighting. Qualitative failure case analysis plans focus on categorizing errors into False Negatives (pedestrians obscured by extreme shadows or heavy distance) and False Positives (misclassifying ambient structures such as poles or tree shadows due to noise over-amplification).</p>

    <div class="table-caption">TABLE I: Dataset Audit & Bounding Box Distribution per Split</div>
    <table class="data-table">
        <thead>
            <tr>
                <th>Split Data</th>
                <th>Citra</th>
                <th>Total Box</th>
                <th>Box Vehicle</th>
                <th>Box Person</th>
                <th>Imbalance</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Train</td>
                <td>2.100</td>
                <td>6.816</td>
                <td>6.018 (88,3%)</td>
                <td>798 (11,7%)</td>
                <td>7,54 : 1</td>
            </tr>
            <tr>
                <td>Valid</td>
                <td>200</td>
                <td>650</td>
                <td>581 (89,4%)</td>
                <td>69 (10,6%)</td>
                <td>8,42 : 1</td>
            </tr>
            <tr>
                <td>Test</td>
                <td>100</td>
                <td>311</td>
                <td>272 (87,5%)</td>
                <td>39 (12,5%)</td>
                <td>6,97 : 1</td>
            </tr>
            <tr>
                <td><strong>Total</strong></td>
                <td><strong>2.400</strong></td>
                <td><strong>7.777</strong></td>
                <td><strong>6.871 (88,4%)</strong></td>
                <td><strong>906 (11,6%)</strong></td>
                <td><strong>7,58 : 1</strong></td>
            </tr>
        </tbody>
    </table>

    <div class="table-caption">TABLE II: Controlled Experiment Design & Evaluation Framework</div>
    <table class="data-table">
        <thead>
            <tr>
                <th>ID Exp</th>
                <th>Strategi Model</th>
                <th>Augmentasi / Prep</th>
                <th>Target Split</th>
                <th>Fokus Metrik</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>B0</td>
                <td>YOLOv8n (COCO)</td>
                <td>None (Zero-shot)</td>
                <td>Test (100)</td>
                <td>Baseline Recall & mAP</td>
            </tr>
            <tr>
                <td>Exp-A</td>
                <td>Fine-tuned YOLOv8n</td>
                <td>Augmentasi standar</td>
                <td>Test (100)</td>
                <td>Recall Fine-Tuning</td>
            </tr>
            <tr>
                <td>Exp-B</td>
                <td>Fine-tuned YOLOv8n</td>
                <td>L-channel CLAHE</td>
                <td>Test (100)</td>
                <td>Target Recall (+10%)</td>
            </tr>
            <tr>
                <td>Exp-C</td>
                <td>Fine-tuned YOLOv8n</td>
                <td>Gamma (&gamma;=0.6)</td>
                <td>Test (100)</td>
                <td>Mid-tone Contrast Recall</td>
            </tr>
        </tbody>
    </table>

    <h2 class="section-head">4. Conclusion / Kesimpulan</h2>
    <p>This technical proposal establishes a robust, controlled methodology to enhance nighttime pedestrian detection using YOLOv8-Nano and adaptive illumination augmentation (CLAHE and Gamma Correction). By addressing low contrast and sensor noise at the data preprocessing level while keeping model parameters lightweight (~3.2M), the proposed pipeline seeks to achieve a substantial boost in pedestrian Recall without sacrificing real-time inference throughput (&ge; 20 FPS). Future work will focus on completing all controlled training iterations, quantifying precision-recall trade-offs across brightness slices, and exploring low-bit INT8 quantization for deployment on low-power edge platforms.</p>

    <p>Proposal teknis ini menetapkan metodologi terkontrol yang kokoh untuk meningkatkan deteksi pejalan kaki di malam hari menggunakan YOLOv8-Nano dan augmentasi pencahayaan adaptif (CLAHE dan Gamma Correction). Dengan mengatasi masalah kontras rendah dan derau sensor pada tingkat prapemrosesan data tanpa menambah beban parameter model (~3,2M), pipeline ini menargetkan peningkatan signifikan pada Recall pejalan kaki tanpa mengorbankan kecepatan inferensi real-time (&ge; 20 FPS). Pekerjaan selanjutnya akan berfokus pada penyelesaian seluruh iterasi pelatihan terkontrol, pengukuran trade-off precision-recall pada berbagai tingkat kecerahan, serta eksplorasi kuantisasi INT8 untuk penerapan pada perangkat edge berdaya rendah.</p>

    <h2 class="section-head">References / Referensi</h2>
    <div class="reference-item">[1] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," GitHub repository, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics</div>
    <div class="reference-item">[2] Y. P. Loh and C. S. Chan, "Getting to know low-light images with The ExDark dataset," <em>Computer Vision and Image Understanding</em>, vol. 178, pp. 30-42, 2019.</div>
    <div class="reference-item">[3] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," <em>Graphics Gems IV</em>, Academic Press, pp. 474-485, 1994.</div>
    <div class="reference-item">[4] D. Guan, Y. Cao, J. Yang, C. Shen, and M. G. Trefzer, "Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection," <em>IEEE Transactions on Information Forensics and Security</em>, vol. 14, no. 11, pp. 3048-3059, 2019.</div>
    <div class="reference-item">[5] H. Ye et al., "Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge," <em>arXiv preprint arXiv:2604.19054</em>, 2026.</div>

</div>

</body>
</html>
"""
    temp_html_path = Path("docs/proposal/temp_technical_report.html")
    temp_html_path.write_text(html_content, encoding="utf-8")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.goto(f"file:///{temp_html_path.resolve().as_posix()}", wait_until="networkidle")
        page.wait_for_timeout(1000)

        header_html = """
        <div style="font-size: 7.5pt; font-family: 'Times New Roman', serif; width: 100%; margin: 0 14mm; display: flex; justify-content: space-between; color: #475569; border-bottom: 0.5pt solid #cbd5e1; padding-bottom: 2px;">
            <span>Computer Vision (IF0PP006) — Technical Report Proposal (v02)</span>
            <span>Semester Ganjil 2026/2027</span>
        </div>
        """

        footer_html = """
        <div style="font-size: 7.5pt; font-family: 'Times New Roman', serif; width: 100%; margin: 0 14mm; display: flex; justify-content: space-between; color: #475569; border-top: 0.5pt solid #cbd5e1; padding-top: 2px;">
            <span>Kelompok 9: Deteksi Pedestrian Low-Light</span>
            <span>Halaman <span class="pageNumber"></span> dari <span class="totalPages"></span></span>
        </div>
        """

        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template=header_html,
            footer_template=footer_html,
            margin={
                "top": "18mm",
                "bottom": "18mm",
                "left": "14mm",
                "right": "14mm",
            },
        )

        OUTPUT_PDF_PROPOSAL.write_bytes(pdf_bytes)
        ROOT_PDF_PROPOSAL.write_bytes(pdf_bytes)
        browser.close()

    if temp_html_path.exists():
        temp_html_path.unlink()

    print(f"[OK] PDF Proposal generated:\n  - {OUTPUT_PDF_PROPOSAL}\n  - {ROOT_PDF_PROPOSAL}")


def sync_obsidian() -> None:
    """Syncs obsidian note summary."""
    obsidian_text = """---
date: 2026-10-03
project: Deteksi Pedestrian Low-Light
tags: [proposal, technical-report, uts, computer-vision, yolov8, clahe]
---

# 📑 Proposal Teknis UTS (IEEE Technical Report Format)

> **Dokumen Resmi:** `K09_DeteksiPedestrianLowLight_Proposal_v02`  
> **Format:** IEEE 2-Column Technical Report (`Technical_Report_Template.docx`)  
> **Mata Kuliah:** Visi Komputer (IF0PP006) — Semester Ganjil 2026/2027  
> **Kelompok:** 9 (Krisantus Linardi & Kristofer Leonard)  
> **Tautan File Repositori:** [[K09_DeteksiPedestrianLowLight_Proposal_v02.md]]

---

## Ringkasan Pembaruan Format Proposal UTS
Proposal UTS telah diperbarui secara penuh mengikuti **Technical Report Template** resmi 2-kolom IEEE:
1. **Title & Subtitle:** Terjemahan bilinguistik Inggris & Indonesia.
2. **Authors Table:** Identitas kelompok 9 dan pembagian peran (Krisantus: Data & Preprocessing Lead; Kristofer: Model & Deployment Lead).
3. **Abstract / Abstrak:** Ringkasan 200-250 kata dalam Bahasa Inggris dan Bahasa Indonesia.
4. **1. Introduction / Pendahuluan:** Permasalahan keselamatan jalan malam hari, kegagalan filter konvensional vs deep learning.
5. **2. Methodology / Metodologi:**
   - **2.1 A. Data Strategy:** Dataset Roboflow Low-light v2 (2.400 citra, 7.777 box), audit imbalance 7,58:1, Prapemrosesan L-channel CLAHE & Gamma Correction, pencegahan leakage.
   - **2.2 B. Model Architecture:** Arsitektur YOLOv8-Nano (~3,2M param), C2f module, loss function CIoU + DFL + BCE, hyperparameter terstandarisasi.
   - **2.3 C. Deployment & Environment:** Spesifikasi GPU Colab NVIDIA T4 (training) & CPU Laptop Intel i5/i7 (inferensi), target throughput $\\ge 20$–$25$ FPS.
6. **3. Results and Discussion:** Desain 4 eksperimen terkontrol (B0, Exp-A, Exp-B, Exp-C), Tabel I audit dataset, Tabel II kerangka eksperimen, visualisasi anotasi audit, dan rencana failure case analysis.
7. **4. Conclusion / Kesimpulan:** Target peningkatan Recall pejalan kaki $>10\\%$, keterbatasan, dan rencana kuantisasi INT8 edge.
8. **References / Referensi:** 5 rujukan standar IEEE citation [1]-[5].

---
*(Dokumen ini terintegrasi penuh dengan repositori di `K09_DeteksiPedestrianLowLight_Proposal_v02.docx` dan `K09_DeteksiPedestrianLowLight_Proposal_v02.pdf`)*
"""
    OBSIDIAN_NOTE.write_text(obsidian_text, encoding="utf-8")
    print(f"[OK] Obsidian sync note written to:\n  - {OBSIDIAN_NOTE}")


def main() -> None:
    build_docx_report()
    build_md_report()
    build_pdf_report()
    sync_obsidian()


if __name__ == "__main__":
    main()
