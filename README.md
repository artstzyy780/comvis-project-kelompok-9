# 🌙 Low-Light Pedestrian Detection Using YOLOv8 & Adaptive Illumination Enhancement

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-FF6F00.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Computer Vision Final Project (IF0PP006) — Semester Ganjil 2026/2027**  
> **Kelompok 9:** Krisantus Amka Ginting · Linardi Halim · Kristofer Ryan Giggs Eka Saputra · Leonard Franscois

---

## 📌 1. Latar Belakang & Motivasi

Deteksi pejalan kaki di malam hari (*nighttime pedestrian detection*) adalah tantangan krusial dalam Sistem Transportasi Cerdas (ITS) dan *Advanced Driver Assistance Systems* (ADAS). Sebagian besar kecelakaan fatal pejalan kaki terjadi dalam kondisi pencahayaan rendah (*low-light*), di mana detektor berbasis deep learning standar (yang dominan dilatih pada citra siang hari seperti MS COCO) mengalami penurunan performa drastis akibat kontras rendah, derau sensor (*sensor grain*), dan silau lampu kendaraan (*glare*).

Proyek ini menghadirkan pipeline deteksi objek *real-time* berbasis **Ultralytics YOLOv8-Nano** yang dipadukan dengan teknik prapemrosesan pencahayaan adaptif:
1. **L-Channel CLAHE (CIE $L^*a^*b^*$)**: Meningkatkan kontras lokal siluet tanpa merusak fidelitas krominansi.
2. **Non-linear Gamma Correction ($\gamma = 0.6$)**: Memetakan fungsi daya non-linear melalui Lookup Table (LUT) untuk mencerahkan area bayangan gelap secara global.

---

## 📊 2. Dataset & Protokol Anti-Leakage

* **Dataset:** Roboflow Low-light v2 (`object-detection-in-ll/low-light-sxzx2`)
* **Total Citra:** 2.400 citra jalan raya beresolusi seragam $640 \times 640$ piksel.
* **Distribusi Split:** 
  * **Train:** 2.100 citra (6.816 kotak)
  * **Validation:** 200 citra (650 kotak)
  * **Test:** 100 citra (311 kotak)
* **Ketidakseimbangan Kelas (Imbalance):**
  * Kendaraan (`Vehicle`): 6.871 anotasi (88,35%)
  * Pejalan Kaki (`person`): 906 anotasi (11,65%)
  * **Rasio: 7,58 : 1**
* **Protokol Anti-Leakage:** Seluruh transformasi pencahayaan adaptif (CLAHE dan Gamma) **hanya diterapkan pada data latih (Train)**. Data validasi dan data uji dibiarkan 100% murni citra malam mentah untuk menjamin objektivitas evaluasi ilmiah.

### ⚠️ Temuan Audit Lanjutan: Kebocoran pada Level *Source Scene* (CP03/CP07)

Audit lanjutan terhadap penamaan berkas Roboflow (`*_augmented_N_*`) menemukan bahwa **eksport bawaan bukan benar-benar independen pada level scene**:

| Split | Total file | File yang scene asalnya juga ada di split lain |
|:---|---:|---:|
| Train | 2.100 | 468 (22,3%) |
| Valid | 200 | 104 (52,0%) |
| **Test** | **100** | **55 (55,0%)** |

Dari 39 kotak `person` pada test set, **28 kotak (71,8%)** berasal dari scene yang juga ada di train. Klaim *"strictly isolated test set"* karena itu **tidak akurat** pada level scene.

**Perbaikan yang dilakukan** (`src/rebuild_split_dedup.py`): seluruh augmented copy dari satu scene dipaksa berada pada **split yang sama**, dengan stratifikasi berdasarkan keberadaan pejalan kaki dan seed tetap (42).

| Split (bebas-leakage) | Citra | Scene | Box `person` |
|:---|---:|---:|---:|
| Train | 1.643 | 481 | 649 |
| Valid | 384 | 103 | 178 |
| Test | 373 | 103 | **79** |

Split baru ini **0 scene bocor** dan test set menjadi **2× lebih besar** (39 → 79 kotak pejalan kaki), sehingga interval keyakinanRecall jauh lebih sempit.

---

## 🏆 3. Hasil Eksperimen & Benchmark

### A. Evaluasi Akurasi (Validation Set - 50 Epochs, GPU NVIDIA Tesla T4)

| ID Eksperimen | Skenario Eksperimen | Recall Pejalan Kaki (IoU=0.5) | mAP@50 (All Classes) | $\Delta$ Recall vs B0 |
|:---:|---|:---:|:---:|:---:|
| **B0** | Zero-Shot Baseline (COCO Pretrained) | **39.10%** | **38.00%** | — |
| **Exp-A** | Fine-Tuned YOLOv8n Standar | **73.28%** | **78.12%** | **+34.18%** |
| **Exp-B** | Proposed: Fine-Tuned + L-channel CLAHE | **69.27%** | **77.73%** | **+30.17%** |
| **Exp-C** | Proposed: Fine-Tuned + Gamma Correction ($\gamma=0.6$) | **73.41%** | **74.17%** | **+34.31%** (Juara 1 Recall) |

### B. Benchmark Latensi & Throughput CPU Lokal (Test Set - 100 Citra Malam Murni)

*Host: AMD Ryzen 6-Core / 12-Threads @ 2.3 GHz, RAM 15.4 GB, `device='cpu'`, Input Size: $640 \times 640$, Batch Size = 1.*

| ID Model | Pedestrian Recall | Pedestrian Precision | Pedestrian F1 | Latensi Rata-rata | Throughput (FPS) | Status Real-Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0_ZeroShot** | 20.51% | 15.09% | 17.39% | 53.49 ms | 18.69 FPS | Near Real-Time |
| **Exp_A_Standard** | **66.67%** | 53.06% | **59.09%** | **52.04 ms** | **19.21 FPS** | Near Real-Time |
| **Exp_B_CLAHE** | 48.72% | **67.86%** | 56.72% | 54.67 ms | 18.29 FPS | Near Real-Time |
| **Exp_C_Gamma** | **64.10%** | 45.45% | 53.19% | 54.25 ms | 18.43 FPS | Near Real-Time |

---

## 📁 4. Struktur Repositori

```text
├── dataset/                   # Dataset Low-light v2 (data.yaml, train, valid, test)
├── dataset_dedup/             # Split bebas-leakage hasil rebuild (dihasilkan, di-gitignore)
├── configs/                   # Konfigurasi terpusat (eksperimen, evaluasi, ukuran sukses)
│   ├── experiments.yaml       # Registry 4 eksperimen terkontrol + hyperparameter
│   ├── evaluation.yaml        # Threshold evaluasi (conf, IoU, target FPS)
│   └── success_criteria.yaml  # Ukuran sukses proyek (CP04)
├── docs/                      # Dokumen proposal, peta bukti checkpoint, naskah
│   ├── CP_EVIDENCE_MAP.md     # Peta CP01–CP13 → perintah & berkas bukti
│   └── proposal/              # Proposal UTS, slide, naskah, kontrak kerja
├── hasil_final_model/         # Bobot model terbaik (.pt) hasil fine-tuning Google Colab
│   └── hasil_final_model/
│       ├── Exp_A_Standard_best.pt
│       ├── Exp_B_CLAHE_best.pt
│       ├── Exp_C_Gamma_best.pt
│       └── komparasi_recall_pedestrian.png
├── notebooks/                 # Notebook Google Colab untuk pelatihan GPU T4
│   └── K09_DeteksiPedestrian_Colab_Training.ipynb
├── results/                   # Laporan metrik kuantitatif dan visualisasi
│   ├── audit/                 # Audit dataset, bias, leakage, laporan split bebas-leakage
│   ├── experiments/           # Benchmark latensi, unified results, run manifest
│   ├── freeze/                # Manifest beku artefak (SHA-256)
│   ├── preprocessing/         # Komparasi visual Original vs CLAHE vs Gamma
│   └── proposal/              # Matriks kandidat, diagram pipeline, desain eksperimen
├── src/                       # Modul Python modular dan bertipe data ketat
│   ├── config.py              # Konfigurasi terpusat + registry eksperimen
│   ├── scene_utils.py         # Grouping scene, dHash, deteksi kebocoran, split grouped
│   ├── metrics.py             # Metrik per-kelas, IoU matching, Wilson/bootstrap CI
│   ├── audit_dataset.py       # Audit dataset, bias, dan deteksi leakage
│   ├── rebuild_split_dedup.py # Bangun split bebas-leakage (grouped by scene)
│   ├── train.py               # Entry point pelatihan + run manifest
│   ├── evaluate_experiments.py# Tabel hasil unified pada satu split (CP10)
│   ├── evaluate_baseline.py   # Evaluasi zero-shot Baseline B0
│   ├── benchmark_inference.py # Benchmark latensi, FPS, dan akurasi CPU
│   ├── prepare_experiments.py # Dataset generator CLAHE & Gamma (anti-leakage)
│   ├── preprocessing.py       # Fungsi transformasi CLAHE LAB & Gamma LUT
│   ├── freeze_artifacts.py    # Beku artefak + deteksi perubahan
│   └── build_candidate_matrix.py, collect_references.py, render_pipeline_diagram.py
├── tests/                     # Unit test pytest (metrics, scene utils)
├── GEMINI.md                  # Panduan teknis & standar clean code proyek
├── README.md                  # Dokumentasi utama proyek
└── .gitignore                 # Konfigurasi pengabaian berkas biner/besar
```

---

## 🚀 5. Cara Menjalankan Proyek (Getting Started)

### 1. Kloning Repositori & Persiapan Lingkungan

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt  # atau install ultralytics, opencv-python, psutil, torch
```

### 2. Menjalankan Audit Dataset

Untuk memverifikasi integritas file citra dan menghitung distribusi anotasi:

```bash
python src/audit_dataset.py
```

### 3. Membangun Split Bebas-Leakage (CP07)

Eksport Roboflow menempatkan augmented copy dari satu scene ke beberapa split. Perintah berikut membangun ulang split sehingga tidak ada scene yang muncul di dua split:

```bash
python src/rebuild_split_dedup.py            # tulis dataset_dedup/
python src/rebuild_split_dedup.py --dry-run  # pratinjau tanpa menulis
```

### 4. Menjalankan Benchmarking Latensi & Throughput CPU

Untuk menguji kecepatan inferensi per frame dan akurasi deteksi seluruh model:

```bash
python src/benchmark_inference.py
```

Laporan lengkap akan otomatis dihasilkan pada `results/experiments/benchmark_latency_report.md`.

### 5. Melatih Model (Google Colab / GPU T4)

Setiap run menyimpan manifest berisi hyper-parameter, seed, versi library, commit Git, durasi, dan metrik **per-kelas**:

```bash
python src/train.py --experiment Exp_A_Standard
python src/train.py --all
```

### 6. Evaluasi Terpadu pada Satu Split (CP10)

Menghasilkan satu tabel untuk seluruh model pada split yang sama, dilengkapi interval keyakinan 95% (Wilson) dan verdict statistik per variabel bebas:

```bash
python src/evaluate_experiments.py --split test --dataset-root dataset_dedup
```

### 7. Beku Artefak & Deteksi Perubahan (CP06/CP13)

```bash
python src/freeze_artifacts.py         # rekam sidik jari SHA-256
python src/freeze_artifacts.py --check # deteksi perubahan setelah pembekuan
```

### 8. Unit Test

```bash
python -m pytest tests -q
```

### 9. Daftar Perintah → Checkpoint

Peta lengkap perintah dan berkas bukti untuk tiap checkpoint ada di
[`docs/CP_EVIDENCE_MAP.md`](docs/CP_EVIDENCE_MAP.md). Ringkasnya:

| CP | Perintah utama |
|:--|:---|
| CP01 | `python src/build_candidate_matrix.py` |
| CP02 | `python src/collect_references.py` |
| CP03 | `python src/audit_dataset.py` |
| CP05 | `python src/render_pipeline_diagram.py` |
| CP07 | `python src/rebuild_split_dedup.py` |
| CP08/CP09 | `python src/train.py --all` |
| CP10 | `python src/evaluate_experiments.py --split test` |
| CP12 | `python src/benchmark_inference.py` |
| CP13 | `python src/freeze_artifacts.py --check` |

---
