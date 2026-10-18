# 🌙 Low-Light Pedestrian Detection Using YOLOv8 & Adaptive Illumination Enhancement

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-FF6F00.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Computer Vision Final Project (IF0PP006) — Semester Ganjil 2026/2027**  
> **Kelompok 9:** Krisantus Linardi & Kristofer Leonard

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
├── docs/                      # Dokumen proposal, slide presentasi sidang UTS, dan naskah
├── hasil_final_model/         # Bobot model terbaik (.pt) hasil fine-tuning Google Colab
│   └── hasil_final_model/
│       ├── Exp_A_Standard_best.pt
│       ├── Exp_B_CLAHE_best.pt
│       ├── Exp_C_Gamma_best.pt
│       └── komparasi_recall_pedestrian.png
├── notebooks/                 # Notebook Google Colab untuk pelatihan GPU T4
│   └── K09_DeteksiPedestrian_Colab_Training.ipynb
├── results/                   # Laporan metrik kuantitatif dan visualisasi
│   ├── audit/                 # Laporan audit dataset & sampel anotasi
│   ├── experiments/           # Benchmark latensi JSON & Markdown reports
│   └── preprocessing/         # Komparasi visual Original vs CLAHE vs Gamma
├── src/                       # Modul Python modular dan bertipe data ketat
│   ├── audit_dataset.py       # Skrip audit distribusi dan integritas dataset
│   ├── benchmark_inference.py # Skrip benchmark latensi, FPS, dan akurasi CPU
│   ├── evaluate_baseline.py   # Skrip evaluasi zero-shot Baseline B0
│   ├── prepare_experiments.py # Dataset generator CLAHE & Gamma (anti-leakage)
│   └── preprocessing.py       # Fungsi transformasi CLAHE LAB & Gamma LUT
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

### 3. Menjalankan Benchmarking Latensi & Throughput CPU

Untuk menguji kecepatan inferensi per frame dan akurasi deteksi seluruh model:

```bash
python src/benchmark_inference.py
```

Laporan lengkap akan otomatis dihasilkan pada `results/experiments/benchmark_latency_report.md`.

---
