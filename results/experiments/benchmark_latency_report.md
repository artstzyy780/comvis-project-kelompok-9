# ⏱️ Comprehensive Local Hardware Inference & Throughput Benchmark Report

- **Mata Kuliah:** Visi Komputer (IF0PP006) — Semester Ganjil 2026/2027
- **Kelompok:** 9 (Krisantus Linardi & Kristofer Leonard)
- **Topik:** Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif
- **Tanggal Pengujian:** 18 Oktober 2026

---

## 1. Spesifikasi Lingkungan Pengujian (Host Machine)

- **Operating System:** `Windows 11 (10.0.26200)`
- **CPU:** `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD` (6 Physical Cores / 12 Threads @ 2301.0 MHz)
- **RAM Terpasang:** `15.4 GB`
- **Python Version:** `3.14.6` | **Device:** `CPU` (Batch Size = 1, Resolution = 640x640)

---

## 2. Tabel Komparasi Throughput & Latensi Inferensi CPU

| ID Model | Nama Varian | Rata-rata Latensi | P50 (Median) | P95 (Tail) | Throughput (FPS) | Status Real-Time (>= 20 FPS) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **B0_ZeroShot** | B0: Zero-Shot Baseline (COCO Pretrained) | **53.49 ms** | 51.54 ms | 62.37 ms | **18.69 FPS** | ⚠️ *BORDERLINE / SUB-REALTIME* |
| **Exp_A_Standard** | Exp-A: Fine-Tuned Standard Low-Light | **52.04 ms** | 49.67 ms | 58.59 ms | **19.21 FPS** | ⚠️ *BORDERLINE / SUB-REALTIME* |
| **Exp_B_CLAHE** | Exp-B: Proposed Fine-Tuned + L-channel CLAHE | **54.67 ms** | 52.26 ms | 66.83 ms | **18.29 FPS** | ⚠️ *BORDERLINE / SUB-REALTIME* |
| **Exp_C_Gamma** | Exp-C: Proposed Fine-Tuned + Gamma Correction | **54.25 ms** | 52.40 ms | 60.68 ms | **18.43 FPS** | ⚠️ *BORDERLINE / SUB-REALTIME* |

### Rincian Latensi per Tahapan Pipeline Ultralytics (ms)

> Kolom *Preprocessing* di bawah adalah preprocessing internal Ultralytics
> (resize/normalize), **bukan** CLAHE/Gamma. Biaya enhancement end-to-end
> belum diukur oleh benchmark ini.

| ID Model | Preprocessing (internal) | Model Forward (Inference) | Postprocessing (NMS) | Total Mean |
|:---:|:---:|:---:|:---:|:---:|
| **B0_ZeroShot** | 1.94 ms | 50.07 ms | 0.94 ms | **53.49 ms** |
| **Exp_A_Standard** | 2.71 ms | 47.94 ms | 0.86 ms | **52.04 ms** |
| **Exp_B_CLAHE** | 2.19 ms | 50.98 ms | 0.95 ms | **54.67 ms** |
| **Exp_C_Gamma** | 2.36 ms | 50.19 ms | 0.89 ms | **54.25 ms** |

---

## 3. Tabel Kinerja Deteksi Objek pada Test Set (100 Citra Malam)

| ID Model | Pedestrian Recall | Pedestrian Prec. | Pedestrian F1 | Vehicle Recall | Macro Recall | Macro F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0_ZeroShot** | **20.51%** | 15.09% | **17.39%** | 40.81% | 30.66% | 31.82% |
| **Exp_A_Standard** | **66.67%** | 53.06% | **59.09%** | 79.41% | 73.04% | 67.98% |
| **Exp_B_CLAHE** | **48.72%** | 67.86% | **56.72%** | 77.94% | 63.33% | 65.88% |
| **Exp_C_Gamma** | **64.10%** | 45.45% | **53.19%** | 83.82% | 73.96% | 64.34% |

---

## 4. Temuan Kritis & Analisis Kelayakan Sidang UAS

1. **Validasi Hipotesis Proposal**:
   - Fine-tuning (Exp-A) menghasilkan Recall `person` tertinggi (66.67%), diikuti Exp-C Gamma (64.10%). Exp-B CLAHE justru lebih rendah (48.72%). Semua varian tetap jauh di atas baseline B0 (20.51%).
   - Angka di tabel ini adalah metrik per-kelas pada conf 0.25, bukan macro recall Ultralytics. Lihat `METRICS_RECONCILIATION.md`.
2. **Kelayakan Real-Time pada Perangkat Edge / Komputer Konsumen**:
   - Latensi inferensi murni pada CPU berada pada kisaran **52-55 ms** (18.29-19.21 FPS). Semua varian **belum memenuhi** target 20 FPS.
   - Optimasi seperti OpenCV threading, ONNX Runtime, atau INT8 quantization belum diuji dalam proyek ini, sehingga klaim >20 FPS belum didukung bukti.
3. **Kesiapan Demonstrasi Langsung**:
   - Model berbobot ringan (~6.5 MB) dan latensi stabil, tetapi aplikasi demo (Streamlit/OpenCV) belum termasuk dalam repositori ini. Benchmark ini hanya mengukur model, bukan aplikasi end-to-end.

---
*(Laporan ini dihasilkan secara otomatis oleh `src/benchmark_inference.py`)*
