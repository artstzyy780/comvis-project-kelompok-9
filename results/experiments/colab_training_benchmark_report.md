# 📊 Laporan Hasil Eksperimen Pelatihan & Benchmarking Model (UAS Phase)

- **Mata Kuliah:** Visi Komputer (IF0PP006) — Semester Ganjil 2026/2027
- **Kelompok:** 9 (Krisantus Linardi & Kristofer Leonard)
- **Topik:** Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif
- **Platform Pelatihan:** Google Colab (GPU NVIDIA Tesla T4)
- **Parameter Terstandarisasi:** Epochs: 50 | Image Size: 640x640 | Batch: 16 | Seed: 42

> **Koreksi metrik:** tabel pada versi sebelumnya mencampur dua metrik berbeda.
> `metrics/recall(B)` dari Ultralytics adalah **macro recall semua kelas**, bukan
> recall kelas `person`. Angka B0 juga berasal dari metrik per-kelas pada split
> validasi. Detail lengkap: `results/experiments/METRICS_RECONCILIATION.md`.

---

## 1. Hasil Pelatihan pada Validation Split (Macro, Bukan Recall `person`)

Sumber: `metrics/recall(B)` dan `metrics/mAP50(B)` Ultralytics pada split validasi.

| ID Eksperimen | Skenario Eksperimen | Macro Recall (semua kelas) | mAP@50 (semua kelas) | Catatan |
|:---:|---|:---:|:---:|:---|
| **B0** | Zero-Shot Baseline (COCO Pretrained) | 42.28% | — | Macro dari `evaluate_baseline.py` (conf 0.25) |
| **Exp-A** | Fine-Tuned YOLOv8n Standar | **73.28%** | **78.12%** | `metrics/recall(B)`, max-F1 |
| **Exp-B** | Proposed: Fine-Tuned + L-channel CLAHE | 69.27% | 77.73% | `metrics/recall(B)`, max-F1 |
| **Exp-C** | Proposed: Fine-Tuned + Gamma ($\gamma=0.6$) | 73.41% | 74.17% | `metrics/recall(B)`, max-F1 |

> ⚠️ Kolom **Exp-A/B/C bukan recall pejalan kaki.** Nilai tersebut adalah rata-rata
> semua kelas pada operating point max-F1. Recall `person` validasi untuk ketiga
> model harus dihitung ulang dengan `src/evaluate_experiments.py` atau
> `src/metrics.py::per_class_metrics_from_ultralytics`.

## 2. Hasil Per-Kelas pada Test Split (Satu-Satunya Perbandingan Apple-to-Apples)

Sumber: `results/experiments/benchmark_latency_metrics.json` (conf 0.25, IoU 0.5,
100 citra test, 39 box `person`).

| ID Eksperimen | Recall `person` | Precision `person` | F1 `person` | $\Delta$ Recall vs B0 |
|:---:|---:|---:|---:|---:|
| **B0** | 20.51% | 15.09% | 17.39% | — |
| **Exp-A** | **66.67%** | 53.06% | **59.09%** | **+46.16 pp** |
| **Exp-B** | 48.72% | **67.86%** | 56.72% | +28.21 pp |
| **Exp-C** | 64.10% | 45.45% | 53.19% | +43.59 pp |

## 3. Temuan Utama & Analisis Ilmiah

1. **Pemenuhan Target Proposal**:
   - Target peningkatan Recall pejalan kaki minimal **>10 poin persen** terlampaui.
     Pada test split: Exp-A **+46.16 pp**, Exp-C **+43.59 pp**, Exp-B **+28.21 pp**
     terhadap B0 (20.51%).
2. **Pendorong utama adalah fine-tuning, bukan preprocessing**:
   - Exp-A (tanpa CLAHE/Gamma) mencapai Recall `person` tertinggi, yaitu **66.67%**.
   - Gamma (Exp-C) hampir setara (**64.10%**), sedangkan CLAHE (Exp-B) pada sampel
     test ini lebih rendah (**48.72%**) meskipun precision-nya paling tinggi (67.86%).
     Selisih antar-varian belum diuji signifikansinya (39 box `person`, satu seed).
3. **Implikasi untuk narasi proposal**:
   - Klaim lama bahwa "Exp-C adalah juara Recall" tidak lagi didukung setelah
     metrik dipisahkan per kelas. Kontribusi preprocessing perlu dilaporkan sebagai
     **temuan negatif/nuansa**, bukan sebagai kemenangan metode yang diusulkan.
4. **Keterbatasan yang wajib disebut**:
   - Split test masih bocor pada level *source scene* (55% file; 71,8% box `person`).
     Split bebas-leakage `dataset_dedup/` sudah dibangun tetapi belum dipakai untuk
     melatih atau mengevaluasi model final.
   - Sampel test hanya 39 box `person`, satu seed (42), dan belum ada interval
     keyakinan pada tabel ini.
   - Semua varian berada di bawah target **20 FPS** (18,29–19,21 FPS) pada CPU.

---

*(Laporan ini disusun sebagai pemenuhan luaran teknis UAS proyek Visi Komputer)*
