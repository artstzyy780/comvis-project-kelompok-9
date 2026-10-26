# 📊 Unified Evaluation Results (Single Split, Per-Class Metrics)

- **Split evaluasi:** `test` (dataset `dataset_dedup`) — *satu split untuk seluruh model*
- **Confidence threshold:** `0.25` | **IoU matching:** `0.5`
- **Metrik:** per-kelas (bukan macro-average), dilengkapi CI 95% (Wilson).

## 1. Tabel Hasil Terpadu

| Model | Recall person (CI 95%) | Precision person | F1 person | Recall vehicle | Macro F1 | mAP@50 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0_ZeroShot** | **15.19%** [8.9, 24.7] | 8.28% | 10.71% | 40.14% | 28.98% | 0.03% |
| **Exp_A_Standard** | **81.01%** [71.0, 88.1] | 62.75% | 70.72% | 91.55% | 78.41% | 83.93% |
| **Exp_B_CLAHE** | **40.51%** [30.4, 51.5] | 64.00% | 49.61% | 80.48% | 64.25% | 66.38% |
| **Exp_C_Gamma** | **65.82%** [54.8, 75.3] | 33.33% | 44.26% | 88.83% | 61.58% | 67.90% |
| **Exp_D_Robust** | **72.15%** [61.4, 80.8] | 46.72% | 56.72% | 87.63% | 69.13% | 76.20% |

## 2. Catatan Perubahan Satu per Satu

Kontrol grup = **Exp_A_Standard** (fine-tuning tanpa augmentasi pencahayaan).
Setiap baris menguji **satu variabel bebas** terhadap kontrol tersebut.

| Eksperimen | Variabel bebas yang diuji | Recall person (CI 95%) | Δ vs kontrol | Verdict statistik |
|:---|:---|:---:|:---:|:---|
| **B0_ZeroShot** | Tanpa fine-tuning (acuan mutlak). | 15.19% [8.9, 24.7] | -65.82 pp | menurun secara signifikan |
| **Exp_B_CLAHE** | Tambahan: CLAHE L-channel pada data latih. | 40.51% [30.4, 51.5] | -40.50 pp | menurun secara signifikan |
| **Exp_C_Gamma** | Tambahan: Gamma Correction (gamma=0.6) pada data latih. | 65.82% [54.8, 75.3] | -15.19 pp | tidak signifikan (CI beririsan) |
| **Exp_D_Robust** | Tambahan: Backbone freezing (freeze=10), lr0=0.001, scale=0.5, mosaic=1.0. | 72.15% [61.4, 80.8] | -8.86 pp | tidak signifikan (CI beririsan) |

## 3. Catatan Metodologis

1. **Per-kelas, bukan macro.** Recall perolehan diambil dari array per-kelas (`results.box.r`) sehingga tidak lagi tercampur dengan rata-rata antar kelas.
2. **Interval keyakinan 95%.** Karena kelas `person` jarang (puluhan box), selisih beberapa poin persentase belum tentu signifikan secara statistik.
3. **Split tunggal.** Seluruh model dievaluasi pada split yang sama agar perbandingan adil (syarat CP10).
4. **Baseline B0 memakai taksonomi COCO** yang dipetakan ke 2 kelas, sehingga perbandingan mAP dengan model fine-tuned tidak sepenuhnya setara dan hanya disajikan sebagai acuan kasar.
