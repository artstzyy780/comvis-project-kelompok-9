# Rekonsiliasi Metrik — Satu Sumber Kebenaran

> Dihasilkan otomatis oleh `python src/reconcile_metrics.py`.
> Modul ini **tidak** menjalankan model; ia hanya membaca artefak JSON yang sudah ada.

## 1. Masalah yang diperbaiki

- `metrics/recall(B)` dari Ultralytics adalah **rata-rata recall semua kelas** (macro),
  bukan recall kelas `person`.
- Tabel hasil sebelumnya mencampur `B0` (person recall) dengan `Exp-A/B/C` (macro recall),
  sehingga selisih seperti `+34,18%` tidak apple-to-apple.
- `src/metrics.py::per_class_metrics_from_ultralytics` sudah menulis peringatan ini;
  dokumen ini memakai angka per-kelas yang memang tersedia.

## 2. Tabel utama — Test split, metrik per-kelas `person`

Sumber: `benchmark_latency_metrics.json` (conf 0.25; IoU 0.5; split test).

| Model | Skenario | n citra | GT `person` | Recall `person` | Precision `person` | F1 `person` | Delta Recall vs B0 |
|:---:|:---|---:|---:|---:|---:|---:|---:|
| **B0_ZeroShot** | B0: Zero-Shot Baseline (COCO Pretrained) | 100 | 39 | **20.51%** | 15.09% | 17.39% | - |
| **Exp_A_Standard** | Exp-A: Fine-Tuned Standard Low-Light | 100 | 39 | **66.67%** | 53.06% | 59.09% | +46.16 pp |
| **Exp_B_CLAHE** | Exp-B: Proposed Fine-Tuned + L-channel CLAHE | 100 | 39 | **48.72%** | 67.86% | 56.72% | +28.21 pp |
| **Exp_C_Gamma** | Exp-C: Proposed Fine-Tuned + Gamma Correction | 100 | 39 | **64.10%** | 45.45% | 53.19% | +43.59 pp |

**Bacaan:** fine-tuning adalah pendorong utama kenaikan recall.
Gamma (Exp-C) hampir setara dengan fine-tuning standar (Exp-A), sedangkan CLAHE (Exp-B)
pada sampel test ini lebih rendah meski precision-nya lebih tinggi.
Selisih antar-varian belum diuji signifikansinya (lihat Keterbatasan).

## 3. Tabel pembanding — Validation split (macro vs per-kelas)

| Model | Macro Recall (semua kelas) | mAP@50 (semua kelas) | Recall `person` | Sumber metrik |
|:---:|---:|---:|---:|:---|
| **B0** | 42.28%* | - | 39.13% | evaluate_baseline.py (valid, conf 0.25) |
| **Exp-A** | 73.28% | 78.12% | - | metrics/recall(B) (valid, max-F1) |
| **Exp-B** | 69.27% | 77.73% | - | metrics/recall(B) (valid, max-F1) |
| **Exp-C** | 73.41% | 74.17% | - | metrics/recall(B) (valid, max-F1) |

> \*B0 dihitung pada conf 0.25 oleh `evaluate_baseline.py`, sedangkan
> Exp-A/B/C memakai operating point max-F1. Baris B0 hanya konteks,
> bukan pembanding langsung.

> **Peringatan:** kolom `Exp-A/B/C` di tabel ini adalah `metrics/recall(B)`
> (macro, operating point max-F1), bukan recall `person` pada conf 0.25.
> Jangan membandingkannya langsung dengan Tabel 2. Recall `person` validasi
> untuk Exp-A/B/C harus dihitung ulang dengan
> `src/metrics.py::per_class_metrics_from_ultralytics` atau `src/evaluate_experiments.py`.

## 4. Latensi & throughput — Test split (CPU)

Target proyek: >= 20 FPS.

| Model | Rata-rata | P50 | P95 | FPS | Memenuhi target |
|:---:|---:|---:|---:|---:|:---:|
| **B0_ZeroShot** | 53.49 ms | 51.54 ms | 62.37 ms | 18.69 | Tidak |
| **Exp_A_Standard** | 52.04 ms | 49.67 ms | 58.59 ms | 19.21 | Tidak |
| **Exp_B_CLAHE** | 54.67 ms | 52.26 ms | 66.83 ms | 18.29 | Tidak |
| **Exp_C_Gamma** | 54.25 ms | 52.40 ms | 60.68 ms | 18.43 | Tidak |

> Kolom `preprocess_mean_ms` pada laporan benchmark adalah preprocessing internal
> Ultralytics (resize/normalize), **bukan** CLAHE/Gamma. Biaya CLAHE/Gamma
> end-to-end belum diukur.

## 5. Keterbatasan yang wajib disebut

1. **Split test masih bocor level scene.** Audit menemukan 55% file test berasal dari
   scene yang juga ada di train; split `dataset_dedup/` sudah dibangun tetapi belum
   dipakai untuk melatih atau mengevaluasi model final.
2. **Sampel kecil.** Tabel test hanya berisi 39 box `person` (18 citra), dan satu seed (42),
   sehingga selisih beberapa poin persen belum tentu signifikan. Interval Wilson/bootstrap
   sudah tersedia di `src/metrics.py` tetapi belum dilampirkan pada tabel hasil.
3. **Operating point berbeda.** Benchmark memakai conf 0.25; Ultralytics validation
   melaporkan max-F1. Keduanya tidak boleh dicampur.
4. **Tidak ada bukti run pelatihan.** `results/experiments/runs/*/run_manifest.json`
   dan `results/experiments/unified_results.md` belum ada.
5. **Freeze artefak perlu diperbarui.** README dan laporan benchmark diedit setelah
   pembekuan 2026-10-20, sehingga `python src/freeze_artifacts.py --check` akan
   melaporkan digest berbeda. Jalankan ulang freeze setelah laporan final selesai.

## 6. Cara menghitung ulang secara konsisten

```bash
# Evaluasi per-kelas pada satu split, dengan interval keyakinan:
python src/evaluate_experiments.py --split test --dataset-root dataset_dedup

# Benchmark latensi + metrik per-kelas:
python src/benchmark_inference.py
```

Setelah `unified_results.json` tersedia, jalankan ulang modul ini agar Tabel 3
memakai recall `person` yang sebenarnya.
