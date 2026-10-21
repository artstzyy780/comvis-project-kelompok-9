# Peta Bukti Checkpoint (CP01–CP13) → Perintah & Berkas Hasil

> Dokumen ini adalah "peta bukti" untuk mengisi kolom **Link bukti** pada
> `Kelompok_09_Tracker_Proyek_Akhir.xlsm`. Setiap baris memuat perintah yang
> harus dijalankan, berkas keluaran yang dihasilkan, dan checkpoint yang dilayani.

## Peta Ringkas

| CP | XP | Perintah | Berkas Bukti | Status Kode |
|:--:|:--:|:---|:---|:---:|
| CP00 | 3 | (manual) kontrak kerja 4 anggota | `docs/proposal/CONTRIBUTING.md` | 🔴 perlu dibuat |
| CP01 | 4 | `python src/build_candidate_matrix.py` | `results/proposal/candidate_matrix.md` | ✅ |
| CP02 | 6 | `python src/collect_references.py` | `docs/proposal/references.md`, `references.bib` | ✅ |
| CP03 | 6 | `python src/audit_dataset.py` | `results/audit/audit_report.md` | ✅ (versi baru: bias + leakage + rencana split) |
| CP04 | 6 | `python src/evaluate_baseline.py --split valid` | `results/experiments/baseline_b0_report.md` | ✅ |
| CP05 | 7 | `python src/render_pipeline_diagram.py` | `results/proposal/pipeline_diagram.png`, `experiment_design.md` | ✅ |
| CP06 | 8 | `python src/freeze_artifacts.py` | `results/freeze/frozen_artifacts.md` | ✅ |
| CP07 | 7 | `python src/rebuild_split_dedup.py` | `results/audit/dedup_split_report.md`, `dataset_dedup/split_manifest.json` | ✅ (baru) |
| CP08 | 8 | `python src/train.py --experiment Exp_A_Standard` | `results/experiments/runs/Exp_A_Standard/run_manifest.json` | ✅ (baru) |
| CP09 | 9 | `python src/train.py --all` | `results/experiments/runs/*/run_manifest.json` | ✅ (baru) |
| CP10 | 10 | `python src/evaluate_experiments.py --split test` | `results/experiments/unified_results.md` | ✅ (baru) |
| CP11 | 10 | `python src/analyze_failures.py --weights <best.pt>` | `results/experiments/failure_cases.md`, `failure_cases.png` | ✅ (baru) |
| CP12 | 7 | `python src/benchmark_inference.py` + `python app.py` | `results/experiments/benchmark_latency_report.md` | ✅ / 🔴 demo app belum ada |
| CP13 | 9 | `python src/freeze_artifacts.py --check` | `results/freeze/artifact_manifest.json` | ✅ |

---

## 1. Pemeriksaan Awal (Wajib Dijalankan Setiap Kali)

```bash
pip install -r requirements.txt
python -m pytest tests -q            # 32 unit test harus lolos
```

## 2. Rantai Audit & Data (CP03, CP07)

```bash
python src/audit_dataset.py           # audit + bias + deteksi leakage
python src/rebuild_split_dedup.py     # bangun split bebas-leakage (scene level)
```

**Temuan kunci yang wajib dibahas di sidang:**

| Split | File | Scene asal yang bocor ke split lain |
|:---|---:|---:|
| Train | 2.100 | 468 (22,3%) |
| Valid | 200 | 104 (52,0%) |
| Test | 100 | 55 (**55,0%**) |

Dari 39 kotak `person` pada test set, **28 kotak (71,8%)** berasal dari scene
yang juga ada di train. Karena itu seluruh eksperimen final **wajib**
menggunakan `dataset_dedup/`, bukan `dataset/`.

## 3. Eksperimen Terkontrol (CP08, CP09)

Di Google Colab (GPU T4):

```bash
pip install ultralytics
python src/train.py --experiment Exp_A_Standard
python src/train.py --experiment Exp_B_CLAHE
python src/train.py --experiment Exp_C_Gamma
# atau sekali jalan:
python src/train.py --all
```

Setiap run menyimpan `run_manifest.json` berisi hyper-parameter, seed, versi
library, commit Git, durasi, dan metrik **per-kelas** (bukan macro-average).

## 4. Evaluasi Terpadu (CP10)

```bash
python src/evaluate_experiments.py --split test --dataset-root dataset_dedup
```

Menghasilkan **satu tabel** untuk seluruh model pada **satu split**, dilengkapi
interval keyakinan 95% (Wilson) dan verdict statistik per perubahan.

## 5. Analisis Kasus Gagal (CP11)

```bash
python src/analyze_failures.py \
  --weights hasil_final_model/hasil_final_model/Exp_A_Standard_best.pt \
  --dataset-root dataset_dedup --split test
```

Menghasilkan kategorisasi error (kecil/terhalang/kontras rendah/keyakinan rendah),
grid visual, serta hipotesis dan satu perbaikan yang diuji.

## 6. Beku Artefak (CP06, CP13)

```bash
python src/freeze_artifacts.py        # rekam sidik jari SHA-256
python src/freeze_artifacts.py --check # deteksi perubahan setelah beku
```

## 7. Dokumentasi Proposal (CP01, CP02, CP05)

```bash
python src/build_candidate_matrix.py
python src/collect_references.py
python src/render_pipeline_diagram.py
```
