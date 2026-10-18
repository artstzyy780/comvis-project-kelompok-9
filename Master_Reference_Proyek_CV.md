# 📋 Master Reference — Proyek Visi Komputer Kelompok 6

> Dokumen ini merangkum seluruh keputusan, arsitektur, rubrik, dan poin penting dari diskusi proyek.
> **Terakhir diperbarui:** 29 September 2026

---

## 🪪 Identitas Proyek

| Info | Detail |
|---|---|
| **Mata Kuliah** | Visi Komputer (Computer Vision) |
| **Kelompok** | 6 |
| **Anggota** | Krisantus Linardi & Kristofer Leonard |
| **Topik Terpilih** | Deteksi Pejalan Kaki di Malam Hari / Minim Cahaya |
| **Judul Sementara** | *Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif* |
| **Nama File Pendek** | `DeteksiPedestrianLowLight` |

---

## 📁 Tiga Kandidat Topik

### Kandidat 1 — ✅ TERPILIH
**Deteksi Pejalan Kaki di Malam Hari (Night-time Pedestrian Detection)**

| Aspek | Detail |
|---|---|
| **Masalah** | Model deteksi standar (dilatih siang hari) jeblok di malam hari karena kontras rendah, sensor noise, dan silau lampu. Pejalan kaki tidak terdeteksi → potensi kecelakaan fatal. |
| **Tugas CV** | Object Detection |
| **Input** | Foto / frame video kamera jalanan malam hari (JPEG/PNG/MP4) |
| **Output** | Bounding box di sekeliling pejalan kaki + label "Pedestrian" + confidence score |

---

### Kandidat 2
**Segmentasi Kerusakan Bodi Mobil (Goresan & Penyok)**

| Aspek | Detail |
|---|---|
| **Masalah** | Penilaian kerusakan mobil untuk klaim asuransi masih manual, subjektif, dan lambat. |
| **Tugas CV** | Segmentasi Semantik (Semantic Segmentation) |
| **Input** | Foto bodi kendaraan rusak |
| **Output** | Mask piksel berwarna yang mengarsir area goresan/penyok + persentase luas kerusakan |

---

### Kandidat 3
**Pelacakan Multi-Objek Bawah Air (Multi-Object Tracking pada Kondisi Oklusi)**

| Aspek | Detail |
|---|---|
| **Masalah** | Sistem tracking sering keliru tukar identitas (ID-Switch) saat objek saling menutupi di air yang visibilitasnya buruk. |
| **Tugas CV** | Multi-Object Tracking (MOT) |
| **Input** | Klip video pendek rekaman bawah air |
| **Output** | ID unik konsisten per objek + garis trajektori + laporan jumlah ID-Switch |

---

## 🎯 Rubrik Penilaian (dari Panduan Dosen)

> Sumber: `Panduan_Proyek_Akhir_Visi_Komputer_UTS_UAS.pdf`

### Prinsip Utama Penilaian
> *"Proyek yang ruang lingkupnya sempit namun eksperimennya valid dan dapat dijelaskan jauh lebih bernilai daripada proyek model raksasa yang tidak punya baseline atau datanya bocor."*

### Checklist Wajib Per Proyek

| No | Komponen | Keterangan |
|:---:|---|---|
| ✅ | **Masalah & Batasan** | Input/output visual jelas, ruang lingkup terbatas |
| ✅ | **Dataset Legal** | Sumber publik, anotasi bersih, split train/val/test tidak bocor |
| ✅ | **Baseline** | Metode dasar pembanding yang realistis |
| ✅ | **Eksperimen Terkontrol** | Minimal 1 variasi perbandingan terukur |
| ✅ | **Evaluasi Kuantitatif** | Metrik standar + analisis kasus gagal |
| ✅ | **Demo Inferensi** | Script/Notebook/App yang berjalan pada input baru |

### Peringatan Keras dari Panduan
- ❌ **Data Leakage**: Jangan potong video jadi frame lalu acak ke train/test
- ❌ **Tanpa Baseline**: Tidak boleh langsung pakai model canggih tanpa pembanding sederhana
- ❌ **Hanya Demo Tanpa Evaluasi**: Demo live tanpa angka metrik tidak bernilai
- ❌ **Label Buruk**: Anotasi dataset harus diverifikasi kualitasnya

### Pembagian Fase Proyek

| Fase | Waktu | Output |
|---|---|---|
| **UTS (Sidang Proposal)** | Minggu 8 | Proposal 5–7 hal + Slide 6–8 hal + Bukti kelayakan awal |
| **Mingguan (CP00–CP13)** | Minggu 3–15 | Logbook individual di Google Sheet (total 100 XP) |
| **UAS (Sidang Demo)** | Minggu 16 | Demo live + Laporan teknis final + Slide + Source code di GitHub |

---

## 🏗️ Arsitektur Teknis — Topik Terpilih (Kandidat 1)

### Stack Teknologi

```
Dataset (Low-light v2 Roboflow)
    ↓
Preprocessing (CLAHE / Gamma Correction via Albumentations)
    ↓
Training (YOLOv8n/s — Ultralytics, Google Colab T4 GPU)
    ↓
Evaluasi (mAP@50, Recall, Precision, FPS)
    ↓
Demo (Streamlit / OpenCV — keputusan diambil setelah eksplorasi)
```

### Model yang Digunakan
| Komponen | Pilihan | Alasan |
|---|---|---|
| **Detektor** | YOLOv8 (Nano → Small) | Cepat, ringan, kompatibel CPU laptop |
| **Framework** | Ultralytics | Pipeline evaluasi otomatis (mAP, Confusion Matrix) |
| **Augmentasi** | Albumentations | Mendukung CLAHE & Gamma Correction native |
| **Platform Training** | Google Colab (GPU T4 gratis) | Estimasi 2–4 jam per eksperimen |
| **Demo** | Streamlit atau OpenCV | Belum diputuskan — akan ditentukan setelah eksplorasi data |

---

## 🧪 Rancangan Eksperimen Terkontrol

> Ini adalah inti proyek — yang membedakan dari sekadar "jalankan YOLO"

| Eksperimen | Deskripsi | Variabel Bebas |
|---|---|---|
| **Baseline (B0)** | YOLOv8n pretrained COCO diuji langsung pada gambar malam (zero-shot) | Tidak ada augmentasi |
| **Eksperimen A** | YOLOv8 fine-tuning pada data malam **tanpa augmentasi** | — |
| **Eksperimen B** | YOLOv8 fine-tuning + **augmentasi CLAHE** | Pencerah lokal adaptif |
| **Eksperimen C** | YOLOv8 fine-tuning + **augmentasi Gamma Correction** | Pencerah global merata |

### Pertanyaan Penelitian Utama
> *"Apakah augmentasi pencahayaan (CLAHE/Gamma) dapat meningkatkan Recall pejalan kaki pada kondisi minim cahaya tanpa mengorbankan Precision secara berlebihan, sambil mempertahankan kecepatan inferensi real-time?"*

---

## 📊 Metrik Evaluasi

| Metrik | Penjelasan Sederhana | Prioritas |
|---|---|---|
| **Recall @ IoU=0.5** | Dari semua pejalan kaki yang ada, berapa % yang berhasil terdeteksi? | 🥇 Utama |
| **Precision** | Dari semua yang dideteksi sebagai pejalan kaki, berapa % yang benar? | 🥈 Penting |
| **mAP@50** | Akurasi deteksi rata-rata standar internasional | 🥈 Penting |
| **mAP@50:95** | Versi lebih ketat dari mAP (IoU 50% s/d 95%) | 🥉 Pelengkap |
| **FPS** | Berapa frame per detik yang bisa diproses (target ≥ 25 FPS) | 🥉 Pelengkap |
| **Latency (ms)** | Waktu inferensi per gambar di CPU laptop | 🥉 Pelengkap |

---

## 📦 Dataset — Detail Lengkap

### Dataset Utama: Low-light v2 (Roboflow)
> **Status:** ✅ Sudah diunduh, diekstrak, dan diverifikasi (*Object Detection YOLOv8 Format*)  
> **Lokasi Repositori:** `c:\Project Github\comvis project\dataset\`  
> **Lisensi:** CC BY 4.0 (bebas pakai untuk akademik)

| Info | Detail |
|---|---|
| **Total Gambar** | 2.400 gambar kondisi gelap/minim cahaya |
| **Resolusi** | 640 × 640 piksel |
| **Kelas Objek** | 2 kelas: `Vehicle` (ID: 0) dan `person` (ID: 1) |
| **Format Label** | Standar YOLOv8 Bounding Box (`<class_id> <x_c> <y_c> <w> <h>` per file `.txt`) |
| **Total Bounding Box** | 7.777 objek (Vehicle: 6.871, Person: 906) |

### Pembagian Split & Distribusi Anotasi

| Split | Gambar | Total Box | Bounding Box `Vehicle` | Bounding Box `person` | Gambar Memuat `person` | Gambar Kosong |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 2.100 | 6.816 | 6.018 (88,3%) | 798 (11,7%) | 413 (19,7%) | 4 |
| **Valid** | 200 | 650 | 581 (89,4%) | 69 (10,6%) | 35 (17,5%) | 0 |
| **Test** | 100 | 311 | 272 (87,5%) | 39 (12,5%) | 18 (18,0%) | 0 |
| **Total** | **2.400** | **7.777** | **6.871 (88,4%)** | **906 (11,6%)** | **466 (19,4%)** | **4** |

> ⚠️ **Class Imbalance:** Objek `person` hanya mencakup **11,6%** dari total objek di jalan (rasio Vehicle : Person = 7,58 : 1). Hal ini memperkuat argumentasi akademik di proposal bahwa metrik **Recall pejalan kaki** harus menjadi prioritas evaluasi utama agar sistem keselamatan tidak melewatkan korban (*false negative* fatal).

### Augmentasi yang Sudah Diterapkan Roboflow (3× lipat dari data asli)
- Flip horizontal (50% probabilitas)
- Random crop 0–20%
- Penyesuaian kecerahan acak ±10%

---

## 🔧 Keputusan Teknis

### ✅ Sudah Diputuskan

| Keputusan | Pilihan | Alasan |
|---|---|---|
| Model utama | YOLOv8 (Ultralytics) | Matang, pipeline evaluasi otomatis, bisa jalan di CPU |
| Teknik augmentasi | CLAHE + Gamma Correction (via Albumentations) | Standar penelitian low-light detection |
| Platform training | Google Colab T4 (gratis) | Cukup untuk estimasi 2–4 jam/eksperimen |
| Jumlah eksperimen | 3 (A, B, C) + 1 Baseline (B0) | Sesuai rubrik eksperimen terkontrol |
| Prioritas metrik | Recall → Precision → mAP | Melewatkan pejalan kaki lebih berbahaya dari false alarm |
| Dataset | Low-light v2 Roboflow (sudah diunduh) | Legal, berlabel, siap pakai |

### ❓ Belum Diputuskan

| Keputusan | Opsi | Kapan Diputuskan |
|---|---|---|
| Platform demo | Streamlit vs OpenCV langsung | Setelah eksplorasi data awal selesai |
| Ukuran model final | YOLOv8-Nano vs YOLOv8-Small | Setelah eksperimen awal (lihat trade-off FPS vs mAP) |
| Penanganan class imbalance | Weighted loss vs threshold tuning | Setelah melihat hasil Baseline |

---

## 📏 Rules Coding yang Diterapkan

> Dipilih dari folder `rules/` yang sudah ada

| Priority | File Rule | Poin Kunci yang Diterapkan |
|:---:|---|---|
| 🥇 | `tensorflow-deep-learning.mdc` | Validasi data sebelum training, anti data leakage, analisis confusion matrix |
| 🥇 | `automl-hyperparameter-optimization.mdc` | Baseline dulu sebelum eksperimen, split sebelum augmentasi, catat semua parameter |
| 🥈 | `clean-code.mdc` | Named constants untuk threshold, fungsi satu tanggung jawab, kode self-documenting |
| 🥈 | `git-conventional-commit-messages.mdc` | Format commit: `feat(dataset):`, `fix(train):`, `docs(readme):` |
| 🥉 | `anti-overengineering.mdc` | Fokus pada 3 eksperimen yang sudah didefinisikan, jangan scope creep |
| 🥉 | `python-llm-ml-workflow.mdc` | Struktur folder rapi, Streamlit/Gradio untuk demo, yaml untuk config |

### Contoh Struktur Folder yang Direkomendasikan
```
DeteksiPedestrianLowLight/
├── data/
│   ├── raw/              # Dataset asli dari Roboflow (jangan diubah)
│   └── processed/        # Hasil setelah preprocessing
├── src/
│   ├── dataset.py        # Load & augment data
│   ├── train.py          # Pipeline training
│   ├── evaluate.py       # Hitung metrik
│   └── demo.py           # Inferensi real-time
├── configs/
│   └── experiment.yaml   # Hyperparameter (bukan hardcode!)
├── notebooks/
│   └── exploratory.ipynb # Eksplorasi data awal
├── results/
│   └── experiments/      # Output tiap eksperimen A/B/C
├── requirements.txt
└── README.md
```

### Contoh Format Commit yang Benar
```
feat(dataset): add CLAHE augmentation pipeline
fix(train): fix class imbalance with weighted loss
docs(readme): add experiment A/B/C description  
perf(model): switch from YOLOv8n to YOLOv8s for recall
test(eval): add confusion matrix visualization
```

---

## 📅 Timeline & Rencana Langkah

### Minggu Ini (Minggu 3 — CP01)
- [x] Pilih 3 kandidat topik
- [x] Unduh dan verifikasi dataset Low-light v2
- [x] Kumpulkan tugas pemilihan topik (Senin, 28 September 2026)
- [x] Presentasi pemilihan topik (Selasa, 29 September 2026)

### Pekan Depan (Minggu 4 — CP02 & CP03)
- [x] Audit dataset & verifikasi integritas YOLOv8 format (selesai via `src/audit_dataset.py`, bukti di `results/audit/`)
- [ ] Kumpulkan 5 rujukan ilmiah (2 paper metode/dataset) & rumuskan problem statement (CP02)
- [ ] Jalankan inferensi Baseline (B0) pada gambar validasi malam hari (CP04 awal)
- [ ] Mulai susun draft Bab 1–3 proposal UTS (Latar Belakang, Tujuan, Batasan)

### Target UTS (Minggu 8)
- [ ] Dokumen Proposal (5–7 halaman) selesai
- [ ] Slide presentasi UTS (6–8 slide) selesai
- [ ] Bukti kelayakan awal: hasil baseline + minimal 1 eksperimen berjalan

### Target UAS (Minggu 16)
- [ ] 3 eksperimen (A, B, C) selesai dan dibandingkan
- [ ] Analisis kasus gagal (failure cases) terdokumentasi
- [ ] Demo Streamlit / OpenCV berjalan pada input baru
- [ ] Laporan teknis final + Slide UAS selesai
- [ ] Source code di-push ke GitHub dengan struktur rapi

---

## 💡 Poin Penting yang Sering Terlupakan

> Berdasarkan peringatan di panduan dosen & diskusi kita

1. **Split data SEBELUM augmentasi** — CLAHE/Gamma hanya diterapkan pada data train, bukan val/test
2. **Test set tidak boleh disentuh** sampai semua eksperimen selesai — satu kali evaluasi final saja
3. **Catat random seed** di setiap eksperimen untuk reproduksibilitas
4. **FPS diukur dengan warm-up run** — bukan langsung frame pertama yang biasanya lambat
5. **Kelas `person` sangat sedikit** (3%) — Recall bisa terlihat buruk bukan karena modelnya jelek, tapi karena datanya tidak seimbang
6. **Demo ≠ hanya live camera** — demo bisa juga pakai video pre-recorded yang sudah terbukti ada pejalan kakinya

---

## 📎 File-File Penting

| File | Lokasi | Deskripsi |
|---|---|---|
| Panduan Proyek | [`Panduan_Proyek_Akhir_Visi_Komputer_UTS_UAS.pdf`](file:///c:/Users/kryan/OneDrive/Documents/kuliah/smt%205/Computer%20vision/Panduan_Proyek_Akhir_Visi_Komputer_UTS_UAS.pdf) | Panduan resmi dosen |
| Laporan Kelompok | [`Pemilihan Topik Proyek Visi Komputer__Komputer Vision _Kelompok 9.docx`](file:///c:/Users/kryan/OneDrive/Documents/kuliah/smt%205/Computer%20vision/Pemilihan%20Topik%20Proyek%20Visi%20Komputer__Komputer%20Vision%20_Kelompok%209.docx) | Tugas yang sudah dikumpul |
| Naskah Presentasi | [`Naskah_Presentasi_Kelompok9.md`](file:///C:/Users/kryan/.gemini/antigravity/brain/0ce34b0d-6c7f-448c-99ca-ef7263aca829/Naskah_Presentasi_Kelompok9.md) | Naskah + antisipasi pertanyaan dosen |
| Dataset | [`Data set topik 1/`](file:///c:/Users/kryan/OneDrive/Documents/kuliah/smt%205/Computer%20vision/Data%20set%20topik%201/) | Low-light v2 Roboflow (sudah diunduh) |
