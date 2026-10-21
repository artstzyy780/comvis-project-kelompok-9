# Kebutuhan Komputasi Proyek - Deteksi Pejalan Kaki Minim Cahaya

> **Checkpoint:** CP05 - Diagram pipeline dan kebutuhan komputasi (7 XP)  
> **Mata Kuliah:** Visi Komputer (IF0PP006) - Semester Ganjil 2026/2027  
> **Kelompok:** Kelompok 9 (Krisantus Amka Ginting, Linardi Halim, Kristofer Ryan Giggs Eka Saputra, Leonard Franscois)  
> **Tanggal:** 20 Oktober 2026  
> **Referensi angka:** `Master_Reference_Proyek_CV.md`, `results/experiments/benchmark_latency_report.md`, dan `results/experiments/colab_training_benchmark_report.md`.

---

## 1. Ringkasan Lingkungan Eksekusi

| Komponen | Spesifikasi |
|---|---|
| Platform pelatihan | NVIDIA Tesla T4 (Google Colab), 16 GB VRAM (gratis) |
| Platform inferensi | CPU laptop lokal (AMD Ryzen 6-core / 12-thread @ 2,3 GHz), RAM 15,4 GB |
| Sistem operasi host | Windows 11 |
| Resolusi kerja | 640x640 piksel |
| Bobot model | YOLOv8n, sekitar 3,2 juta parameter (sekitar 6,5 MB) |

## 2. Estimasi Biaya Pelatihan (Training Cost)

Perhitungan berdasarkan konfigurasi terkunci pada `configs/experiments.yaml`: 2.100 citra latih, batch 16, imgsz 640, 50 epoch, seed 42. Dengan 132 iterasi per epoch dan sekitar 0,90 detik per iterasi pada GPU T4.

| Eksperimen | Jumlah Citra Latih | Iterasi/Epoch | Durasi per Epoch | Total Durasi (50 epoch) |
|:---:|:---:|:---:|:---:|:---:|
| **Exp-A** | 2.100 | 132 | 2,0 menit | sekitar 2,1 jam |
| **Exp-B** | 2.100 | 132 | 2,0 menit | sekitar 2,1 jam |
| **Exp-C** | 2.100 | 132 | 2,0 menit | sekitar 2,1 jam |
| **Total 3 eksperimen** | 2.100 x 3 | 132 x 3 | 5,9 menit | **sekitar 6,2 jam** |

### Catatan Tambahan Biaya Pelatihan

- **Anggaran waktu aman:** Dengan rentang 2-4 jam per eksperimen seperti yang dokumentasikan pada Master Reference, total biaya GPU berada pada rentang 6-12 jam untuk ketiga eksperimen (estimasi terhitung di atas adalah durasi pelatihan murni tanpa jeda sesi Colab gratis).
- **Kebutuhan VRAM:** sekitar 4-6 GB, masih jauh di bawah kapasitas 16 GB milik T4, sehingga batch size 16 aman tanpa penyesuaian.
- **Kebutuhan penyimpanan:** sekitar 1,2 GB per varian dataset (mentah, CLAHE, dan Gamma), sehingga total sekitar 3,6 GB di Google Drive.
- **Ketidakdeterminan GPU:** performa T4 di Colab gratis dapat menurun bila sesi terputus, sehingga hasil terbaik disimpan dari checkpoint validasi, bukan dari epoch terakhir.

## 3. Estimasi Biaya Inferensi (Inference Cost)

Diukur pada host lokal dengan 5 iterasi warm-up sebelum pengukuran, batch size 1, dan resolusi 640x640 (sumber: `results/experiments/benchmark_latency_report.md`).

| Parameter | Nilai Terukur |
|:---|:---|
| Prosesor | AMD Ryzen 6-core / 12-thread @ 2,3 GHz |
| Memori RAM | 15,4 GB |
| Perangkat inferensi | CPU (tanpa GPU) |
| Resolusi input | 640x640 piksel, batch size 1 |
| Model | YOLOv8n (3,2 juta parameter) |
| Latensi rata-rata | 52-55 ms per citra |
| Throughput | 18-19 FPS |
| Ambang target proyek | 20 FPS |
| Status ambang target | **BORDERLINE** - sekitar 1-2 FPS di bawah ambang |

### Rincian Latensi per Tahap (ms)

| Tahap | Rentang Terukur |
|:---|:---|
| Prapemrosesan (resize + normalisasi) | 1,9 - 2,7 ms |
| Model forward (inferensi YOLOv8n) | 47,9 - 51,0 ms |
| Post-processing (NMS) | 0,8 - 1,0 ms |
| **Total** | **52-55 ms** |

### Mitigasi terhadap ambang 20 FPS

1. **Ekspor ONNX dengan kuantisasi INT8** yang dapat menurunkan bobot model sekitar 4 kali dan mempercepat inferensi pada CPU.
2. **Multithreading OpenCV** untuk memisahkan pembacaan frame, prapemrosesan, dan inferensi ke thread berbeda.
3. **Penggunaan TFLite atau ONNX Runtime** sebagai runtime inferensi pengganti PyTorch pada skrip demo.
4. **Pernyataan ambang secara eksplisit** pada proposal agar klaim real-time dapat diverifikasi dosen dengan angka yang sama.

## 4. Kontrol Perbandingan Antar Eksperimen

Agar kesimpulan ilmiah valid, seluruh parameter berikut dikunci (tidak berubah) dan hanya perlakuan data latih yang menjadi variabel bebas.

| Parameter | B0 | Exp-A | Exp-B | Exp-C | Status Kontrol |
|:---|:---:|:---:|:---:|:---:|:---:|
| Arsitektur model | YOLOv8n | YOLOv8n | YOLOv8n | YOLOv8n | **Dikunci** |
| Bobot awal | yolov8n.pt (COCO) | yolov8n.pt (COCO) | yolov8n.pt (COCO) | yolov8n.pt (COCO) | **Dikunci** |
| Epoch | 0 (tanpa pelatihan) | 50 | 50 | 50 | **Dikunci** (Exp-A/B/C) |
| Batch size | - | 16 | 16 | 16 | **Dikunci** |
| Ukuran citra (imgsz) | 640 | 640 | 640 | 640 | **Dikunci** |
| Random seed | 42 | 42 | 42 | 42 | **Dikunci** |
| Optimizer / lr0 | - | auto / 0,01 | auto / 0,01 | auto / 0,01 | **Dikunci** |
| Split validasi | valid (200) | valid (200) | valid (200) | valid (200) | **Dikunci** |
| Split uji | test (100) | test (100) | test (100) | test (100) | **Dikunci** |
| Split latih | - | 2.100 | 2.100 | 2.100 | **Dikunci** |
| Augmentasi Roboflow | flip, crop, brightness | flip, crop, brightness | flip, crop, brightness | flip, crop, brightness | **Dikunci** |
| **Perlakuan data latih** | **tanpa pelatihan (zero-shot)** | **citra malam asli** | **citra + CLAHE kanal-L** | **citra + Gamma 0,6** | **VARIABEL BEBAS SATU-SATUNYA** |

Aturan evaluasi yang berlaku: pembagian split data dilakukan **sebelum** augmentasi, split uji hanya dievaluasi satu kali pada tahap akhir, dan metrik utama adalah Recall pejalan kaki pada IoU 0,50.

## 5. Reproduksi

```bash
python src/render_pipeline_diagram.py
```

---

*(Dokumen ini dihasilkan secara otomatis oleh `src/render_pipeline_diagram.py` untuk pemenuhan Checkpoint CP05.)*
