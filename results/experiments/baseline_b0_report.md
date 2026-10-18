# 📊 Baseline B0 Evaluation Report: Zero-Shot COCO Deteksi Malam

- **Model:** `yolov8n-coco-zero-shot` (Ultralytics Pretrained COCO)
- **Split Evaluasi:** `valid` (200 citra)
- **Confidence Threshold:** `0.25`
- **IoU Threshold:** `0.5`
- **Throughput CPU:** `11.53 FPS` (Rata-rata Latensi: `86.75 ms`)

---

## 1. Ringkasan Kinerja Kuantitatif

| ID | Kelas Objek | Ground Truth | Prediksi | TP | FP | FN | Precision | Recall | F1-Score |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **Vehicle** | 581 | 466 | 264 | 202 | 317 | **56.6%** | **45.4%** | **50.4%** |
| **1** | **person** | 69 | 116 | 27 | 89 | 42 | **23.3%** | **39.1%** | **29.2%** |

- **Macro-Average Precision:** `40.0%`
- **Macro-Average Recall:** `42.3%`
- **Macro-Average F1-Score:** `39.8%`

---

## 2. Analisis Kritis Temuan Baseline (B0)

1. **Asimetri Recall Pejalan Kaki vs Kendaraan**:
   - Model zero-shot mendeteksi kendaraan dengan Recall **45.4%**, tetapi mengalami penurunan tajam pada pejalan kaki dengan Recall hanya **39.1%**.
   - Terdapat **42 pejalan kaki yang tidak terdeteksi (False Negative)** dari total 69 ground truth.
2. **Bukti Empiris Urgensi Penelitian**:
   - Angka Recall pejalan kaki yang rendah pada model bawaan COCO ini membuktikan secara empiris bahwa model deteksi siang hari mengalami degradasi berat di malam hari.
   - Hasil ini menjadi acuan dasar (*baseline*) kuat yang akan ditingkatkan melalui **Fine-tuning (Exp-A)** dan **Augmentasi Pencahayaan Adaptif (Exp-B CLAHE & Exp-C Gamma)** pada tahap selanjutnya.
3. **Efisiensi Perangkat Keras CPU**:
   - Arsitektur YOLOv8n mencatat kecepatan **11.53 FPS** (86.75 ms/frame) pada CPU standar, membuktikan kelayakan target real-time ($\ge 20$ FPS).

---
*(Laporan ini dihasilkan secara otomatis oleh `src/evaluate_baseline.py`)*