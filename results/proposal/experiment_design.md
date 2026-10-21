# Desain Eksperimen Terkontrol - B0, Exp-A, Exp-B, Exp-C

> **Checkpoint:** CP05 - Desain eksperimen dan diagram pipeline (7 XP)  
> **Mata Kuliah:** Visi Komputer (IF0PP006) - Semester Ganjil 2026/2027  
> **Kelompok:** Kelompok 9 (Krisantus Amka Ginting, Linardi Halim, Kristofer Ryan Giggs Eka Saputra, Leonard Franscois)  
> **Tanggal:** 20 Oktober 2026  
> **Sumber konfigurasi:** `configs/experiments.yaml` (sumber kebenaran tunggal untuk registry eksperimen).

---

## 1. Tabel Desain Eksperimen

| ID | Model | Perlakuan Data Latih | Split Evaluasi | Fokus Metrik | Hipotesis yang Diharapkan | Variabel Bebas Tunggal |
|:---:|---|---|:---:|---|---|---|
| **B0** | YOLOv8n (COCO pretrained) | Tanpa pelatihan (zero-shot); citra malam asli apa adanya | Test (100 citra) | Baseline Recall & mAP@50 | Detektor siang hari gagal pada citra malam sehingga Recall pejalan kaki sangat rendah (< 50%) | Tidak ada pelatihan sama sekali (acuan pembanding) |
| **Exp-A** | YOLOv8n fine-tuned | Fine-tuning pada 2.100 citra malam asli (tanpa pencerahan) | Test (100 citra) | Recall domain fine-tuning | Fine-tuning domain menaikkan Recall pejalan kaki secara signifikan dibanding B0 | Fine-tuning tanpa augmentasi pencahayaan (kontrol) |
| **Exp-B** | YOLOv8n fine-tuned | Fine-tuning pada 2.100 citra yang diperkuat CLAHE kanal-L (clip 2,0; tile 8x8) pada ruang CIE L*a*b* | Test (100 citra) | Recall pejalan kaki target (+10%) & mAP@50 | CLAHE kontras lokal meninggikan siluet pejalan kaki sehingga Recall naik tanpa menurunkan Precision secara besar | Pra-pemrosesan CLAHE pada data latih saja |
| **Exp-C** | YOLOv8n fine-tuned | Fine-tuning pada 2.100 citra yang dikoreksi Gamma non-linear (gamma = 0,6) | Test (100 citra) | Recall pejalan kaki pada kecerahan sedang & kontras global | Gamma correction global dapat mengangkat bayangan gelap tanpa memperbesar derau frekuensi tinggi sehingga Recall pejalan kaki paling tinggi | Pra-pemrosesan Gamma Correction pada data latih saja |

## 2. Aturan Eksperimen Terkontrol

1. **Satu variabel bebas per eksperimen.** Arsitektur, bobot awal, jumlah epoch, batch size, ukuran citra, dan random seed dikunci; hanya perlakuan data latih yang berubah (seed 42, epoch 50, batch 16, imgsz 640).
2. **Split data sebelum augmentasi.** Prapemrosesan CLAHE dan Gamma hanya diterapkan pada split latih; split validasi dan uji tetap citra malam asli sehingga tidak terjadi kebocoran data.
3. **Split uji digunakan satu kali.** Evaluasi pada 100 citra test dilakukan hanya setelah seluruh eksperimen selesai dan seluruh parameter terkunci.
4. **Metrik utama Recall pejalan kaki pada IoU 0,50.** Precision, mAP@50, mAP@50:95, dan FPS dilaporkan sebagai metrik pendukung.
5. **Warm-up sebelum pengukuran FPS.** Lima iterasi warm-up tidak dihitung dalam statistik latensi.

## 3. Kriteria Keberhasilan (Kriteria Sukses)

| Kriteria | Ambang |
|:---|:---:|
| Peningkatan Recall pejalan kaki dibanding B0 | > +10 poin persentase |
| Penurunan Precision pejalan kaki akibat augmentasi | < 10 poin persentase |
| Throughput inferensi pada CPU laptop | >= 20 FPS |
| Kepatuhan aturan anti data leakage | 100% (audit otomatis pada split) |

## 4. Reproduksi

```bash
python src/render_pipeline_diagram.py
```

---

*(Dokumen ini dihasilkan secara otomatis oleh `src/render_pipeline_diagram.py` untuk pemenuhan Checkpoint CP05.)*
