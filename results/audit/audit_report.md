# 📊 Laporan Audit Dataset Low-light v2 (YOLOv8 Format)

> Laporan ini dihasilkan secara otomatis untuk pemenuhan **Checkpoint CP03** dan Bab 5 Proposal UTS.

## 1. Ringkasan Keseluruhan

- **Total Gambar:** 2,400 citra
- **Total Bounding Box (Anotasi):** 7,777 kotak
- **Total Objek Kendaraan (`Vehicle`):** 6,871 (88.4%)
- **Total Objek Pejalan Kaki (`person`):** 906 (11.6%)
- **Rasio Ketidakseimbangan (Vehicle : Person):** 7.58 : 1

## 2. Distribusi per Split Data

| Split | Gambar | Total Box | Bounding Box `Vehicle` | Bounding Box `person` | Gambar ada `person` | Gambar Kosong |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 2,100 | 6,816 | 6,018 | 798 | 413 (19.7%) | 4 |
| **Valid** | 200 | 650 | 581 | 69 | 35 (17.5%) | 0 |
| **Test** | 100 | 311 | 272 | 39 | 18 (18.0%) | 0 |

## 3. Catatan Audit Kelayakan Akademik (Rubrik Dosen)

1. **Integritas Anotasi:** Setiap file gambar memiliki pasangan label `.txt` yang sesuai tanpa ada yang hilang atau korup.
2. **Pencegahan Data Leakage:** Split data sudah dipisah di level folder sebelum eksperimen dan pra-pemrosesan.
3. **Tantangan Class Imbalance:** Objek `person` jauh lebih sedikit dibanding `Vehicle`. Hal ini memvalidasi urgensi penggunaan metrik **Recall** sebagai metrik prioritas utama agar pejalan kaki tidak terlewatkan (*false negative minim*).
4. **Visual Verifikasi:** Sampel anotasi tersimpan di file `results/audit/sample_annotations.png`.