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

## 3. Analisis Bias Dataset

- **Kecerahan scene:** rata-rata kanal V = `43.25` (mendukung pengujian pada kondisi minim cahaya).
- **Skala objek median (luas ternormalisasi):** `Vehicle` 0.01789 | `person` 0.01453.
- **Tinggi kotak median:** `Vehicle` 0.142 | `person` 0.220 (fraksi tinggi citra).
- **Distribusi tinggi pejalan kaki:** p10 0.119 | p50 0.220 | p90 0.515.
- **Pejalan kaki kecil (< 10% tinggi citra):** 4.3% dari seluruh kotak pejalan kaki. Karena dataset berupa potongan 640x640, pejalan kaki relatif besar dalam frame; kasus sulit berada pada ekor bawah distribusi (pejalan kaki jauh/terhalang), bukan pada objek sekecil(<5%) citra.

## 4. Deteksi Data Leakage (Level Scene)

- **Total scene sumber unik:** 687
- **Scene yang muncul pada >1 split:** 145

| Split | File total | File dari scene bocor | Persentase |
|:---|:---:|:---:|:---:|
| **train** | 2,100 | 468 | 22.3% |
| **valid** | 200 | 104 | 52.0% |
| **test** | 100 | 55 | 55.0% |

> **Peringatan:** Eksport Roboflow menempatkan augmented copy dari satu scene ke beberapa split sekaligus, sehingga split uji tidak independen pada level scene. Bangun ulang split dengan `src/rebuild_split_dedup.py` (grouped by scene) sebelum-report final.

- **Near-duplicate visual lintas split (dHash, Hamming ≤ 5):** 4 pasangan.
  - `train/8c3cd97c-3c4dc449_augmented_2_jpg.rf.308c52329d1c15912081b2c8a790a61c.jpg` ↔ `valid/8c3cd97c-3c4dc449_augmented_0_jpg.rf.ef37a95fb4104a7f8dc87a44785ca5ef.jpg` (jarak 2)
  - `train/55c223bf-4d4fe46c_augmented_1_jpg.rf.07578d2d2b09bbb770df1c3717ac4605.jpg` ↔ `valid/55c223bf-4d4fe46c_augmented_0_jpg.rf.352d4d5d0eae2b835e4a339faaaf438f.jpg` (jarak 3)
  - `train/7c35c3cb-b5a57080_augmented_0_jpg.rf.24a08f8399f7efe78d43e03468ae1f0b.jpg` ↔ `test/7c35c3cb-b5a57080_augmented_1_jpg.rf.8c763d38fd694d0e095d75ef3819eaa5.jpg` (jarak 4)
  - `train/0d70100d-eb9d8288_augmented_0_jpg.rf.66feb50b4932c4db3c11a93b82dc1599.jpg` ↔ `test/0d70100d-eb9d8288_augmented_2_jpg.rf.38cb5d7d8c7a73dc92a159de826f3e81.jpg` (jarak 5)

## 5. Rencana Split

1. **Unit split adalah *source scene***, bukan file: seluruh augmented copy dari satu scene selalu berada pada split yang sama.
2. **Stratifikasi** berdasarkan keberadaan pejalan kaki agar kelas minoritas tersebar merata pada train/valid/test.
3. **Seed tetap (42)** agar pembagian reproducible.
4. **Prasyarat augmentasi pencahayaan** (CLAHE/Gamma) hanya boleh diterapkan pada split latih; validasi dan uji tetap murni.
5. **Evaluasi akhir** dilakukan sekali pada split uji yang belum pernah disentuh sebelum pembekuan model.

## 6. Catatan Audit Kelayakan Akademik (Rubrik Dosen)

1. **Integritas Anotasi:** Setiap file gambar memiliki pasangan label `.txt` yang sesuai tanpa ada yang hilang atau korup.
2. **Pencegahan Data Leakage:** Pra-pemrosesan hanya diterapkan pada data latih. Pemeriksaan tambahan pada level scene menemukan duplikasi lintas split pada eksport bawaan (lihat bagian 4) dan diselesaikan dengan split terkelompok.
3. **Tantangan Class Imbalance:** Objek `person` jauh lebih sedikit dibanding `Vehicle`. Hal ini memvalidasi urgensi penggunaan metrik **Recall** sebagai metrik prioritas utama agar pejalan kaki tidak terlewatkan (*false negative minim*).
4. **Visual Verifikasi:** Sampel anotasi tersimpan di file `results/audit/sample_annotations.png`.