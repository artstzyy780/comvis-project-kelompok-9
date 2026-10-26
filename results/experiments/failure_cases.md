# 🔍 Analisis Kasus Gagal (Failure Case Analysis)

> Checkpoint CP11 — *kategori kegagalan, contoh kasus, dan hipotesis*.

## 1. Ringkasan

- **Model:** `hasil_final_model\hasil_final_model\Exp_A_Standard_best.pt`
- **Split:** `test` (373 citra)
- **Jumlah kotak pejalan kaki teranotasi:** 79
- **Total kasus gagal teranalisis:** 53

## 2. Distribusi Kategori Kegagalan

| Kategori | Jumlah | Persentase | Interpretasi |
|:---|:---:|:---:|:---|
| `low_confidence` | 26 | 49.1% | Keyakinan model rendah |
| `false_positive` | 18 | 34.0% | False positive (bukan pejalan kaki) |
| `low_contrast` | 7 | 13.2% | Kontras rendah (siluet menghilang) |
| `unlabelled_or_ambiguous` | 2 | 3.8% | Ambiguitas anotasi/taksonomi |
| `small_object` | 0 | 0.0% | Pejalan kaki kecil (jarak jauh) |
| `occluded` | 0 | 0.0% | Terhalang kendaraan |

## 3. Contoh Kasus per Kategori

### Kontras rendah (siluet menghilang) (`low_contrast`)

- `381c576a-6df3dd9a_augmented_0_jpg.rf.9afd395a0133ffe9a8fad2e2f2be6d45.jpg` — Pejalan kaki teranotasi tidak terdeteksi (tinggi 21.1% dari citra).
- `4ec75dfc-e66d0059_augmented_0_jpg.rf.5fbc58531a9424e8a4da9dcb99e0e749.jpg` — Pejalan kaki teranotasi tidak terdeteksi (tinggi 26.5% dari citra).
- `52ddd3f7-e0af4171_augmented_1_jpg.rf.2838152193cc7dc81f4188c436b2c502.jpg` — Pejalan kaki teranotasi tidak terdeteksi (tinggi 23.1% dari citra).

### Keyakinan model rendah (`low_confidence`)

- `05ba4355-bf968804_augmented_1_jpg.rf.0ce56b46aefebc6374cc629dfddc1d9e.jpg` — Deteksi pejalan kaki tanpa anotasi yang cocok. (conf 0.37)
- `15ab701b-7b8fe159_augmented_0_jpg.rf.28188f9b175c2bd47c32090b439e58da.jpg` — Deteksi pejalan kaki tanpa anotasi yang cocok. (conf 0.31)
- `1a5d0011-c34c50cc_augmented_0_jpg.rf.59c1e1d956b35018be915b9826d5fb3a.jpg` — Deteksi pejalan kaki tanpa anotasi yang cocok. (conf 0.26)

### Ambiguitas anotasi/taksonomi (`unlabelled_or_ambiguous`)

- `4ec75dfc-e66d0059_augmented_2_jpg.rf.f872f84d810d8bed1eebb40ceeecd2c9.jpg` — Deteksi pejalan kaki tanpa anotasi yang cocok. (conf 0.55)
- `7cfff646-d2a42999_augmented_1_jpg.rf.7bc6074d18bfdaeed442cb16a0baf3ca.jpg` — Deteksi pejalan kaki tanpa anotasi yang cocok. (conf 0.32)

### False positive (bukan pejalan kaki) (`false_positive`)

- `1a5d0011-c34c50cc_augmented_0_jpg.rf.dc3a7779927e18210923ba7bea8f0e33.jpg` — Deteksi pejalan kaki tanpa anotasi yang cocok. (conf 0.43)
- `1a5d0011-c34c50cc_augmented_2_jpg.rf.6b6187c70908022b3a1229fc1ab50211.jpg` — Deteksi pejalan kaki tanpa anotasi yang cocok. (conf 0.49)
- `1a5d0011-c34c50cc_augmented_2_jpg.rf.6b6187c70908022b3a1229fc1ab50211.jpg` — Deteksi pejalan kaki tanpa anotasi yang cocok. (conf 0.43)

## 4. Hipotesis dan Perbaikan yang Diuji

1. **Hipotesis utama:** sebagian besar false negative berasal dari *near-duplicate antar split* pada eksport Roboflow, bukan dari keterbatasan model.
   - **Perbaikan yang diuji:** `src/rebuild_split_dedup.py` membangun ulang split pada level source scene; seluruh eksperimen diulang pada `dataset_dedup/`.
2. **Hipotesis kedua:** pejalan kaki kecil dan terhalang kendaraan merupakan sumber false negative yang tidak dapat dihilangkan hanya dengan pra-pemrosesan.
   - **Perbaikan yang diuji:**Threshold.confidence diturunkan pada inferensi (operating point recall-first), dibuktikan lewat tabel unified (CP10).
3. **Hipotesis ketiga:** CLAHE menaikkan kontras tetapi juga mengamplifikasi noise, sehingga recall turun sementara precision naik.
   - **Perbaikan yang diuji:** perbandingan Exp-A vs Exp-B vs Exp-C pada split yang sama dengan interval keyakinan (CP10).

## 5. Visual Evidence

Komparasi ground truth dan prediksi tersimpan di `failure_cases.png`.

---

*(Laporan ini dihasilkan secara otomatis oleh `src/analyze_failures.py`)*