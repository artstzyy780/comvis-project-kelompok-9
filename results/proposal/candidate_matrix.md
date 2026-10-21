# Matriks Keputusan Topik - Proyek Visi Komputer

> **Checkpoint:** CP01 - Bukti kelayakan & matriks keputusan topik (4 XP)  
> **Mata Kuliah:** Visi Komputer (IF0PP006) - Semester Ganjil 2026/2027  
> **Kelompok:** Kelompok 9 (Krisantus Amka Ginting, Linardi Halim, Kristofer Ryan Giggs Eka Saputra, Leonard Franscois)  
> **Tanggal:** 20 Oktober 2026  
> **Sumber data:** `Master_Reference_Proyek_CV.md` (bagian Tiga Kandidat Topik dan Rubrik Penilaian)

---

## 1. Ringkasan Kandidat Topik

Tiga kandidat topik yang telah dievaluasi oleh kelompok, beserta status kelulusannya:

| ID | Kandidat Topik | Tugas CV | Status |
|:---:|---|---|:---:|
| **C1** | Deteksi Pejalan Kaki di Malam Hari (Night-time Pedestrian Detection) | Object Detection | **TERPILIH** |
| **C2** | Segmentasi Kerusakan Bodi Mobil (Goresan & Penyok) | Semantic Segmentation | Tidak dipilih |
| **C3** | Pelacakan Multi-Objek Bawah Air (Multi-Object Tracking saat Oklusi) | Multi-Object Tracking (MOT) | Tidak dipilih |

## 2. Profil Rinci Setiap Kandidat

| Aspek | Kandidat C1 | Kandidat C2 | Kandidat C3 |
|---|---|---|---|
| **Masalah utama** | Model deteksi standar yang dilatih pada data siang hari mengalami penurunan drastis pada kondisi malam karena kontras rendah, derau sensor, dan silau lampu kendaraan sehingga pejalan kaki lolos terdeteksi (false negative fatal). | Penilaian kerusakan mobil untuk klaim asuransi masih dilakukan manual, subjektif, dan lambat sehingga menimbulkan ketidaksepakatan antar penilai yang tinggi. | Sistem pelacakan sering keliru menukar identitas objek (ID-Switch) ketika objek saling menutupi di dalam air dengan visibilitas yang sangat buruk. |
| **Input** | Foto atau frame video kamera jalanan malam hari (JPEG/PNG/MP4), resolusi 640x640 piksel | Foto bodi kendaraan rusak (goresan, penyok, cat terkelupas) | Klip video pendek rekaman bawah air (multi-objek, oklusi tinggi) |
| **Output** | Bounding box pejalan kaki + label 'Pedestrian' + confidence score + FPS | Mask piksel area kerusakan + persentase luas kerusakan per citra | ID unik konsisten per objek + garis trajektori + laporan ID-Switch |
| **Ketersediaan dataset** | Tersedia: Roboflow Low-light v2 (2.400 citra, 7.777 anotasi, lisensi CC BY 4.0), sudah diunduh dan diaudit pada src/audit_dataset.py | Belum tersedia: tidak ada dataset anotasi mask yang siap pakai; butuh pengumpulan citra dan pelabelan piksel secara manual | Belum tersedia: dataset MOT bawah air yang legal dan siap pakai tidak ada di repositori proyek; perekaman sendiri berisiko lisensi |
| **Ketersediaan baseline** | Tersedia: YOLOv8n pretrained COCO sebagai baseline zero-shot (B0) dan YOLOv8n fine-tuned sebagai pembanding (Exp-A) | Ada secara teori (U-Net/segmentasi semantik), namun harus dibangun dari nol karena label mask belum tersedia | Ada (SORT dan ByteTrack), namun harus digabungkan dengan detektor sehingga pipeline menjadi dua tahap |
| **Kelayakan metrik** | Sangat layak: mAP@50, mAP@50:95, Precision, Recall, dan FPS merupakan metrik standar deteksi objek | Cukup: mIoU dan F1 per kelas tersedia, tetapi metrik bisnis 'persentase luas kerusakan' memerlukan validasi ulang | Cukup: IDF1, HOTA, dan MOTA tersedia, namun metrik ini kurang khas untuk argumen keselamatan yang dituju proyek |
| **Biaya komputasi** | Low | Medium | High |
| **Risiko orisinalitas** | Sedang: topik deteksi malam sudah banyak dikaji, namun kontribusi spesifik augmentasi pencahayaan adaptif (CLAHE kanal-L vs Gamma) untuk Recall pejalan kaki masih relatif baru | Sedang: banyak riset segmentasi kerusakan kendaraan yang sudah publik, sehingga temuan orisinal sulit dibedakan | Tinggi: topik pelacakan bawah air sudah memiliki banyak karya terbit dengan dataset video sendiri, sehingga pembedaan kontribusi menjadi sulit |
| **Rasional penilaian** | Dataset legal dan siap pakai, baseline melimpah dan metrik standar, biaya komputasi rendah, serta memiliki dampak keselamatan yang jelas sehingga mudah dikaitkan dengan ADAS. | Nilai bisnisnya tinggi, tetapi biaya pengumpulan dan pelabelan data berada di luar jangkauan satu semester dan ruang lingkup menjadi terlalu luas untuk eksperimen terkontrol yang bersih. | Butuh koleksi video, pemrosesan frame yang berat, dan metrik yang kurang terhubung dengan rumusan masalah keselamatan, sehingga merupakan pilihan paling berisiko dari sisi kelayakan satu semester. |

## 3. Kriteria Evaluasi dan Bobot

Bobot ditentukan dari rubrik Panduan Proyek Akhir: kelayakan dataset, baseline, eksperimen terkontrol, dan evaluasi kuantitatif diberi bobot lebih besar daripada orisinalitas murni. Skala skor 1 (sangat lemah) hingga 5 (sangat kuat).

| Kode | Kriteria | Bobot | Penjelasan |
|:---:|---|:---:|---|
| **K1** | Kesesuaian tugas CV | 0.20 | Seberapa dekat topik dengan inti mata kuliah Computer Vision (deteksi objek, segmentasi, tracking) dan relevansinya dengan permasalahan nyata di lapangan. |
| **K2** | Ketersediaan dataset | 0.20 | Ketersediaan dataset publik/legal yang sudah terunduh, teranotasi, dan siap pakai tanpa pengumpulan data tambahan. |
| **K3** | Baseline yang realistis | 0.15 | Keberadaan metode pembanding sederhana yang wajar sehingga kontribusi tiap eksperimen dapat dinilai terhadap sebuah acuan. |
| **K4** | Kesesuaian metrik | 0.15 | Ketersediaan metrik evaluasi standar dan tidak ambigu yang menjadi ukuran keberhasilan eksperimen. |
| **K5** | Biaya komputasi | 0.10 | Kebutuhan GPU/CPU dan waktu pelatihan, serta kecocokannya dengan ketersediaan GPU gratis dan laptop mahasiswa. |
| **K6** | Kontribusi orisinal | 0.10 | Ruang kebaruan yang dapat ditawarkan dari topik yang umumnya sudah banyak dikaji oleh peneliti lain. |
| **K7** | Feasibility dalam 1 semester | 0.10 | Kesesuaian ruang lingkup dengan jadwal UTS-UAS satu semester, termasuk waktu pengujian dan penyusunan laporan. |
| | **Total bobot** | **1.00** | Jumlah bobot diverifikasi secara otomatis bernilai tepat 1,00 |

## 4. Matriks Skor Per Kandidat (1-5)

| Kode | Kriteria | Bobot | C1 | C2 | C3 |
|:---:|---|:---:|:---:|:---:|:---:|
| **K1** | Kesesuaian tugas CV | 0.20 | 5 | 4 | 4 |
| **K2** | Ketersediaan dataset | 0.20 | 5 | 2 | 1 |
| **K3** | Baseline yang realistis | 0.15 | 5 | 3 | 3 |
| **K4** | Kesesuaian metrik | 0.15 | 5 | 4 | 4 |
| **K5** | Biaya komputasi | 0.10 | 4 | 3 | 2 |
| **K6** | Kontribusi orisinal | 0.10 | 4 | 3 | 3 |
| **K7** | Feasibility dalam 1 semester | 0.10 | 5 | 2 | 2 |

## 5. Ranking Akhir (Skor Tertimbang)

Skor tertimbang = jumlah (skor kriteria x bobot kriteria), dengan skor maksimum yang dapat dicapai sebesar 5,00. Persentase dihitung terhadap skor maksimum tersebut.

| Peringkat | Kandidat | Skor Tertimbang (maks 5,00) | Persentase |
|:---:|---|:---:|:---:|
| **1** | **C1** - Deteksi Pejalan Kaki di Malam Hari (Night-time Pedestrian Detection) | **4.80** | **96.0%** |
| **2** | **C2** - Segmentasi Kerusakan Bodi Mobil (Goresan & Penyok) | **3.05** | **61.0%** |
| **3** | **C3** - Pelacakan Multi-Objek Bawah Air (Multi-Object Tracking saat Oklusi) | **2.75** | **55.0%** |

> Selisih skor peringkat 1 terhadap peringkat 2 adalah **1.75 poin** (35.0 poin persentase).

## 6. Keputusan

**Topik yang dipilih adalah Kandidat C1 - Deteksi Pejalan Kaki di Malam Hari (Night-time Pedestrian Detection).**

Alasan keputusan berdasarkan hasil matriks tertimbang:

1. **Skor tertinggi dengan selisih signifikan.** C1 memperoleh skor tertimbang **4.80 / 5,00** atau **96.0%**, melebihi kandidat peringkat kedua (C2, 3.05 / 5,00).
2. **Dataset legal dan siap pakai.** Roboflow Low-light v2 (2.400 citra, 7.777 anotasi bounding box, lisensi CC BY 4.0) telah diunduh dan diaudit sehingga tidak ada waktu tunggu pengumpulan data.
3. **Baseline yang ada dan realistis.** Bobot COCO YOLOv8n (zero-shot) dan YOLOv8n fine-tuned menyediakan dua titik pembanding yang sah bagi eksperimen terkontrol.
4. **Metrik evaluasi tidak ambigu.** Recall, Precision, mAP@50, mAP@50:95, dan FPS adalah standar industri deteksi objek, sehingga hasil dapat diperbandingkan secara terbuka.
5. **Biaya komputasi rendah.** Arsitektur YOLOv8n hanya memiliki sekitar 3,2 juta parameter sehingga seluruh eksperimen dapat diselesaikan pada satu GPU T4 gratis.
6. **Relevansi keselamatan yang kuat.** Fokus pada Recall pejalan kaki pada malam hari menghubungkan proyek ini dengan Intelligent Transportation Systems dan Advance Driver Assistance Systems.

## 7. Risiko dan Alternatif

Meskipun hanya ada satu jalur utama, proyek tetap memerlukan jalur cadangan apabila sewaktu-waktu ada hambatan teknis:

- **Risiko 1 - Ketidakseimbangan kelas yang ekstrem.** Objek `person` hanya 11,6% dari total anotasi (rasio 7,58 : 1) sehingga nilai mAP dapat menutupi kualitas deteksi pejalan kaki. **Mitigasi:** Recall per kelas menjadi metrik utama dan confusion matrix selalu dilampirkan.
- **Risiko 2 - Peningkatan Recall berbanding Precision.** Augmentasi pencahayaan mencerahkan noise sensor sehingga false positive dapat meningkat. **Mitigasi:** pelaporan Recall dan Precision berdampingan serta analisis kasus gagal.
- **Risiko 3 - Throughput CPU di bawah ambang real-time.** Latensi terukur sekitar 52-55 ms (18-19 FPS) masih sedikit di bawah ambang 20 FPS. **Mitigasi:** ekspor ONNX INT8 dan multithreading OpenCV; ambang FPS dinyatakan secara eksplisit pada proposal.
- **Risiko 4 - Ketergantungan pada satu partisi data.** **Mitigasi:** sliced cross-validation pada split validasi 200 citra agar kesimpulan tidak bergantung pada satu partisi.
- **Alternatif 1 - Kandidat C2 (segmentasi kerusakan kendaraan)** bila annotasi mask piksel siap dikumpulkan; memerlukan estimasi tambahan 4-6 minggu untuk pelabelan data.
- **Alternatif 2 - Kandidat C3 (pelacakan bawah air)** bila akses dataset MOT bawah air legal diperoleh; pipeline menjadi dua tahap (deteksi + tracking) sehingga biaya komputasi naik ke kategori High.
- **Alternatif 3 - Tetap pada C1 namun memperluas kontribusi** dengan eksperimen multi-slicing brightness, kuantifikasi INT8, dan uji pada video berjalan tanpa perlu mengganti topik.

## 8. Reproduksi

Matriks ini dihasilkan ulang secara deterministik oleh skrip berikut:

```bash
python src/build_candidate_matrix.py
```

Struktur terhitung versi mesin disimpan pada `results/proposal/candidate_matrix.json`.

---

*(Dokumen ini dihasilkan secara otomatis oleh `src/build_candidate_matrix.py` untuk pemenuhan Checkpoint CP01.)*
