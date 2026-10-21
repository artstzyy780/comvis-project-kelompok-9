# Daftar Rujukan Ilmiah - Proyek Visi Komputer

> **Checkpoint:** CP02 - Problem statement, pengguna atau konteks, 5 rujukan, dan ringkasan relevansi (6 XP)  
> **Mata Kuliah:** Visi Komputer (IF0PP006) - Semester Ganjil 2026/2027  
> **Kelompok:** Kelompok 9 (Krisantus Amka Ginting, Linardi Halim, Kristofer Ryan Giggs Eka Saputra, Leonard Franscois)  
> **Topik:** Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif  
> **Tanggal:** 20 Oktober 2026  
> **Catatan:** daftar rujukan ini menjadi sumber kebenaran tunggal (single source of truth) dan dipakai oleh skrip `src/collect_references.py`.

---

## 1. Daftar Rujukan (Gaya IEEE)

[1] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," *GitHub repository*, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics
[2] Y. P. Loh and C. S. Chan, "Getting to know low-light images with the ExDark dataset," *Computer Vision and Image Understanding*, vol. 178, pp. 30-42, 2019. [Online]. Available: https://doi.org/10.1016/j.cviu.2018.10.010 [doi: 10.1016/j.cviu.2018.10.010]
[3] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," *Graphics Gems IV*, Academic Press, pp. 474-485, 1994. [Online]. Available: https://doi.org/10.1016/B978-0-12-336156-1.50061-6 [doi: 10.1016/B978-0-12-336156-1.50061-6]
[4] D. Guan, Y. Cao, J. Yang, C. Shen, and M. G. Trefzer, "Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection," *IEEE Transactions on Information Forensics and Security*, vol. 14, no. 11, pp. 3048-3059, 2019. [Online]. Available: https://arxiv.org/abs/1802.09972 [doi: 10.48550/arXiv.1802.09972]
[5] Z. Ye, Y.-H. Lu, X. Hu, S. Zhang, T. Jing, X. Li, and others, "Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge," *arXiv preprint arXiv:2604.19054*, 2026. [Online]. Available: https://arxiv.org/abs/2604.19054 [doi: 10.48550/arXiv.2604.19054]

## 2. Rujukan dan Komponen Proyek yang Didukung

| # | Rujukan | Jenis | Komponen Proyek yang Didukung |
|:---:|---|---|---|
| **1** | G. Jocher, A. Chaurasia, and J. Qiu ("Ultralytics YOLOv8", 2023) | Perangkat lunak / repositori | Detektor utama YOLOv8n (3,2 juta parameter), baseline zero-shot B0, pipeline evaluasi mAP/Recall/Precision |
| **2** | Y. P. Loh and C. S. Chan ("Getting to know low-light images with the ExDark dataset", 2019) | Paper dataset | Argumentasi masalah degradasi deteksi pada kondisi minim cahaya, klasifikasi kondisi pencahayaan untuk slicing brightness |
| **3** | K. Zuiderveld ("Contrast Limited Adaptive Histogram Equalization", 1994) | Paper metode | Implementasi CLAHE pada kanal L ruang CIE L*a*b* di src/preprocessing.py (clip limit 2,0; tile grid 8x8) |
| **4** | D. Guan, Y. Cao, J. Yang, C. Shen, and M. G. Trefzer ("Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection", 2019) | Paper metode | Argumen teoretis bahwa informasi pencahayaan dapat dieksploitasi untuk meningkatkan deteksi pejalan kaki; desain pipeline dua tahap |
| **5** | Z. Ye, Y.-H. Lu, X. Hu, S. Zhang, T. Jing, X. Li, and others ("Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge", 2026) | Review tantangan / challenge | Justifikasi ambang target 20 FPS, metodologi pengukuran latensi dan throughput, pelaporan metrik per kelas secara terbuka |

## 3. Ringkasan Relevansi per Rujukan

Setiap rujukan menjelaskan secara spesifik keputusan desain proyek mana yang disandarkan oleh sumber tersebut.

### [1] G. Jocher, A. Chaurasia, and J. Qiu - "Ultralytics YOLOv8" (2023)

- **Venue:** GitHub repository
- **Jenis rujukan:** Perangkat lunak / repositori
- **Tautan:** https://github.com/ultralytics/ultralytics

Repositori resmi YOLOv8 menjadi justifikasi formal pemilihan arsitektur YOLOv8-Nano sebagai detektor utama proyek ini. Head anchor-free yang memprediksi titik pusat dan dimensi bounding box secara langsung sesuai dengan rasio aspek pejalan kaki yang bervariasi, sementara ukuran bobot sekitar 3,2 juta parameter menjadikan ambang 20 FPS pada CPU laptop tetap realistis. Framework ini juga menyediakan perhitungan metrik mAP, Recall, dan Precision yang dipakai pada skrip evaluasi dan benchmarking, sehingga angka yang dilaporkan konsisten di seluruh eksperimen. Bobot COCO miliknya dipakai langsung sebagai baseline zero-shot B0 tanpa pengumpulan data tambahan.

### [2] Y. P. Loh and C. S. Chan - "Getting to know low-light images with the ExDark dataset" (2019)

- **Venue:** Computer Vision and Image Understanding, vol. 178, pp. 30-42
- **Jenis rujukan:** Paper dataset
- **Tautan:** https://doi.org/10.1016/j.cviu.2018.10.010

Paper ini mendokumentasikan dataset ExDark berisi 7.363 citra low-light dengan 12 kelas dan anotasi bounding box, sehingga membuktikan bahwa masalah minim cahaya memiliki benchmark publik dan bukan sekadar fenomena visual. Eksperimen pada paper menunjukkan degradasi deteksi objek pada citra low-light dibandingkan MS COCO; temuan inilah yang menjadi landasan argumentasi bahwa model pretrained siang hari tidak dapat dipakai langsung pada citra malam. Klasifikasi kondisi pencahayaan yang dipublikasikan paper (twilight, shadow, dark, dan variasinya) mengilhami pembagian citra menjadi slice brightness untuk analisis kasus gagal pada eksperimen kami. Paper ini juga menjadi contoh akademik bahwa dataset low-light seharusnya dipecah secara bersih agar tidak terjadi kebocoran data antar partisi.

### [3] K. Zuiderveld - "Contrast Limited Adaptive Histogram Equalization" (1994)

- **Venue:** Graphics Gems IV, Academic Press, pp. 474-485
- **Jenis rujukan:** Paper metode
- **Tautan:** https://doi.org/10.1016/B978-0-12-336156-1.50061-6

Bab ini merupakan sumber primer algoritma CLAHE yang diimplementasikan pada modul prapemrosesan proyek, sehingga parameter clip limit 2,0 dan tile grid 8x8 yang dipakai pada eksperimen Exp-B memiliki dasar teoretis yang jelas. Pembatas kontras yang diperkenalkan algoritma ini menjaga agar histogram lokal tidak menjadi over-amplified, yang persis menyelesaikan masalah derau sensor pada citra malam. CLAHE diterapkan hanya pada kanal L di ruang warna CIE L*a*b* supaya krominansi tidak berubah dan tidak memunculkan color cast, sementara pemrosesan hanya dilakukan pada citra latih untuk mencegah kebocoran data.

### [4] D. Guan, Y. Cao, J. Yang, C. Shen, and M. G. Trefzer - "Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection" (2019)

- **Venue:** IEEE Transactions on Information Forensics and Security, vol. 14, no. 11, pp. 3048-3059
- **Jenis rujukan:** Paper metode
- **Tautan:** https://arxiv.org/abs/1802.09972

Paper ini menunjukkan secara eksplisit bahwa informasi pencahayaan dapat dimanfaatkan untuk meningkatkan deteksi pejalan kaki pada kondisi siang maupun malam, yang menjadi dasar argumentasi bahwa masalah low-light memang bisa ditangani di lapangan pemrosesan citra dan bukan murni keterbatasan benchmark. Mekanisme illumination-aware weighting yang diperkenalkan mengilhami desain pipeline proyek ini, yaitu augmentasi pencahayaan diperlakukan sebagai bagian dari pipeline secara langsung, bukan sebagai langkah yang dijalankan setelah training. Pendekungan multi-task pada paper juga menjadi alasan proyek memilih evaluasi per kelas (Recall pejalan kaki) di samping mAP gabungan, agar kualitas deteksi tiap kelas tidak tertutup oleh kelas dominan.

### [5] Z. Ye, Y.-H. Lu, X. Hu, S. Zhang, T. Jing, X. Li, and others - "Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge" (2026)

- **Venue:** arXiv preprint arXiv:2604.19054
- **Jenis rujukan:** Review tantangan / challenge
- **Tautan:** https://arxiv.org/abs/2604.19054

Evaluasi solusi pemenang LPCVC 2025 menunjukkan bahwa pada perangkat dengan sumber daya terbatas, tradeoff antara akurasi dan latensi menjadi kriteria utama, sehingga ambang 20 FPS yang kami nyatakan pada proposal bukan angka sembarang melainkan target yang masuk akal untuk inferensi edge. Kerangka benchmarking competition tersebut memakai protokol pengukuran latensi dan FPS yang konsisten, dan pendekatan yang sama diadopsi pada skrip benchmarking inferensi lokal proyek ini. Paper ini juga menegaskan bahwa penilaian solusi terbaik harus dilakukan dengan metrik yang dilaporkan secara terbuka, yang menjadi alasan kami mencantumkan Recall dan Precision per kelas bukan hanya mAP gabungan. Selain itu, tantangan LPCVC ini menjadi bukti terbaru bahwa isu efisiensi komputasi masih aktif dibahas pada komunitas computer vision, sehingga arah riset proyek tetap relevan.

## 4. Uji Cakupan terhadap Kriteria Rubrik

Setiap komponen rubrik wajib didukung oleh minimal satu rujukan.

| Komponen Rubrik | Rujukan pendukung | Jumlah | Status |
|:---|:---|:---:|:---:|
| Pemahaman masalah | [2], [4], [5] | 3 | TERPENUHI |
| Dataset | [2] | 1 | TERPENUHI |
| Metodologi | [1], [3], [4] | 3 | TERPENUHI |
| Baseline | [1], [4], [5] | 3 | TERPENUHI |
| Evaluasi | [1], [2], [5] | 3 | TERPENUHI |
| Rujukan (kredibilitas sumber) | [1], [2], [3], [4], [5] | 5 | TERPENUHI |

## 5. Uji Kepatuhan Rubrik Rujukan

| Uji | Ambang | Nilai Aktual | Status | Keterangan |
|:---|:---|:---|:---:|---|
| Jumlah sumber kredibel | >= 5 rujukan | 5 rujukan | **PASS** | Proyek memuat 5 rujukan ilmiah (ambang rubrik: 5). |
| Paper metode atau dataset | >= 2 paper metode/dataset | 3 paper (loh2019exdark, zuiderveld1994clahe, guan2019fusion) | **PASS** | Rujukan yang diklasifikasikan sebagai paper metode atau paper dataset: loh2019exdark, zuiderveld1994clahe, guan2019fusion. |
| Kelengkapan sitasi per komponen | 5 dari 5 komponen tercakup | 5 dari 5 | **PASS** | Seluruh lima komponen rubrik didukung oleh minimal satu rujukan. |

## 6. Status Verifikasi Online

Verifikasi daring dijalankan dengan `python src/collect_references.py --verify-online`. Rujukan yang gagal dijangkau tetap valid secara akademik; status hanya mencerminkan ketersediaan akses jaringan pada saat pemeriksaan.

| # | Rujukan | URL | Status |
|:---:|---|---|---|
| 1 | Ultralytics YOLOv8 | https://github.com/ultralytics/ultralytics | terjangkau (HTTP 200) |
| 2 | Getting to know low-light images with the ExDark dataset | https://doi.org/10.1016/j.cviu.2018.10.010 | terjangkau (HTTP 200) |
| 3 | Contrast Limited Adaptive Histogram Equalization | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 | terjangkau (HTTP 200) |
| 4 | Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection | https://arxiv.org/abs/1802.09972 | terjangkau (HTTP 200) |
| 5 | Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge | https://arxiv.org/abs/2604.19054 | terjangkau (HTTP 200) |

## 7. Reproduksi

```bash
python src/collect_references.py                 # mode luring (default)
python src/collect_references.py --verify-online  # tambahan verifikasi URL
```

Berkas `references.bib` di root proyek dihasilkan oleh skrip yang sama dan dapat dipakai langsung pada Overleaf, Zotero, atau Mendeley.

---

*(Dokumen ini dihasilkan secara otomatis oleh `src/collect_references.py` untuk pemenuhan Checkpoint CP02.)*
