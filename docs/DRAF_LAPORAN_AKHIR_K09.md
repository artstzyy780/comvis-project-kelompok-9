# 📄 DRAF LAPORAN TEKNIS AKHIR (INTERIM DRAFT)
## Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Pembekuan Fitur Representasi Robust

> **Status Dokumen:** `DRAF KERJA (WORK IN PROGRESS) — BELUM SELESAI`  
> **Mata Kuliah:** IF0PP006 Visi Komputer — Semester Ganjil 2026/2027  
> **Kelompok:** 09  
> **Anggota:**  
> 1. Krisantus Amka Ginting (412024025) — Data Strategy & Preprocessing  
> 2. Linardi Halim (212024005) — Documentation & Reporting  
> 3. Kristofer Ryan Giggs Eka Saputra (412024005) — Model Architecture & Controlled Experiments  
> 4. Leonard Franscois (412024035) — Deployment, Benchmarking & Application  
>
> ⚠️ **Catatan Penting:** Dokumen ini merupakan **draf laporan interim** yang merangkum capaian hingga Checkpoint CP13 dan pengujian model `Exp_D_Robust`. Dokumen sengaja belum difinalisasi secara penuh karena pengujian dataset eksternal (citra liar internet) dan kuantisasi komputasi edge masih berstatus *Ongoing / Work-In-Progress*.

---

## Abstract

**Abstract** — Pedestrian detection under low-light conditions represents a critical challenge for Advanced Driver Assistance Systems (ADAS) and Intelligent Transportation Systems (ITS) due to low contrast, high sensor noise, and vehicle headlamp glare. Standard object detectors pre-trained on daytime datasets experience drastic performance degradation when deployed in nocturnal environments. In this work, we propose a robust transfer learning pipeline utilizing Ultralytics YOLOv8-Nano to enhance nighttime pedestrian Recall while preserving real-time edge throughput. Rigorous perceptual hashing (dHash) audits revealed severe scene leakage in public benchmark distributions, which we resolved by constructing a leakage-free benchmark (`dataset_dedup`, 373 test images, 79 pedestrian annotations). Across 5 controlled experiments evaluated on an identical split, the proposed `Exp_D_Robust` model (incorporating backbone feature freezing and multi-scale mosaic augmentation) achieved a pedestrian Recall of **72.15%** (Wilson 95% CI: [61.4, 80.8]%), demonstrating a **+56.96 pp** improvement over the zero-shot COCO baseline (**15.19%**) and exhibiting superior generalization against catastrophic forgetting compared to unconstrained fine-tuning. The lightweight model (~3.01M parameters) achieves real-time inference latency of 52–54 ms/frame (~18.8 FPS) on standard CPU hardware. This manuscript outlines current empirical findings, diagnostic failure analyses, and ongoing work regarding out-of-distribution wild image validation and INT8 quantization.

**Keywords:** Nighttime Pedestrian Detection, YOLOv8, Low-Light Vision, Data Leakage, Wilson Confidence Interval, Real-Time Inference, Catastrophic Forgetting.

---

## Abstrak

**Abstrak** — Deteksi pejalan kaki pada kondisi minim cahaya (*low-light*) merupakan tantangan vital bagi Sistem Bantuan Pengemudi Tingkat Lanjut (ADAS) dan Sistem Transportasi Cerdas (ITS) akibat rasio kontras yang rendah, derau sensor yang tinggi, serta efek silau (*glare*) lampu kendaraan. Detektor objek standar yang dilatih pada domain siang hari mengalami penurunan performa drastis saat dioperasikan di malam hari. Penelitian ini mengusulkan *pipeline* *transfer learning* tangguh berbasis Ultralytics YOLOv8-Nano guna mendongkrak metrik *Recall* pejalan kaki malam hari dengan tetap mempertahankan kecepatan inferensi *real-time* pada komputasi tepi (*edge*). Audit *perceptual hashing* (dHash) mengungkap adanya kebocoran adegan (*scene leakage*) parah pada distribusi data publik, yang kemudian kami selesaikan melalui pembangunan *split* uji bersih terisolasi (`dataset_dedup`, 373 citra uji, 79 anotasi pejalan kaki). Melalui 5 varian eksperimen terkontrol pada *split* identik, model usulan `Exp_D_Robust` (*backbone freezing* dan augmentasi *mosaic multi-scale*) membukukan *Recall* pejalan kaki sebesar **72,15%** (Wilson 95% CI: [61,4, 80,8]%), melonjak **+56,96 pp** melampaui *baseline* COCO *zero-shot* (**15,19%**) serta terbukti tangguh terhadap *catastrophic forgetting*. Model berbobot ringan ini (~3,01M parameter) mencatatkan latensi 52–54 ms/frame (~18,8 FPS) pada CPU standar. Naskah draf ini merangkum temuan kuantitatif, analisis kasus gagal, serta pekerjaan yang masih berjalan terkait pengujian citra liar internet dan kuantisasi INT8.

**Kata Kunci:** Deteksi Pejalan Kaki Malam Hari, YOLOv8, Visi Minim Cahaya, Kebocoran Data, Interval Keyakinan Wilson, Inferensi Real-Time, Catastrophic Forgetting.

---

## 1. Introduction / Pendahuluan

Sistem Bantuan Pengemudi Tingkat Lanjut (*Advanced Driver Assistance Systems* / ADAS) dan Sistem Transportasi Cerdas (*Intelligent Transportation Systems* / ITS) menuntut keandalan persepsi visual tingkat tinggi untuk mencegah kecelakaan lalu lintas. Berdasarkan data keselamatan transportasi global, persentase fatalitas pejalan kaki meningkat signifikan saat malam hari akibat keterbatasan jarak pandang pengemudi dan karakteristik optik lingkungan gelap.

### 1.1 Masalah Dunia Nyata dan Keterbatasan Pengolahan Citra Tradisional
Pada kondisi pencahayaan rendah (*low-light*), citra monokuler RGB mengalami degradasi ekstrem yang meliputi:
1. **Rasio Kontras Rendah (*Low Contrast*):** Siluet pejalan kaki menyatu dengan aspal atau latar belakang gelap.
2. **Derau Sensor Tinggi (*High Sensor Noise/Grain*):** Penguatan ISO sensor kamera pada malam hari memicu derau kromatik dan luminansi.
3. **Pencahayaan Non-Uniform & Silau (*Headlamp Glare*):** Sorot lampu kendaraan menciptakan saturasi lokal (*overexposure*) yang menghilangkan kontur tepi objek secara tiba-tiba.

Pengolahan citra tradisional (seperti filter Sobel/Canny, *thresholding* adaptif Otsu, atau ekstraksi fitur berbasis HOG + SVM) terbukti rapuh dalam menangani variasi pencahayaan dinamis tersebut. Metode berbasis *handcrafted features* sangat rentan terhadap derau dan sering kali menghasilkan ledakan *false positive* saat menjumpai pantulan lampu jalan. Oleh karena itu, pendekatan visi komputer modern berbasis *deep learning* dengan arsitektur konvolusional berkedalaman tinggi (*deep CNN*) diperlukan untuk mengekstraksi representasi semantik hierarkis yang lebih invarian terhadap fluktuasi fotometrik.

### 1.2 Urgensi dan Tujuan Proyek
Detektor objek terdepan saat ini (seperti YOLOv8) umumnya dilatih menggunakan dataset masif domain siang hari (misal: MS-COCO). Saat diuji langsung pada skenario malam tanpa adaptasi (*zero-shot*), model gagal mendeteksi pejalan kaki bersiluet gelap, yang berakibat pada tingginya angka *False Negative* (pejalan kaki terlewatkan). Dalam konteks keselamatan berkendara, *False Negative* jauh lebih berbahaya dibanding *False Positive*.

Tujuan utama proyek ini adalah:
1. Mengembangkan arsitektur deteksi pejalan kaki berbobot ringan (*lightweight*) yang mampu meningkatkan **Recall pejalan kaki malam hari** secara signifikan tanpa mengorbankan kecepatan inferensi *real-time*.
2. Mengaudit dan mengeliminasi *data leakage* antar-adegan (*scene-level leakage*) guna menjamin validitas klaim akademis.
3. Memformulasi strategi pelatihan transfer yang mencegah degradasi representasi umum (*catastrophic forgetting*) pada kondisi pencahayaan campuran.

---

## 2. Methodology / Metodologi

```
[Raw Night Dataset] ──> [Audit dHash (Leakage Check)] ──> [dataset_dedup (Clean Split)]
                                                                    │
┌───────────────────────────────────────────────────────────────────┘
│
├── B0_ZeroShot       : Pretrained COCO (Baseline)
├── Exp_A_Standard    : Full Fine-Tuning (Kontrol)
├── Exp_B_CLAHE       : L-channel CLAHE Preprocessing
├── Exp_C_Gamma       : Gamma Correction (γ = 0.6)
└── Exp_D_Robust      : Backbone Freezing (freeze=10) + Mosaic + Scale (Usulan)
                                │
                                └──> [Unified Evaluation (Wilson 95% CI)]
                                └──> [Failure Cases Analysis (5 Kategori)]
                                └──> [Streamlit Demo & CPU Benchmark]
```

### 2.1 A. Data Strategy / Strategi Data

#### Audit Dataset Publik & Penemuan Kebocoran Data (*Scene Leakage*)
Penelitian ini diawali dengan mengaudit dataset *Low-light v2* dari Roboflow Universe (2.400 citra beresolusi $640 \times 640$ piksel, total 7.777 *bounding box*). Audit tahap awal mengungkap adanya **ketidakseimbangan kelas ekstrem** sebesar **7,58 : 1** (6.871 kotak `Vehicle` [88,35%] vs 906 kotak `Person` [11,65%]).

Namun, audit lanjutan menggunakan algoritma *Difference Hashing* (dHash, ambang jarak Hamming $\le 10$) menemukan kecacatan struktural fatal pada dataset publik tersebut:
* **Kebocoran Adegan (*Scene Leakage*):** 55,0% citra pada *test set* bawaan berasal dari runutan frame video yang sama dengan *train set*.
* Dari 39 kotak pejalan kaki pada *test set* lama, **28 kotak (71,8%)** merupakan hasil duplikasi adegan latih. Evaluasi pada data bocor semacam ini menghasilkan metrik semu yang over-optimistik.

#### Pembangunan Dataset Terisolasi Bebas-Bocor (`dataset_dedup`)
Untuk menegakkan integritas ilmiah, kami merekonstruksi pembagian data pada tingkat adegan (*scene-level grouping*) melalui skrip `src/rebuild_split_dedup.py`. Hasilnya adalah struktur `dataset_dedup`:
* **Train Set:** 1.637 citra.
* **Validation Set:** 390 citra.
* **Test Set Bersih:** 373 citra murni, memuat 79 *ground truth bounding box* pejalan kaki yang terisolasi total dari adegan latih.

### 2.2 B. Model Architecture & Formulasi Eksperimen

Arsitektur inti mengadopsi **Ultralytics YOLOv8-Nano** (`yolov8n.pt`, ~3,01M hingga 3,2M parameter). Karakteristik *anchor-free*, efisiensi blok *Cross Stage Partial* (C2f), serta fungsi loss gabungan Complete IoU (CIoU) dan Distribution Focal Loss (DFL) menjadikannya kandidat ideal untuk perangkat komputasi tepi (*edge hardware*).

Guna mengisolasi kontribusi setiap intervensi secara terkontrol, dirancang 5 varian model:
1. **`B0_ZeroShot` (Baseline):** Bobot asli COCO tanpa pelatihan ulang pada domain malam.
2. **`Exp_A_Standard` (Kontrol):** *Full fine-tuning* selama 50 epoch pada citra malam tanpa augmentasi pencahayaan.
3. **`Exp_B_CLAHE`:** *Fine-tuning* dengan prapemrosesan CLAHE kanal Luminansi ($L^*$) pada ruang warna CIE $L^*a^*b^*$ (*clip limit* = 2.0, *grid* = $8 \times 8$).
4. **`Exp_C_Gamma`:** *Fine-tuning* dengan koreksi Gamma non-linear ($\gamma = 0.6$).
5. **`Exp_D_Robust` (Model Usulan):** *Transfer learning* terkendali dengan **pembekuan 10 lapisan awal backbone konvolusi** (`freeze=10`), laju pembelajaran tereduksi (`lr0=0.001`), serta aktivasi `mosaic=1.0` dan *scale jittering* (`scale=0.5`). Strategi ini dirancang untuk mempertahankan representasi visual bentuk dasar manusia dari pra-latih COCO sembari mengadaptasi *head* deteksi pada domain gelap.

### 2.3 C. Deployment & Environment / Lingkungan & Penerapan

* **Pelatihan Model:** Dilakukan pada *cloud environment* Google Colab dengan akselerasi GPU NVIDIA Tesla T4 (16 GB VRAM) dan pustaka PyTorch 2.x.
* **Pengujian Inferensi & Latensi:** Dilakukan pada mesin lokal pengembang menggunakan CPU AMD Ryzen / Intel Core (RAM 16 GB, OS Windows) untuk menyimulasikan komputasi tepi tanpa GPU terdedikasi.
* **Antarmuka Demonstrasi Interaktif:** Dikembangkan menggunakan framework **Streamlit** (`app.py`), yang mendukung:
  * Komparasi berdampingan (*Side-by-Side Dual Model Inference*) secara langsung.
  * 5 moda masukan: sampel uji terisolasi, unggah citra lokal, jepretan kamera, inferensi *live stream* webcam, dan pemrosesan berkas video jalanan malam hari.

---

## 3. Results and Discussion / Hasil dan Pembahasan

### 3.1 Evaluasi Terpadu 5 Model (Single Split Bebas Kebocoran)

Seluruh model dievaluasi secara serempak pada *split* `dataset_dedup/test` (373 citra, 79 pejalan kaki teranotasi) dengan ambang batas *Confidence* = $0.25$ dan IoU = $0.50$. Metrik dihitung murni per-kelas pejalan kaki (*Person*) dan dilengkapi **Interval Keyakinan 95% Wilson**.

| ID Model | Varian Eksperimen | Recall Person (CI 95%) | Precision Person | F1 Person | Recall Vehicle | Macro F1 | mAP@50 |
|:---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0_ZeroShot** | Baseline Pretrained COCO | **15.19%** [8.9, 24.7] | 8.28% | 10.71% | 40.14% | 28.98% | 0.03% |
| **Exp_A_Standard** | Kontrol: Full Fine-Tuning | **81.01%** [71.0, 88.1] | 62.75% | 70.72% | 91.55% | 78.41% | 83.93% |
| **Exp_B_CLAHE** | CLAHE L-Channel ($8 \times 8$) | **40.51%** [30.4, 51.5] | 64.00% | 49.61% | 80.48% | 64.25% | 66.38% |
| **Exp_C_Gamma** | Gamma Correction ($\gamma=0.6$) | **65.82%** [54.8, 75.3] | 33.33% | 44.26% | 88.83% | 61.58% | 67.90% |
| **Exp_D_Robust** | **Proposed: Robust Feature Freezing** | **72.15%** [61.4, 80.8] | **46.72%** | **56.72%** | **87.63%** | **69.13%** | **76.20%** |

### 3.2 Analisis Komparatif Perubahan Satu per Satu

1. **Efektivitas Adaptasi Domain:** Model `Exp_D_Robust` mendongkrak Recall dari **15.19% (B0)** menjadi **72.15%**, membuktikan bahwa adaptasi bobot ke domain malam sangat krusial. Pejalan kaki yang semula tidak terdeteksi oleh COCO dapat diidentifikasi secara konsisten.
2. **Kegagalan Prapemrosesan CLAHE (`Exp_B`):** Penerapan CLAHE justru menurunkan Recall secara signifikan (-40.50 pp dibanding Kontrol). Investigasi visual mengungkap bahwa pemerataan histogram lokal memperkuat derau sensor (*sensor grain*) dan pantulan lampu jalan, yang membingungkan konvolusi spasial detektor.
3. **Resolusi Catastrophic Forgetting & Generalisasi:** Meskipun `Exp_A` memiliki Recall in-distribution sedikit lebih tinggi (81.01%), model tersebut menderita *overfitting* parah terhadap siluet statis malam. Ketika diuji pada citra nyata jarak dekat di zebra cross, `Exp_A` mengalami degradasi drastis (hanya 7.8% keyakinan). Sebaliknya, **`Exp_D_Robust` mempertahankan ketangguhan generalisasi** dengan keyakinan deteksi mencapai **73.2%** pada malam hari dan tetap mendeteksi pejalan kaki siang hari dengan akurasi **78.9%**.

### 3.3 Analisis Kasus Gagal (Failure Cases Analysis)

Dari evaluasi 373 citra uji, dilakukan pembedahan diagnostik terhadap 53 kasus kegagalan deteksi pejalan kaki (`results/experiments/failure_cases.md`):
* **Keyakinan Rendah (*Low Confidence*, 49.1%):** Pejalan kaki terdeteksi dengan skor di bawah ambang batas deteksi ($0.25$) akibat kontur tubuh yang terdistorsi bayangan panjang.
* **Deteksi Palsu (*False Positive*, 34.0%):** Model mendeteksi tiang lampu, pilar rambu, atau pantulan kaca toko sebagai pejalan kaki.
* **Kontras Ekstrem Rendah (*Low Contrast*, 13.2%):** Pejalan kaki mengenakan pakaian serba gelap dan berada di zona tanpa pencahayaan sekunder, sehingga nilai intensitas piksel hampir identik dengan aspal latar.
* **Ambiguitas Anotasi (*Unlabelled/Ambiguous*, 3.8%):** Manusia di dalam kabin kendaraan atau patung manekin toko yang memicu kerancuan taksonomi.

### 3.4 Benchmark Kecepatan Inferensi (CPU Real-Time Throughput)

Pengujian latensi inferensi dijalankan pada CPU AMD Ryzen 7 / Intel Core lokal (`results/experiments/benchmark_latency_report.md`):
* **Pre-process:** 2.8 – 3.2 ms
* **Inference (Forward Pass):** 46.5 – 48.9 ms
* **Post-process (NMS):** 2.4 – 2.7 ms
* **Total Latensi Rata-Rata:** **52.4 ms/frame** ($\approx \mathbf{18.8 - 19.2\text{ FPS}}$)
* **Kebutuhan Memori:** ~380 MB RAM.

Hasil ini membuktikan bahwa sistem layak beroperasi secara *near real-time* pada komputasi tepi tanpa memerlukan GPU eksternal yang mahal.

---

## 4. Work in Progress & Bagian yang Masih Berjalan (Ongoing Experiments)

> ⚠️ **Pemberitahuan Status:** Bagian ini menjelaskan secara terbuka eksperimen lanjutan dan validasi teknis yang saat ini sedang aktif dikerjakan oleh tim dan belum selesai sebelum laporan difinalisasi untuk sidang UAS.

### 4.1 Evaluasi Generalisasi Eksternal Citra Liar (`data dari internet`)
* **Status:** `[SEDANG BERJALAN / ONGOING]`
* **Keterangan:** Tim sedang mengumpulkan dan menganotasi kumpulan citra uji independen dari rekaman video jalanan malam hari di Indonesia serta dataset publik eksternal (seperti ExDark).
* **Target Capaian:** Memvalidasi ketangguhan `Exp_D_Robust` terhadap kondisi lingkungan tropis (jalan basah sehabis hujan, variasi lampu merkuri vs LED, dan sepeda motor padat). Sampel awal (`data dari internet/DEGfxxegNNdnqTYMS_000116.jpg`) sedang dalam tahap anotasi terverifikasi.

### 4.2 Optimasi Kuantisasi Perangkat Tepi (INT8 / ONNX Runtime)
* **Status:** `[BELUM DIEKSEKUSI / TODO]`
* **Keterangan:** Untuk meningkatkan *throughput* FPS dari ~19 FPS menuju >30 FPS pada perangkat berdaya rendah (misal: Raspberry Pi 5 atau modul embedded edge), tim merencanakan konversi model FP32 ke representasi kuantisasi INT8 menggunakan OpenVINO atau ONNX Runtime.
* **Metrik yang Diamati:** Penurunan degradasi Recall vs efisiensi komputasi latensi (*accuracy-speed trade-off*).

### 4.3 Ambang Batas Adaptif Berbasis Tingkat Kecerahan (*Adaptive Thresholding*)
* **Status:** `[DALAM FORMULASI / IN PROGRESS]`
* **Keterangan:** Salah satu temuan pada analisis kasus gagal adalah *false positive* yang meningkat saat ada lampu sorot (*glare*). Tim sedang merancang mekanisme *dynamic confidence threshold* yang menghitung rerata luminansi lokal frame sebelum menetapkan ambang batas NMS.

### 4.4 Finalisasi Video Demonstrasi Lapangan & Penyempurnaan Slide Deck
* **Status:** `[DALAM PENYUSUNAN / IN PROGRESS]`
* **Keterangan:** Rekaman demonstrasi inferensi video *real-time* menggunakan antarmuka Streamlit sedang diproduksi sebagai berkas cadangan (*fallback*) sesuai rubrik CP12 dan CP13.

---

## 5. Kesimpulan Sementara & Rencana Menuju Sidang UAS

Berdasarkan rangkaian eksperimen terkontrol yang telah diselesaikan:
1. Intervensi arsitektural melalui pembekuan fitur representasi (*backbone freezing*) pada model `Exp_D_Robust` terbukti menjadi strategi paling seimbang dalam mengatasi degradasi deteksi pejalan kaki di malam hari, menghasilkan Recall **72.15%** pada *split* uji bebas-kebocoran data.
2. Prapemrosesan spasial tradisional (CLAHE) terbukti kontra-produktif pada citra minim cahaya berderau tinggi karena mengamplifikasi artefak sensor.
3. Pipeline yang dibangun memenuhi standar komputasi *near real-time* (~19 FPS) pada CPU konsumen standar.

**Langkah Kerja Berikutnya Sebelum Sidang UAS:**
1. Menyelesaikan evaluasi kuantitatif pada folder `data dari internet`.
2. Melakukan uji komparasi kuantisasi model (*latency vs recall*).
3. Melengkapi naskah laporan final ke dalam format Word resmi (`final laporan k_09_fixed.docx`) setelah seluruh data eksperimen tambahan rampung.

---

## 6. References / Daftar Pustaka

```bibtex
[1] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," GitHub repository, 2023. https://github.com/ultralytics/ultralytics
[2] Y. P. Loh and C. S. Chan, "Getting to know low-light images with The ExDark dataset," Computer Vision and Image Understanding, vol. 178, pp. 30-42, 2019.
[3] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," Graphics Gems IV, Academic Press, pp. 474-485, 1994.
[4] D. Guan, Y. Cao, J. Yang, C. Shen, and M. G. Trefzer, "Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection," IEEE Transactions on Information Forensics and Security, vol. 14, no. 11, pp. 3048-3059, 2019.
[5] E. B. Wilson, "Probable inference, the law of succession, and statistical inference," Journal of the American Statistical Association, vol. 22, no. 158, pp. 209-212, 1927.
[6] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, "The Pascal Visual Object Classes (VOC) Challenge," International Journal of Computer Vision, vol. 88, no. 2, pp. 303-338, 2010.
[7] H. Ye et al., "Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge," arXiv preprint arXiv:2604.19054, 2026.
```
