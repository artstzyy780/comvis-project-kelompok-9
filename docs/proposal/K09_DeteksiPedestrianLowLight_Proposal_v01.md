# PROPOSAL TUGAS AKHIR VISI KOMPUTER (IF0PP006)
## Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif

**Kode Berkas:** `K09_DeteksiPedestrianLowLight_Proposal_v01`  
**Mata Kuliah:** Visi Komputer (*Computer Vision* — IF0PP006)  
**Kelas / Semester:** 5PITA / Ganjil 2026/2027  
**Fase:** Ujian Tengah Semester (Sidang Proposal UTS)  

---

### Identitas Kelompok 9

| Nama Mahasiswa | Peran & Tanggung Jawab Utama |
|---|---|
| **Krisantus Linardi** | *Data & Preprocessing Lead*: Manajemen dataset, audit integritas data, implementasi pipeline prapemrosesan/augmentasi pencahayaan (CLAHE & Gamma), perancangan metrik evaluasi kuantitatif. |
| **Kristofer Leonard** | *Model & Deployment Lead*: Konfigurasi arsitektur YOLOv8, eksekusi pelatihan baseline & eksperimen komparasi di Google Colab, analisis latensi/FPS, perancangan prototipe inferensi/demo. |

---

## BAB 1: Judul dan Identitas Proyek

* **Judul Proyek:**  
  *Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif*
* **Topik Utama:** Deteksi Objek Real-Time (*Real-Time Object Detection in Low-Light Scenarios*)
* **Model Detektor:** Ultralytics YOLOv8 (varian *lightweight* YOLOv8-Nano / YOLOv8-Small)
* **Teknik Intervensi:** *Contrast Limited Adaptive Histogram Equalization* (CLAHE) dan *Gamma Correction* (Pencahayaan Adaptif)
* **Repositori & Berkas:** `K09_DeteksiPedestrianLowLight`

---

## BAB 2: Latar Belakang dan Masalah

### 2.1 Konteks Masalah
Dalam ekosistem keselamatan transportasi cerdas (*Intelligent Transportation Systems* / ITS) dan sistem bantuan pengemudi tingkat lanjut (*Advanced Driver Assistance Systems* / ADAS), deteksi pejalan kaki (*pedestrian detection*) memegang peranan paling krusial untuk mencegah fatalitas tabrakan di jalan raya. Pejalan kaki dikategorikan sebagai *vulnerable road users* (VRU) yang tidak memiliki perlindungan fisik saat terjadi benturan.

Secara statistik lalu lintas, mayoritas kecelakaan fatal yang melibatkan pejalan kaki terjadi pada rentang waktu senja hingga malam hari. Ironisnya, sebagian besar model deteksi objek modern berbasis *deep learning* dilatih menggunakan korpus data visual standar siang hari (seperti MS COCO atau Pascal VOC) yang memiliki distribusi pencahayaan seragam, rasio kontras tajam, serta tingkat derau (*noise*) sensor yang rendah.

### 2.2 Tantangan Teknis Visi Komputer pada Kondisi *Low-Light*
Ketika model detektor diterapkan pada kondisi malam hari atau pencahayaan minim (*low-light conditions*), performa deteksi mengalami degradasi drastis akibat tiga karakteristik visual utama:
1. **Rasio Kontras Rendah (*Low Contrast*):** Siluet pejalan kaki menyatu dengan latar belakang gelap, bayangan bangunan, atau pepohonan, sehingga fitur tepi (*gradient features*) dan tekstur tubuh manusia sulit diekstraksi oleh *convolutional backbone*.
2. **Derau Sensor Tinggi (*High Sensor Noise / Grain*):** Pengambilan gambar pada intensitas foton rendah memicu peningkatan *gain* sensor kamera (ISO tinggi), memunculkan artefak bintik (*noise*) yang mengaburkan batas objek.
3. **Rentang Dinamis Ekstrem & Silau Lampu (*Glare & Extreme Dynamic Range*):** Keberadaan lampu sorot kendaraan (*headlights*) atau lampu penerangan jalan menciptakan area *over-exposure* lokal sekaligus mempergelap area sekitar (*shadow clipping*), mendistorsi distribusi spasial piksel.

### 2.3 Alasan Pemilihan Pendekatan Visi Komputer
Pendekatan deteksi alternatif seperti sensor LiDAR atau kamera termal inframerah gelombang panjang (*Long-Wave Infrared* / LWIR) memang tahan terhadap kegelapan, namun memiliki kendala biaya perangkat keras yang sangat mahal (ribuan dolar) serta komputasi pembacaan *point cloud* yang berat. Kamera RGB monokuler standar tetap menjadi opsi paling ekonomis, fleksibel, dan mudah diadopsi secara massal pada kendaraan maupun CCTV perkotaan. Oleh karena itu, solusi peningkatan performa harus difokuskan pada tingkat algoritma visi komputer: **memperkuat representasi visual citra gelap sebelum diproses oleh detektor deep learning**.

### 2.4 Batas Klaim (*Claim Boundary*)
Sesuai dengan panduan akademik proyek, kelompok menetapkan batasan klaim yang jujur dan terukur:
* Proyek ini **tidak mengklaim** menciptakan sistem deteksi ADAS otonom penuh yang siap produksi (*production-grade automotive system*), karena pengujian dilakukan pada lingkungan akademik terkontrol.
* Proyek ini **tidak mengklaim** arsitektur jaringan saraf tiruan baru dari nol (*novel deep learning backbone*), melainkan berfokus pada **studi empiris terkontrol mengenai efektivitas augmentasi peningkatan pencahayaan (CLAHE vs. Gamma Correction)** dalam mendongkrak metrik *Recall* pejalan kaki pada detektor real-time YOLOv8.
* Keberhasilan dinilai dari perbaikan metrik deteksi kuantitatif dan analisis kegagalan yang transparan pada split data uji (*test set*) yang valid.

---

## BAB 3: Tujuan, Pertanyaan Proyek, Batasan, dan Ukuran Keberhasilan

### 3.1 Tujuan Proyek
1. Mengembangkan pipeline deteksi objek pejalan kaki kondisi minim cahaya berbasis model efisien YOLOv8.
2. Mengintegrasikan teknik prapemrosesan adaptif lokal (CLAHE) dan pencerahan non-linear global (Gamma Correction) pada tahap pelatihan untuk meningkatkan daya diskriminasi fitur visual pejalan kaki.
3. Melakukan evaluasi kuantitatif terkontrol untuk membuktikan peningkatan performa deteksi, khususnya metrik *Recall*, tanpa mengorbankan kecepatan inferensi *real-time*.

### 3.2 Pertanyaan Proyek (*Research Questions*)
* **RQ1 (Baseline Degradation):** Seberapa jauh performa model deteksi dasar YOLOv8 *zero-shot* (pretrained MS COCO) mengalami kegagalan deteksi pejalan kaki (*false negatives*) pada dataset jalanan malam hari?
* **RQ2 (Augmentation Impact):** Apakah penambahan augmentasi pencahayaan adaptif (CLAHE atau Gamma Correction) secara signifikan meningkatkan *Recall* kelas pejalan kaki dibandingkan model hasil *fine-tuning* biasa tanpa penanganan pencahayaan?
* **RQ3 (Precision vs. Recall Trade-off):** Bagaimana dampak peningkatan pencahayaan terhadap tingkat alarm palsu (*False Positives*) pada objek non-manusia (misalnya tiang atau bayangan jalan)?
* **RQ4 (Edge Feasibility):** Apakah keseluruhan pipeline preprocessing dan deteksi inferensi tetap mampu mempertahankan *throughput* *real-time* ($\ge 20$–$25$ FPS) pada perangkat berdaya komputasi standar (CPU laptop)?

### 3.3 Batasan Ruang Lingkup Proyek (*Scope & Boundaries*)
1. **Dataset:** Dibatasi pada dataset publik *Roboflow Low-light v2* yang berisi 2.400 citra kondisi malam jalanan dengan resolusi $640 \times 640$ piksel.
2. **Kategori Objek:** Fokus evaluasi utama ditujukan pada kelas `person` (pejalan kaki), dengan kelas `Vehicle` sebagai kelas pembanding konteks lalu lintas.
3. **Arsitektur Model:** Dibatasi pada varian *lightweight* YOLOv8 (YOLOv8-Nano dan YOLOv8-Small) untuk memastikan kelayakan komputasi satu semester.
4. **Lingkungan Komputasi:** Pelatihan (*fine-tuning*) dijalankan menggunakan GPU NVIDIA T4 (Google Colab), sedangkan pengujian latensi inferensi dilakukan pada arsitektur CPU/GPU laptop lokal.
5. **Format Masukan:** Citra statis RGB beresolusi $640 \times 640$ serta simulasi aliran frame video beruntun.

### 3.4 Kriteria Ukuran Keberhasilan (*Measurable Success Criteria*)

| Metrik | Nilai Acuan / Target | Justifikasi & Kriteria Sukses |
|---|---|---|
| **Recall (Pejalan Kaki)** | Target kenaikan $\ge +10\%$ relatif terhadap baseline | Metrik prioritas utama: meminimalkan pejalan kaki yang tidak terdeteksi (*false negative* fatal). |
| **mAP@50 (Overall)** | Stabil atau meningkat ($\ge 0.50$) | Memastikan akurasi lokalisasi bounding box tetap presisi dan tidak terdegradasi. |
| **Precision** | Dijaga seimbang ($\Delta \text{Precision} \le -5\%$) | Menghindari lonjakan deteksi palsu akibat amplifikasi noise citra. |
| **Inference Speed** | $\ge 20$–$25$ Frame Per Second (FPS) | Memenuhi batas kelayakan inferensi *real-time* pada skenario pemantauan jalan. |

---

## BAB 4: Rujukan Terkait dan Posisi Proyek

Untuk mendasari justifikasi metodologi dan desain eksperimen, proyek ini mengacu pada 5 literatur ilmiah kredibel:

```
[1] Jocher et al. (2023) ──> Detektor Real-Time Anchor-Free (YOLOv8)
[2] Loh & Chan (2019)   ──> Benchmark Empiris Degradasi Low-Light (ExDark)
[3] Zuiderveld (1994)   ──> Pencerahan Adaptif Lokal Kontras Terbatas (CLAHE)
[4] Guan et al. (2019)  ──> Illumination-Aware Pedestrian Detection
[5] Ye et al. (2026)    ──> Metodologi Benchmark Latensi & Edge CV (LPCVC 2025/2026)
```

### 4.1 Rujukan Literatur Utama
1. **Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8*. GitHub Repository.**  
   *Relevansi:* Menyediakan fondasi arsitektur deteksi modern *anchor-free* dengan *backbone* CSPDarknet termodifikasi dan modul C2f. YOLOv8 dipilih karena menghasilkan *trade-off* terbaik antara akurasi mAP dan latensi rendah pada perangkat komputasi terbatas dibandingkan generasi YOLO terdahulu.
2. **Loh, Y. P., & Chan, C. S. (2019). *Getting to know low-light images with The ExDark dataset*. Computer Vision and Image Understanding (CVIU), Elsevier, 178, 30-42.**  
   *Relevansi:* Studi komprehensif pertama yang menganalisis degradasi performa detektor visual pada kondisi minim cahaya (menemukan penurunan mAP berkisar antara 20% hingga 40%). Paper ini merekomendasikan perbaikan pada ruang iluminasi citra sebagai langkah krusial sebelum ekstraksi fitur konvolusi.
3. **Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization (CLAHE)*. Graphics Gems IV, Academic Press, 474-485.**  
   *Relevansi:* Metode peningkatan kontras lokal adaptif yang membagi citra ke dalam blok kontekstual (*tiles*) dan menerapkan pemangkasan histogram (*clipping limit*). Pendekatan ini secara spesifik mencegah amplifikasi derau berlebih (*noise over-amplification*) pada area gelap gulita, sangat ideal untuk memunculkan tepi siluet pejalan kaki.
4. **Guan, D., Cao, Y., Yang, J., Shen, C., & Trefzer, M. G. (2019). *Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection*. IEEE Transactions on Information Forensics and Security, 14(11), 3048-3059.**  
   *Relevansi:* Menunjukkan bahwa tingkat iluminasi lingkungan mempengaruhi keandalan fitur visual, dan pembobotan berbasis kondisi pencahayaan terbukti mampu menurunkan tingkat kegagalan deteksi pejalan kaki.
5. **Ye, H., et al. (2026). *Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge*. arXiv preprint arXiv:2604.19054.**  
   *Relevansi:* Panduan resmi rujukan dosen untuk evaluasi model visi komputer di bawah keterbatasan daya (*low-power*). Paper ini mendasari protokol pengukuran latensi, FPS, serta pemisahan siklus *warm-up* agar estimasi performa edge objektif.

### 4.2 Posisi Proyek terhadap Penelitian Terdahulu (*Positioning & Research Gap*)
Sebagian besar penelitian mutakhir pada ranah *low-light vision* mengandalkan model restorasi berbasis *generative adversarial networks* (seperti EnlightenGAN) atau jaringan *dual-stream multi-modal* (RGB + Thermal). Meskipun akurasinya tinggi, metode tersebut membutuhkan komputasi masif, memori GPU besar, serta parameter ganda yang tidak mungkin dioperasikan secara *real-time* pada perangkat edge terjangkau.

**Celah penelitian (*gap*) yang diisi proyek ini:**  
Proyek ini menguji pendekatan *lightweight preprocessing augmentation* yang efisien secara komputasi. Kami mengevaluasi apakah intervensi augmentasi matematis terarah (CLAHE dan Gamma Correction) yang disematkan langsung pada *training pipeline* YOLOv8 sudah cukup untuk menjembatani jurang *domain shift* antara kondisi terang dan gelap, sehingga mendongkrak *Recall* pejalan kaki tanpa perlu menambah latensi inferensi arsitektur jaringan secara signifikan.

---

## BAB 5: Dataset, Karakteristik, dan Integritas Data

### 5.1 Sumber dan Lisensi Dataset
* **Nama Dataset:** *Low-light v2 (Object Detection in Low Light)*
* **Penyedia / Platform:** Roboflow Universe (`object-detection-in-ll/low-light-sxzx2`)
* **Lisensi Penggunaan:** *Creative Commons Attribution 4.0 International* (CC BY 4.0) — legal dan diizinkan secara penuh untuk riset akademik dan modifikasi dengan atribusi yang sesuai.
* **Format Anotasi:** Format Bounding Box YOLOv8 (`.txt` per citra: `<class_id> <x_center> <y_center> <width> <height>` ternormalisasi $[0, 1]$).

### 5.2 Hasil Audit Kuantitatif Dataset (Pemenuhan CP03)
Audit dataset dilakukan secara ketat menggunakan skrip `src/audit_dataset.py` yang memvalidasi integritas file, parsing label koordinat, dan distribusi anotasi per split. Hasil audit kuantitatif disajikan pada Tabel 5.1:

**Tabel 5.1: Distribusi Dataset dan Statistik Bounding Box per Split**

| Split Data | Jumlah Citra | Total Bounding Box | Bounding Box `Vehicle` (ID: 0) | Bounding Box `person` (ID: 1) | Citra Berisi `person` | Citra Tanpa Objek |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 2.100 | 6.816 | 6.018 (88,3%) | 798 (11,7%) | 413 (19,7%) | 4 |
| **Valid** | 200 | 650 | 581 (89,4%) | 69 (10,6%) | 35 (17,5%) | 0 |
| **Test** | 100 | 311 | 272 (87,5%) | 39 (12,5%) | 18 (18,0%) | 0 |
| **Total** | **2.400** | **7.777** | **6.871 (88,4%)** | **906 (11,6%)** | **466 (19,4%)** | **4** |

### 5.3 Analisis Karakteristik dan Ketidakseimbangan Kelas (*Class Imbalance*)
Dari hasil audit ditemukan temuan empiris penting:
* **Ketidakseimbangan Kelas Tinggi:** Rasio jumlah anotasi antara kendaraan (`Vehicle`) dan pejalan kaki (`person`) mencapai **7,58 : 1**. Pejalan kaki hanya mencakup **11,65%** dari keseluruhan anotasi.
* **Implikasi Desain Evaluasi:** Jika model hanya dievaluasi menggunakan *Overall Accuracy* atau *Mean Average Precision* (mAP) agregat, model yang gagal mendeteksi pejalan kaki pun masih bisa memperoleh skor tinggi karena terkompensasi oleh akurasi deteksi kendaraan. Oleh karena itu, metrik **Recall kelas `person` dijadikan fokus utama evaluasi**.

### 5.4 Mitigasi Kebocoran Data (*Data Leakage Prevention*)
Untuk memenuhi prinsip integritas data sesuai rubrik dosen:
1. **Pemisahan Tingkat Direktori:** Pembagian data telah terisolasi secara kaku ke dalam folder terpisah (`dataset/train`, `dataset/valid`, dan `dataset/test`).
2. **Isolasi Prapemrosesan:** Seluruh proses *fitting*, kalkulasi histogram, dan augmentasi adaptif (CLAHE/Gamma) **hanya diterapkan pada split `train`**.
3. **Penyegelan Test Set (*Test Set Freezing*):** Split `test` (100 citra, 39 pejalan kaki) dijaga murni (*unseen*) tanpa augmentasi dan tidak disentuh selama proses penyetelan hyperparameter, hanya digunakan satu kali saat evaluasi performa akhir.

---

## BAB 6: Metodologi Penelitian dan Desain Teknis

### 6.1 Alur Kerja Sistem (*End-to-End System Pipeline*)
Sistem dirancang modular dari persiapan data hingga penyajian hasil, digambarkan dalam diagram alir berikut:

```
[ Dataset Low-Light ]
         │
         ├──> Split Train (2.100 citra) ──> [ Pipeline Augmentasi: CLAHE / Gamma ] 
         │                                                     │
         ├──> Split Valid (200 citra)   ───────────────────────┼──> [ Training YOLOv8 (Colab T4) ]
         │                                                     │                   │
         └──> Split Test (100 citra)    ──> [ Evaluasi Uji ] <─┘                   │
                                                   │                               v
                                                   └──> [ Metrik: Recall, mAP, FPS, Failure Cases ]
```

### 6.2 Arsitektur Model: YOLOv8-Nano
Model detektor yang digunakan adalah **YOLOv8-Nano** (`yolov8n.pt`). Pemilihan arsitektur ini didasarkan pada pertimbangan:
* **Ukuran Parameter Kecil (~3.2M parameter):** Sangat ringan dan memungkinkan proses *training* cepat di Google Colab serta inferensi lancar di CPU laptop standar.
* **Anchor-Free Detection Head:** Memprediksi pusat dan ukuran kotak secara langsung tanpa *predefined anchor boxes*, lebih adaptif terhadap variasi aspek rasio pejalan kaki dalam jarak yang beragam.
* **Backbone & Neck C2f Module:** Menggabungkan representasi fitur multiskala (*high-level semantic* dan *low-level spatial*) untuk mengenali objek pejalan kaki berukuran kecil hingga sedang.

### 6.3 Desain Intervensi Prapemrosesan & Augmentasi
Kami merancang dua metode peningkatan citra adaptif berbasis pustaka OpenCV dan Albumentations:

1. **Contrast Limited Adaptive Histogram Equalization (CLAHE):**
   * Citra RGB dikonversi ke ruang warna yang memisahkan intensitas cahaya dari krominansi, yaitu ruang warna **CIE L\*a\*b\*** (atau YCrCb).
   * CLAHE hanya diaplikasikan pada kanal pencahayaan (**L-channel**) dengan parameter *Clip Limit* $= 2.0$ dan ukuran blok *Tile Grid* $= 8 \times 8$.
   * Citra digabungkan kembali dan dikonversi ke format RGB. Hal ini meningkatkan ketajaman siluet lokal tanpa mengubah karakteristik warna alami (*color distortion*).
2. **Gamma Correction (Non-Linear Brightness Mapping):**
   * Transformasi nilai intensitas piksel $I_{\text{out}} = 255 \times (I_{\text{in}} / 255)^\gamma$.
   * Nilai $\gamma$ dipilih pada rentang $0.5 \le \gamma \le 0.7$ untuk mengangkat nilai piksel pada rentang *mid-to-low tones* secara halus tanpa menyebabkan saturasi berlebih (*over-exposure*) pada lampu jalan.

### 6.4 Fungsi Loss dan Konfigurasi Pelatihan
Pelatihan YOLOv8 memanfaatkan kombinasi fungsi objektif multi-tugas:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}}$$
* $\mathcal{L}_{\text{CIoU}}$ (*Complete Intersection over Union Loss*): Mempertimbangkan rasio tumpang tindih, jarak pusat kotak, dan rasio aspek bounding box.
* $\mathcal{L}_{\text{DFL}}$ (*Distribution Focal Loss*): Mengoptimalkan batas tepi kotak pada area yang kabur akibat kegelapan.
* $\mathcal{L}_{\text{BCE}}$ (*Binary Cross-Entropy Loss*): Mengklasifikasikan kelas `person` versus `Vehicle`.

**Tabel 6.1: Konfigurasi Hyperparameter Pelatihan Terstandarisasi**

| Parameter | Nilai Pengaturan | Keterangan / Alasan |
|---|---|---|
| **Base Model** | YOLOv8n (pretrained COCO) | Memanfaatkan bobot awal representasi visual umum |
| **Input Resolution** | $640 \times 640$ piksel | Menjaga keseimbangan detail visual dan latensi komputasi |
| **Batch Size** | 16 | Optimal untuk alokasi memori VRAM GPU NVIDIA T4 |
| **Epochs** | 50 epoch (Early Stopping: 15) | Mencegah overfitting dengan memantau validasi loss |
| **Optimizer** | AdamW / SGD ($\text{lr}_0 = 0.01$) | One-cycle cosine learning rate decay |
| **Random Seed** | 42 | Menjamin seluruh eksperimen dapat direproduksi (*reproducibility*) |

---

## BAB 7: Rencana Evaluasi dan Eksperimen Terkontrol

### 7.1 Matriks Eksperimen Terkontrol (*Controlled Experiment Design*)
Untuk memastikan kesimpulan eksperimen bersifat ilmiah dan tidak bias, seluruh model dievaluasi pada split pengujian (*test set*) yang identik menggunakan satu variabel bebas yang diisolasi secara sistematis:

**Tabel 7.1: Desain Eksperimen Komparatif Terkontrol**

| ID Eksperimen | Nama Eksperimen | Strategi Model & Prapemrosesan | Variabel Bebas Utama |
|---|---|---|---|
| **B0** | **Baseline (Zero-Shot)** | YOLOv8n pretrained MS COCO langsung diuji pada data malam | Kontrol awal tanpa *fine-tuning* domain malam |
| **Exp-A** | **Fine-Tuning Standar** | Fine-tuning YOLOv8n pada data train low-light tanpa augmentasi pencahayaan | Kontrol pembelajaran transfer domain malam |
| **Exp-B** | **Fine-Tuning + CLAHE** | Fine-tuning YOLOv8n dengan pipeline augmentasi CLAHE (L-channel, clip=2.0) | Peningkatan kontras lokal adaptif |
| **Exp-C** | **Fine-Tuning + Gamma** | Fine-tuning YOLOv8n dengan augmentasi Gamma Correction ($\gamma = 0.6$) | Pencerahan kurva non-linear global |

### 7.2 Metrik Evaluasi Kuantitatif
Evaluasi menggunakan metrik standar visi komputer yang dihitung secara independen:
1. **Recall Kelas Pejalan Kaki ($\text{Recall}_p$):**
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   Mengukur persentase pejalan kaki nyata yang berhasil dideteksi oleh sistem (sasaran: $\text{FN} \to 0$).
2. **Precision Kelas Pejalan Kaki ($\text{Precision}_p$):**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   Mengukur tingkat kebenaran ketika model mengklaim menemukan pejalan kaki.
3. **Mean Average Precision (mAP@50 & mAP@50:95):**
   Rata-rata area di bawah kurva Precision-Recall pada ambang batas $\text{IoU} = 0.50$ dan ambang ketat $[0.50 : 0.95]$.
4. **Efisiensi Edge (Latensi & FPS):**
   Diukur menggunakan protokol benchmarking resmi (mengacu pada Ye et al., 2026): 50 iterasi *warm-up* diikuti oleh pencatatan rata-rata waktu inferensi per citra (ms) dan kalkulasi FPS pada CPU laptop.

### 7.3 Rencana Analisis Kesalahan (*Error Analysis Plan*)
Evaluasi tidak hanya menampilkan tabel angka, melainkan membedah kegagalan model ke dalam 3 kategori:
* **Analisis False Negatives (FN):** Meneliti karakteristik citra pejalan kaki yang lolos dari deteksi (misal: jarak terlalu jauh, pencahayaan $< 5$ lux, atau tertutup sebagian/oklusi oleh kendaraan).
* **Analisis False Positives (FP):** Mengidentifikasi objek lingkungan (pohon, tiang reklame, bayangan) yang salah diklasifikasikan sebagai pejalan kaki akibat artefak noise setelah penerapan CLAHE.
* **Analisis Confusion Matrix:** Memantau apakah terjadi pertukaran prediksi (*misclassification*) antara `person` dan `Vehicle`.

---

## BAB 8: Rencana Kerja, Manajemen Risiko, dan Kebutuhan Sumber Daya

### 8.1 Pembagian Peran dan Tugas Anggota (*Role Distribution*)

**Tabel 8.1: Matriks Tanggung Jawab Anggota Kelompok 9**

| Anggota Kelompok | Peran Utama | Rincian Tugas Konkret |
|---|---|---|
| **Krisantus Linardi** | *Data & Preprocessing Lead* | 1. Melakukan audit dan kurasi dataset YOLOv8 (`src/audit_dataset.py`).<br>2. Mengembangkan modul prapemrosesan CLAHE dan Gamma Correction.<br>3. Menganalisis distribusi data, risiko leakage, dan class imbalance.<br>4. Menyusun laporan evaluasi kuantitatif dan analisis failure cases. |
| **Kristofer Leonard** | *Model & Deployment Lead* | 1. Mengonfigurasi arsitektur YOLOv8 dan pipeline pelatihan Ultralytics.<br>2. Menjalankan evaluasi baseline zero-shot (B0) dan training Exp A/B/C di Colab.<br>3. Mengukur latensi inferensi, FPS, dan efisiensi model di perangkat lokal.<br>4. Membangun antarmuka inferensi demo (OpenCV/Streamlit) untuk demonstrasi. |

*Catatan: Kedua anggota terlibat aktif dalam diskusi arsitektur, penulisan proposal, dan sesi tanya jawab presentasi.*

### 8.2 Jadwal Kerja Menuju UTS dan UAS (*Project Timeline*)

**Tabel 8.2: Rencana Jadwal Kerja Mingguan**

| Minggu | Milestone / Capaian | Output yang Dihasilkan | Penanggung Jawab |
|:---:|---|---|---|
| **M03** | Pemilihan Topik & Dataset (CP00, CP01) | Dokumen topik terpilih, verifikasi dataset Roboflow | Bersama |
| **M04** | Audit Data & Studi Literatur (CP02, CP03) | Laporan audit dataset (`audit_report.md`), 5 paper rujukan, draf awal proposal | Krisantus |
| **M05** | Eksekusi Baseline B0 & Evaluasi Awal (CP04) | Skrip baseline zero-shot, metrik acuan awal pada valid/test set | Kristofer |
| **M06** | Desain Eksperimen & Pipeline CLAHE (CP05) | Modul augmentasi teruji, skrip pelatihan terkontrol siap jalan | Krisantus |
| **M07** | Training Eksperimen & Freeze Proposal (CP06) | Hasil awal Exp-A/B, draf final proposal UTS beku, slide presentasi | Bersama |
| **M08** | **Sidang Proposal UTS** | **Presentasi 8 menit & Ujian Proposal Kelompok 9** | **Bersama** |
| **M09-12** | Fase Eksperimen Lengkap & Analisis (CP07-CP10)| Tabel komparasi lengkap (B0, Exp-A, Exp-B, Exp-C), kurva PR | Kristofer & Krisantus |
| **M13-14** | Analisis Kesalahan & Pembuatan Demo (CP11, CP12)| Dokumen error analysis, aplikasi demo inferensi video | Bersama |
| **M15-16** | **Sidang Hasil & Demonstrasi UAS (CP13)** | **Laporan teknis final, repositori GitHub bersih, demo langsung** | **Bersama** |

### 8.3 Kebutuhan Sumber Daya Komputasi
* **Komputasi Pelatihan:** Google Colab GPU NVIDIA T4 (16 GB VRAM), estimasi waktu pelatihan berkisar antara 1,5 hingga 3 jam per eksperimen (total alokasi $\approx 8$–$10$ jam GPU compute).
* **Komputasi Evaluasi & Inferensi:** Laptop personal (Processor Intel Core i5/i7, RAM 16GB) untuk mengukur latensi CPU murni dan memvalidasi kelayakan deploy pada perangkat komputasi umum.
* **Perangkat Lunak:** Python 3.10+, PyTorch, Ultralytics YOLOv8, OpenCV, Albumentations, Matplotlib, Pandas, Git.

### 8.4 Analisis Risiko dan Rencana Kontingensi (*Risk Management & Fallback*)

**Tabel 8.3: Matriks Mitigasi Risiko Teknis**

| Potensi Risiko Teknis | Dampak | Probabilitas | Rencana Mitigasi & Rencana Alternatif (*Fallback Plan*) |
|---|:---:|:---:|---|
| **1. Batasan Kuota GPU Google Colab Terputus** | Tinggi | Sedang | *Mitigasi:* Menyimpan model checkpoint setiap 5 epoch secara otomatis ke Google Drive. Jika kuota habis, pelatihan dialihkan ke Kaggle Notebooks (kuota 30 jam/minggu) atau GPU lokal alternatif. |
| **2. CLAHE Memperparah Noise (*Over-amplification*)** | Sedang | Sedang | *Mitigasi:* Membatasi *clip limit* maksimum pada angka 2.0. Jika False Positive tetap meningkat, pasang filter penghalus tepian ringan seperti *Bilateral Filter* atau *Gaussian Blur* sebelum CLAHE. |
| **3. Pejalan Kaki Tetap Sulit Dideteksi Akibat Class Imbalance** | Tinggi | Rendah | *Mitigasi:* Melakukan *tuning confidence threshold* khusus untuk kelas `person` (misal diturunkan ke 0.25) atau menerapkan penyesuaian bobot loss (*class-weighted loss*) pada fungsi klasifikasi. |
| **4. Latensi Preprocessing Terlalu Berat di CPU Laptop** | Rendah | Rendah | *Mitigasi:* Mengoptimalkan operasi CLAHE dengan format array C-contiguous OpenCV, atau mengalihkan augmentasi hanya pada tahap pelatihan, sementara pada tahap inferensi menggunakan CLAHE sub-sampling pada resolusi $320 \times 320$. |

---

## DAFTAR PUSTAKA

1. Guan, D., Cao, Y., Yang, J., Shen, C., & Trefzer, M. G. (2019). Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection. *IEEE Transactions on Information Forensics and Security*, 14(11), 3048–3059. https://doi.org/10.1109/TIFS.2019.2910065
2. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.0.0) [Computer software]. GitHub. https://github.com/ultralytics/ultralytics
3. Loh, Y. P., & Chan, C. S. (2019). Getting to know low-light images with The ExDark dataset. *Computer Vision and Image Understanding*, 178, 30–42. https://doi.org/10.1016/j.cviu.2018.10.010
4. Ye, H., et al. (2026). Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge. *arXiv preprint arXiv:2604.19054*. https://arxiv.org/abs/2604.19054
5. Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. In P. S. Heckbert (Ed.), *Graphics Gems IV* (pp. 474–485). Academic Press. https://doi.org/10.1016/B978-0-12-336156-1.50061-6
