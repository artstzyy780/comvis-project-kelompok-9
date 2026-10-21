# Computer Vision Final Project: Technical Report / Proposal
## Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif

**Kode Berkas:** `K09_DeteksiPedestrianLowLight_Proposal_v02`  
**Mata Kuliah:** Visi Komputer (*Computer Vision* — IF0PP006)  
**Kelas / Semester:** 5PITA / Ganjil 2026/2027  
**Fase:** Ujian Tengah Semester (Sidang Proposal UTS)  

---

### Kelompok 9

| Nama Mahasiswa | Peran & Tanggung Jawab Utama |
|---|---|
| **Krisantus Amka Ginting** | Kelompok 9 — Data & Preprocessing Lead |
| **Linardi Halim** | Kelompok 9 — Documentation & Reporting Lead |
| **Kristofer Ryan Giggs Eka Saputra** | Kelompok 9 — Model & Training Lead |
| **Leonard Franscois** | Kelompok 9 — Demo & Deployment Lead |

---

### Abstract
Abstract—This project addresses the critical challenge of nighttime pedestrian detection in Intelligent Transportation Systems (ITS) and Advanced Driver Assistance Systems (ADAS). Standard deep learning detectors often suffer severe performance degradation under low-light conditions due to low contrast, sensor noise, and glare from headlights. We propose a real-time computer vision pipeline utilizing the Ultralytics YOLOv8-Nano detector coupled with adaptive illumination preprocessing techniques, specifically Contrast Limited Adaptive Histogram Equalization (CLAHE) on the CIE L\*a\*b\* L-channel and non-linear Gamma Correction. Evaluated on the Roboflow Low-light v2 dataset comprising 2,400 street images with 7,777 annotated objects (showing a 7.58:1 vehicle-to-person class imbalance), our study prioritizes pedestrian Recall to minimize life-threatening false negatives. Controlled experiments compare zero-shot baseline, fine-tuned baseline, and illumination-augmented variants on a strictly isolated test set. The proposed lightweight approach aims to boost pedestrian Recall by over 10% while maintaining real-time inference throughput ($\ge 20$ FPS) on standard CPU hardware.

### Abstrak
Abstrak—Proyek ini mengatasi tantangan krusial deteksi pejalan kaki di malam hari pada Sistem Transportasi Cerdas (ITS) dan Advanced Driver Assistance Systems (ADAS). Model detektor deep learning standar kerap mengalami degradasi performa drastis pada kondisi minim cahaya akibat kontras rendah, derau sensor, dan silau lampu kendaraan. Kami mengusulkan pipeline visi komputer real-time berbasis Ultralytics YOLOv8-Nano yang dipadukan dengan teknik prapemrosesan pencahayaan adaptif, yaitu Contrast Limited Adaptive Histogram Equalization (CLAHE) pada L-channel ruang warna CIE L\*a\*b\* serta Gamma Correction non-linear. Dievaluasi pada dataset Roboflow Low-light v2 berukuran 2.400 citra jalanan dengan 7.777 objek teranotasi (memiliki ketidakseimbangan kelas kendaraan terhadap pejalan kaki sebesar 7,58:1), penelitian ini memprioritaskan metrik Recall pejalan kaki untuk meminimalkan kegagalan deteksi (false negative) yang membahayakan nyawa. Eksperimen terkontrol membandingkan baseline zero-shot, fine-tuned baseline, dan varian teraugmentasi pencahayaan pada split test yang terisolasi. Pendekatan ringan yang diusulkan menargetkan peningkatan Recall pejalan kaki sebesar >10% dengan tetap mempertahankan kecepatan inferensi real-time ($\ge 20$ FPS) pada perangkat CPU standar.

---

## 1. Introduction / Pendahuluan

Pedestrian detection is a cornerstone safety component in smart mobility and autonomous driving applications. Statistically, a disproportionate number of fatal pedestrian accidents occur during twilight and nighttime hours when human visibility is severely compromised. Conventional object detection models trained predominantly on daytime datasets (such as MS COCO) struggle in nighttime environments because low photon counts reduce image contrast, amplify sensor grain, and induce extreme dynamic range from artificial light sources. Traditional image processing filters often fail to dynamically adapt to varying local darkness, while heavy multi-modal or generative low-light restoration networks demand excessive compute incompatible with real-time edge processing. Therefore, a deep learning object detection approach reinforced by adaptive, lightweight illumination enhancement is required to reliably extract spatial and textural features of pedestrians without introducing prohibitive computational latency.

Deteksi pejalan kaki merupakan komponen keselamatan utama dalam mobilitas cerdas dan sistem kendali otonom. Secara statistik, sebagian besar kecelakaan fatal pejalan kaki terjadi pada malam hari ketika jarak pandang manusia sangat terbatas. Model deteksi objek konvensional yang dilatih mayoritas pada data siang hari (seperti MS COCO) mengalami penurunan keandalan di lingkungan malam karena intensitas foton yang rendah mengurangi kontras citra, mengamplifikasi derau sensor, dan menciptakan rentang dinamis ekstrem dari pencahayaan buatan. Filter pemrosesan citra tradisional tidak mampu beradaptasi secara dinamis terhadap tingkat kegelapan lokal, sedangkan model restorasi generatif yang berat membutuhkan daya komputasi tinggi yang tidak cocok untuk inferensi edge real-time. Oleh karena itu, pendekatan visi komputer deep learning yang diperkuat oleh prapemrosesan pencahayaan adaptif yang ringan sangat diperlukan untuk mengekstraksi fitur spasial pejalan kaki secara andal tanpa mengorbankan kecepatan inferensi.

---

## 2. Methodology / Metodologi

### 2.1 A. Data Strategy / Strategi Data

We utilize the Roboflow Low-light v2 dataset, licensed under CC BY 4.0, containing 2,400 nighttime traffic images at 640x640 resolution. A rigorous audit using our custom script (`src/audit_dataset.py`) verified 7,777 bounding box annotations partitioned into Train (2,100 images; 6,816 boxes), Validation (200 images; 650 boxes), and Test (100 images; 311 boxes) splits. The dataset exhibits a severe class imbalance, with 6,871 Vehicle boxes (88.4%) versus 906 Person boxes (11.6%), yielding a 7.58:1 ratio. Preprocessing involves L-channel CLAHE (clip limit = 2.0, tile grid = 8x8) in CIE L\*a\*b\* space to selectively enhance local silhouette contrast, as well as Gamma Correction (gamma = 0.6) to brighten mid-tones without saturating highlights. To strictly prevent data leakage, all histogram fitting and augmentations are applied solely to the training set, keeping the test split completely unseen.

Kami menggunakan dataset Roboflow Low-light v2 (lisensi CC BY 4.0) yang terdiri dari 2.400 citra lalu lintas malam beresolusi 640x640 piksel. Hasil audit ketat menggunakan skrip khusus (`src/audit_dataset.py`) mengonfirmasi 7.777 anotasi bounding box yang terbagi ke dalam split Train (2.100 citra; 6.816 box), Validasi (200 citra; 650 box), dan Uji (100 citra; 311 box). Dataset mengalami ketidakseimbangan kelas yang signifikan dengan 6.871 box Vehicle (88,4%) berbanding 906 box Person (11,6%), atau rasio 7,58:1. Prapemrosesan menerapkan CLAHE pada kanal L (clip limit = 2.0, tile grid = 8x8) dalam ruang warna CIE L\*a\*b\* untuk memperjelas siluet lokal, serta Gamma Correction (gamma = 0.6) untuk mencerahkan tone sedang tanpa membuat area terang menjadi jenuh. Untuk mencegah kebocoran data (data leakage), seluruh kalkulasi histogram dan augmentasi hanya diterapkan pada data latih, sementara data uji dijaga murni.

### 2.2 B. Model Architecture / Arsitektur Model

We adopt the lightweight Ultralytics YOLOv8-Nano (`yolov8n.pt`) architecture (~3.2 million parameters) due to its optimal balance between detection capability and low computational footprint on edge devices. YOLOv8 employs an anchor-free detection head that directly predicts object center points and bounding box dimensions, making it highly adaptable to varying pedestrian aspect ratios. Its backbone and neck utilize Cross Stage Partial C2f modules to effectively aggregate low-level spatial contours with high-level semantic context. Training is guided by a composite loss function comprising Complete IoU (CIoU) for bounding box spatial alignment, Distribution Focal Loss (DFL) for fine-grained boundary regression under blurry low-light conditions, and Binary Cross-Entropy (BCE) for class assignment. Hyperparameters are standardized: batch size = 16, initial learning rate = 0.01 (AdamW/SGD with cosine decay), random seed = 42, trained for 50 epochs with early stopping.

Kami menggunakan arsitektur Ultralytics YOLOv8-Nano (`yolov8n.pt`) berukuran ~3,2 juta parameter karena memiliki keseimbangan terbaik antara akurasi deteksi dan efisiensi komputasi pada perangkat edge. YOLOv8 menggunakan head deteksi anchor-free yang secara langsung memprediksi titik pusat dan dimensi bounding box, sehingga sangat adaptif terhadap berbagai rasio aspek pejalan kaki. Backbone dan neck memanfaatkan modul C2f untuk menggabungkan kontur spasial tingkat rendah dengan konteks semantik tingkat tinggi. Pelatihan dipandu oleh kombinasi fungsi loss Complete IoU (CIoU) untuk akurasi bounding box, Distribution Focal Loss (DFL) untuk regresi batas objek yang kabur pada kondisi gelap, dan Binary Cross-Entropy (BCE) untuk klasifikasi objek. Hyperparameter distandarisasi: batch size = 16, learning rate awal = 0,01 (AdamW/SGD dengan cosine decay), random seed = 42, dilatih selama 50 epoch dengan early stopping.

### 2.3 C. Deployment & Environment / Lingkungan & Penerapan

Model training and fine-tuning experiments are conducted using an NVIDIA T4 GPU environment hosted on Google Colab (16 GB VRAM, PyTorch framework). Evaluation of latencies and throughput benchmarking is performed on a local host machine equipped with an Intel Core i5/i7 processor and 16 GB RAM to verify real-time feasibility on standard consumer hardware. For deployment, the trained YOLOv8 model is integrated into a lightweight Python inference pipeline (OpenCV / Streamlit) that ingests video frames or static image streams, executes adaptive illumination preprocessing, and renders bounding box predictions with confidence scores at $\ge 20$–$25$ FPS.

Eksperimen pelatihan dan fine-tuning model dijalankan menggunakan perangkat keras GPU NVIDIA T4 pada Google Colab (16 GB VRAM, framework PyTorch). Pengukuran latensi dan throughput inferensi dilakukan pada perangkat lokal berprosesor Intel Core i5/i7 dan RAM 16 GB untuk memverifikasi kelayakan real-time pada perangkat keras umum. Untuk penerapan, model YOLOv8 yang telah dilatih diintegrasikan ke dalam pipeline inferensi Python (OpenCV / Streamlit) yang menerima masukan frame video atau gambar statis, menjalankan prapemrosesan pencahayaan adaptif, dan menampilkan prediksi bounding box beserta confidence score secara real-time ($\ge 20$–$25$ FPS).

---

## 3. Results and Discussion / Hasil dan Pembahasan

Our quantitative evaluation framework prioritizes pedestrian Recall alongside standard object detection metrics (Precision, mAP@50, mAP@50:95, and FPS). Four controlled experiment variants are established on identical test splits: B0 (Zero-shot YOLOv8n COCO baseline), Exp-A (Fine-tuned YOLOv8n without illumination augmentation), Exp-B (Fine-tuned YOLOv8n with L-channel CLAHE augmentation), and Exp-C (Fine-tuned YOLOv8n with Gamma Correction). Dataset audit results (Table I) confirm a 7.58:1 class imbalance, proving that overall mAP can mask poor pedestrian detection if evaluated uncritically. Preliminary zero-shot baseline testing on nighttime validation images reveals significant false negatives where dark silhouetted pedestrians are missed entirely. The controlled experiment design aims to demonstrate that CLAHE and Gamma augmentations significantly improve pedestrian Recall (+10% target) without incurring excessive false positive rates.

Kerangka evaluasi kuantitatif kami memprioritaskan Recall pejalan kaki di samping metrik deteksi standar (Precision, mAP@50, mAP@50:95, dan FPS). Empat varian eksperimen terkontrol ditetapkan pada split uji yang identik: B0 (Baseline zero-shot YOLOv8n COCO), Exp-A (Fine-tuned YOLOv8n tanpa augmentasi pencahayaan), Exp-B (Fine-tuned YOLOv8n dengan augmentasi CLAHE), dan Exp-C (Fine-tuned YOLOv8n dengan Gamma Correction). Hasil audit dataset (Tabel I) mengonfirmasi ketidakseimbangan kelas 7,58:1, membuktikan bahwa nilai mAP keseluruhan dapat mengeluhkan rendahnya deteksi pejalan kaki jika tidak dievaluasi per kelas. Pengujian awal baseline zero-shot menunjukkan tingginya false negative di mana pejalan kaki bersiluet gelap tidak terdeteksi sama sekali. Desain eksperimen terkontrol ini bertujuan membuktikan bahwa augmentasi CLAHE dan Gamma meningkatkan Recall pejalan kaki secara signifikan (target +10%) tanpa memicu peningkatan false positive berlebih.

### Visualizing Results / Visualisasi Hasil

Quantitative data distribution from the dataset audit is summarized in Table I. Controlled experiment configurations are detailed in Table II. Figure 1 illustrates visual annotations generated during the dataset audit phase (`results/audit/sample_annotations.png`), showing green bounding boxes for Person objects and blue boxes for Vehicle objects under realistic nighttime lighting. Qualitative failure case analysis plans focus on categorizing errors into False Negatives (pedestrians obscured by extreme shadows or heavy distance) and False Positives (misclassifying ambient structures such as poles or tree shadows due to noise over-amplification).

Distribusi data kuantitatif dari hasil audit dataset dirangkum dalam Tabel I. Konfigurasi eksperimen terkontrol dirinci dalam Tabel II. Gambar 1 menampilkan visualisasi anotasi yang dihasilkan pada tahap audit data (`results/audit/sample_annotations.png`), yang memperlihatkan kotak hijau untuk objek Person dan kotak biru untuk objek Vehicle pada kondisi pencahayaan malam. Rencana analisis kasus gagal (failure cases) difokuskan pada pengkategorian kesalahan menjadi False Negatives (pejalan kaki terhalang bayangan gelap atau berjarak jauh) dan False Positives (salah mengklasifikasikan struktur lingkungan seperti tiang atau bayangan pohon akibat amplifikasi derau).

**Table I: Dataset Audit & Bounding Box Distribution per Split**

| Split Data | Jumlah Citra | Total Bounding Box | Box `Vehicle` | Box `person` | Rasio Imbalance |
|---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 2.100 | 6.816 | 6.018 (88,3%) | 798 (11,7%) | 7,54 : 1 |
| **Valid** | 200 | 650 | 581 (89,4%) | 69 (10,6%) | 8,42 : 1 |
| **Test** | 100 | 311 | 272 (87,5%) | 39 (12,5%) | 6,97 : 1 |
| **Total** | **2.400** | **7.777** | **6.871 (88,4%)** | **906 (11,6%)** | **7,58 : 1** |

**Table II: Controlled Experiment Design & Evaluation Framework**

| ID Eksperimen | Strategi Model | Augmentasi / Prapemrosesan | Split Evaluasi | Fokus Metrik Utama |
|---|---|---|---|---|
| **B0** | YOLOv8n (COCO) | None (Zero-shot baseline) | Test (100 citra) | Baseline Recall & mAP50 |
| **Exp-A** | Fine-tuned YOLOv8n | Augmentasi standar (tanpa pencerahan) | Test (100 citra) | Recall Domain Fine-Tuning |
| **Exp-B** | Fine-tuned YOLOv8n | L-channel CLAHE (clip=2.0, tile=8x8) | Test (100 citra) | Target Recall (+10%) & mAP |
| **Exp-C** | Fine-tuned YOLOv8n | Non-linear Gamma Correction ($\gamma=0.6$) | Test (100 citra) | Mid-tone Contrast Recall |

---

## 4. Conclusion / Kesimpulan

This technical proposal establishes a robust, controlled methodology to enhance nighttime pedestrian detection using YOLOv8-Nano and adaptive illumination augmentation (CLAHE and Gamma Correction). By addressing low contrast and sensor noise at the data preprocessing level while keeping model parameters lightweight (~3.2M), the proposed pipeline seeks to achieve a substantial boost in pedestrian Recall without sacrificing real-time inference throughput ($\ge 20$ FPS). Future work will focus on completing all controlled training iterations, quantifying precision-recall trade-offs across brightness slices, and exploring low-bit INT8 quantization for deployment on low-power edge platforms.

Proposal teknis ini menetapkan metodologi terkontrol yang kokoh untuk meningkatkan deteksi pejalan kaki di malam hari menggunakan YOLOv8-Nano dan augmentasi pencahayaan adaptif (CLAHE dan Gamma Correction). Dengan mengatasi masalah kontras rendah dan derau sensor pada tingkat prapemrosesan data tanpa menambah beban parameter model (~3,2M), pipeline ini menargetkan peningkatan signifikan pada Recall pejalan kaki tanpa mengorbankan kecepatan inferensi real-time ($\ge 20$ FPS). Pekerjaan selanjutnya akan berfokus pada penyelesaian seluruh iterasi pelatihan terkontrol, pengukuran trade-off precision-recall pada berbagai tingkat kecerahan, serta eksplorasi kuantisasi INT8 untuk penerapan pada perangkat edge berdaya rendah.

---

## References / Referensi

[1] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," GitHub repository, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics  
[2] Y. P. Loh and C. S. Chan, "Getting to know low-light images with The ExDark dataset," *Computer Vision and Image Understanding*, vol. 178, pp. 30–42, 2019.  
[3] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," *Graphics Gems IV*, Academic Press, pp. 474–485, 1994.  
[4] D. Guan, Y. Cao, J. Yang, C. Shen, and M. G. Trefzer, "Fusion of Multispectral Data Through Illumination-Aware Deep Neural Networks for Pedestrian Detection," *IEEE Transactions on Information Forensics and Security*, vol. 14, no. 11, pp. 3048–3059, 2019.  
[5] H. Ye et al., "Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge," *arXiv preprint arXiv:2604.19054*, 2026.
