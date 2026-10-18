# 🎙️ Naskah Presentasi (Speaker Script) Sidang Proposal UTS
## Proyek Visi Komputer — Kelompok 9 (Kelas 5PITA)

> **Judul Proyek:** Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif  
> **Berkas PPTX:** `K09_Sidang_Proposal_DeteksiPedestrianLowLight.pptx`  
> **Anggota Tim:**  
> 1. **Krisantus Linardi** — Data & Preprocessing Lead  
> 2. **Kristofer Leonard** — Model & Deployment Lead  
> **Estimasi Total Durasi:** ~10 – 12 Menit (Penyampaian tenang, tegas, dan percaya diri)

---

### Panduan Transisi Pembicara (Rekomendasi Bergantian):
* **Pembicara 1 (Krisantus):** Membuka presentasi, Slide 1 sampai Slide 6 (Latar Belakang, Masalah, Pipeline Solusi, Filosofi Recall, dan Kesiapan Data).
* **Pembicara 2 (Kristofer):** Melanjutkan Slide 7 sampai Slide 11 (Eksperimen Terkontrol, Rencana Deployment, Roadmap, Argumentasi Kelayakan, dan Penutup Q&A).
*(Catatan: Naskah ini juga fleksibel jika ingin dibawakan oleh satu orang pembicara penuh).*

---

## SLIDE 1: Cover / Judul Proyek
**[Action: Tampilkan Slide 1. Berdiri tegak, tersenyum, dan tatap dewan penguji]**

> **Naskah:**  
> "Selamat pagi/siang Bapak dan Ibu dewan penguji yang kami hormati.  
> 
> Terima kasih atas waktu dan kesempatan yang diberikan kepada kami. Pada kesempatan kali ini, kami dari **Kelompok 9** akan mempresentasikan rancangan proposal tugas akhir mata kuliah Visi Komputer dengan judul:  
> **'Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya Menggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif'**.  
> 
> Tim kami terdiri dari dua orang dengan pembagian fokus yang jelas:  
> Rekan saya, **Krisantus Linardi**, bertindak sebagai *Data and Preprocessing Lead* yang bertanggung jawab atas audit dataset, kurasi *class imbalance*, dan pipeline peningkatan citra.  
> Dan saya sendiri, **Kristofer Leonard**, bertindak sebagai *Model and Deployment Lead* yang bertanggung jawab atas pelatihan arsitektur deteksi, benchmarking metrik, serta penerapan prototype aplikasi.  
> 
> Mari kita masuk ke latar belakang mengapa topik ini sangat penting untuk diangkat."

---

## SLIDE 2: Latar Belakang & Urgensi Masalah (The "Why")
**[Action: Ganti ke Slide 2. Arahkan perhatian ke banner merah di bagian atas]**

> **Naskah:**  
> "Bapak dan Ibu penguji, mari kita lihat fakta di lapangan.  
> 
> Berdasarkan laporan keselamatan resmi dari **NHTSA (*National Highway Traffic Safety Administration*)** dan basis data **IIHS**, secara konsisten tercatat bahwa **sekitar 76% kecelakaan fatal pejalan kaki terjadi pada kondisi gelap atau malam hari**. Sebagai perbandingan, kondisi siang hari hanya menyumbang kurang dari 20%.  
> 
> Angka ini menegaskan satu hal penting: pejalan kaki adalah *Vulnerable Road Users*—pengguna jalan yang paling rentan karena tidak memiliki perlindungan fisik layaknya pengendara mobil. Di malam hari, jarak pandang pengemudi menurun drastis, pakaian pejalan kaki yang gelap sering kali menyatu dengan aspal, dan pencahayaan lampu jalan sangat terbatas.  
> 
> Sementara itu, solusi sensor canggih seperti LiDAR atau kamera inframerah termal (*Thermal IR*) berbiaya sangat mahal untuk dapat diintegrasikan pada kendaraan massal. Oleh karena itu, industri mobilitas cerdas (*Intelligent Transportation Systems*) sangat membutuhkan solusi visi komputer berbasis kamera RGB standar yang terjangkau, namun mampu bekerja andal dan cepat di kondisi malam hari."

---

## SLIDE 3: Permasalahan Utama & Gap Analysis
**[Action: Ganti ke Slide 3. Jelaskan ketiga kendala secara berurutan, lalu soroti analisis kesenjangan di bawah]**

> **Naskah:**  
> "Mengapa kamera dan model deteksi standar saat ini gagal bekerja dengan baik di malam hari? Ada **tiga hambatan utama**:  
> 
> 1. **Kontras Ekstrem dan Siluet Gelap:** Pada kondisi minim cahaya, nilai intensitas piksel tubuh pejalan kaki sangat dekat dengan latar belakang. Fitur tepi (*edge*) dan tekstur tubuh hilang tertelan kegelapan, sehingga model sering mengalami *False Negative*—pejalan kaki terlewat sama sekali.  
> 2. **Derau Sensor dan Silau Lampu:** Saat kamera mencoba menaikkan ISO untuk menerangi gambar, muncul *grain noise* yang mengaburkan kontur. Di saat yang sama, lampu sorot kendaraan dari arah berlawanan (*headlight glare*) menimbulkan *over-exposure* lokal yang ekstrem.  
> 3. **Bias Domain Siang Hari:** Model deteksi populer seperti YOLO yang dilatih dengan bobot pre-trained COCO, mayoritas datanya adalah kondisi siang hari, sehingga bobot filternya bias terhadap warna dan pencahayaan terang.  
> 
> Jika kita melihat solusi yang ada sebelumnya:  
> Model restorasi berbasis deep learning seperti EnlightenGAN atau LLNet memang menghasilkan visual yang cerah, tetapi komputasinya sangat lambat—di bawah 5 FPS—dan membutuhkan VRAM GPU besar, sehingga mustahil diterapkan di perangkat *edge* kendaraan.  
> Sebaliknya, filter histogram konvensional global justru memperparah silau lampu mobil dan membuat noise malam semakin liar.  
> 
> Dari sinilah kami menemukan celah penelitian (*sweet spot*): kita membutuhkan teknik prapemrosesan adaptif lokal yang sangat ringan, dipadukan dengan detektor modern yang ultra-cepat."

---

## SLIDE 4: Solusi yang Diusulkan (Lightweight Pipeline)
**[Action: Ganti ke Slide 4. Jelaskan alur secara runut dari Tahap 1, 2, hingga 3]**

> **Naskah:**  
> "Solusi yang kami usulkan dirancang sebagai sebuah **Pipeline 3 Tahap yang Ringkas dan Efisien**:  
> 
> * **Tahap Pertama (Input):** Menerima frame citra jalan raya beresolusi 640x640 piksel. Citra dikonversi ke ruang warna **CIE L\*a\*b\*** untuk memisahkan kanal luminansi atau tingkat terang-gelap (kanal-L) dari kanal warna (a dan b). Hal ini krusial agar manipulasi kontras tidak merusak saturasi dan keseimbangan warna alami.  
> * **Tahap Kedua (Prapemrosesan Pencahayaan Adaptif):**  
>   Kami menerapkan **L-channel CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) dengan pembagian grid lokal 8x8 dan *clip limit* 2.0 untuk mempertegas kontur pejalan kaki tanpa mengamplifikasi noise gelap. Dipadukan dengan **Gamma Correction non-linear (gamma = 0.6)** untuk mengangkat kecerahan *mid-tone* tanpa membuat area lampu sorot menjadi jenuh (*white-out*).  
> * **Tahap Ketiga (Deteksi Objek):**  
>   Citra yang telah dioptimalkan langsung diumpankan ke detektor **Ultralytics YOLOv8-Nano**. Kami memilih varian Nano karena berbobot ultra-ringan—hanya sekitar 3,2 juta parameter. Head deteksinya bersifat *anchor-free*, sehingga sangat fleksibel mendeteksi variasi rasio tubuh pejalan kaki yang dinamis.  
> 
> Hasilnya adalah sistem yang mampu menjaga *throughput* di atas 20 FPS pada CPU laptop biasa."

---

## SLIDE 5: Filosofi Metrik: Prioritas Recall
**[Action: Ganti ke Slide 5. Gunakan intonasi yang tegas untuk menekankan aspek 'Safety-First']**

> **Naskah:**  
> "Salah satu poin terpenting yang membedakan proposal kami dari proyek deteksi biasa adalah **filosofi evaluasi metriknya**. Kami secara sadar menempatkan **Recall Pejalan Kaki** sebagai prioritas tertinggi di atas metrik mAP agregat biasa.  
> 
> Mengapa demikian?  
> Pertama, karena **Asimetri Risiko Keselamatan**:  
> Jika terjadi *False Positive*—misalnya tiang atau bayangan pohon salah dideteksi sebagai orang—dampaknya hanyalah peringatan dini defensif di dasbor.  
> Namun, jika terjadi *False Negative*—pejalan kaki nyata di depan mobil GAGAL dideteksi—maka tidak ada pengereman otomatis, dan dampaknya adalah tabrakan fatal. Dalam sistem ADAS, keselamatan nyawa bergantung pada tingginya angka Recall.  
> 
> Kedua, adanya **Jebakan Class Imbalance**:  
> Dari audit data yang kami lakukan, objek kendaraan berjumlah 6.871 box (88,4%), sedangkan pejalan kaki hanya 906 box (11,6%)—rasio ketimpangannya mencapai **7,58 banding 1**.  
> Jika kami hanya mengejar nilai mAP atau Akurasi keseluruhan, model yang hanya pintar mendeteksi mobil tetapi buta pejalan kaki masih bisa mendapatkan nilai mAP yang terlihat tinggi secara semu.  
> 
> Oleh karena itu, target eksplisit proyek ini adalah **menaikkan Recall pejalan kaki sebesar minimal 10%** dibanding baseline standar."

---

## SLIDE 6: Kesiapan Data & Bukti Eksplorasi Awal
**[Action: Ganti ke Slide 6. Tunjuk Gambar 1 di sisi kanan untuk memperlihatkan progres nyata]**

> **Naskah:**  
> "Proposal ini tidak berangkat dari sekadar rencana di atas kertas, melainkan telah didukung oleh **fondasi data yang matang dan terverifikasi penuh** pada Checkpoint awal (CP03).  
> 
> Kami menggunakan dataset **Roboflow Low-light v2** berlisensi terbuka (CC BY 4.0) dengan total **2.400 citra dan 7.777 bounding box**. Melalui skrip audit otomatis yang telah kami bangun (`src/audit_dataset.py`), kami memverifikasi bahwa:  
> 1. Integritas anotasi 100% sempurna: tidak ada label korup, koordinat NaN, atau berkas tanpa pasangan.  
> 2. Di sebelah kanan, Bapak dan Ibu dapat melihat **Gambar 1**, yaitu sampel visual hasil audit kami di mana bounding box hijau menandai pejalan kaki dan biru menandai kendaraan pada kondisi pencahayaan malam yang nyata.  
> 3. Yang terpenting, kami menerapkan protokol **Zero Data Leakage**: data telah dipisah secara fisik ke folder Train (2.100 citra), Validasi (200 citra), dan Test (100 citra). Seluruh kalkulasi histogram CLAHE dan augmentasi HANYA diterapkan pada data latih, sementara data uji dijaga murni dan tidak pernah terlihat oleh model selama proses pelatihan."

---

## SLIDE 7: Metodologi Eksperimen Terkontrol (Ablation Study)
**[Action: Ganti ke Slide 7. Tegaskan bahwa model dikunci pada YOLOv8-Nano]**

> **Naskah:**  
> "Untuk membuktikan efektivitas metode yang kami ajukan secara ilmiah, kami merancang **4 skenario eksperimen komparatif terkontrol**.  
> 
> Kami ingin menegaskan bahwa di sini **kami tidak mengganti-ganti arsitektur model**, melainkan mengunci arsitektur pada **YOLOv8-Nano** agar pengujian terbebas dari *confounding variable* (variabel perancu). Eksperimen ini murni menguji perlakuan augmentasi pencahayaan (*Ablation Study*):  
> 
> * **B0 (Baseline Zero-Shot):** Model YOLOv8n bobot COCO diuji langsung pada data malam tanpa pelatihan ulang, untuk mengukur seberapa parah kegagalan deteksi bawaannya.  
> * **Exp-A (Fine-Tuning Standar):** YOLOv8n dilatih pada data malam dengan augmentasi standar tanpa manipulasi pencahayaan khusus. Ini menjadi *control group* kami.  
> * **Exp-B (Usulan Utama):** YOLOv8n dilatih dengan prapemrosesan **L-channel CLAHE**. Di sinilah kami menguji hipotesis target lonjakan Recall pejalan kaki > 10%.  
> * **Exp-C (Pembanding Non-Linear):** YOLOv8n dilatih dengan **Gamma Correction (gamma = 0.6)** sebagai pembanding komparatif penyesuaian kurva luminansi.  
> 
> Keempat eksperimen ini dievaluasi pada test set yang identik, dengan random seed 42 dan hyperparameter yang terstandarisasi penuh."

---

## SLIDE 8: Rencana Deployment & Target Deliverables
**[Action: Ganti ke Slide 8. Paparkan aspek implementasi teknis dan prototype]**

> **Naskah:**  
> "Beralih ke rencana implementasi dan luaran proyek. Proyek ini diarahkan pada ketercapaian sistem yang aplikatif:  
> 
> * **Lingkungan Pelatihan:** Kami memanfaatkan GPU NVIDIA Tesla T4 (Google Colab 16 GB VRAM) berbasis framework PyTorch dan Ultralytics, yang memungkinkan pelatihan 50 epoch selesai secara efisien dan terdokumentasi.  
> * **Perangkat Target Inferensi:** Kami menargetkan pengujian langsung pada **CPU Laptop Standar (Intel Core i5/i7)** tanpa GPU tambahan, dengan target *throughput* stabil pada **20 hingga 25 FPS**. Ini membuktikan kepraktisan sistem untuk diaplikasikan pada perangkat *edge* atau kamera lalu lintas cerdas.  
> * **Prototype Aplikasi Interaktif:** Kami akan mengemas model terbaik ke dalam aplikasi interaktif berbasis **Streamlit dan OpenCV**. Antarmuka ini memungkinkan penguji melihat demonstrasi live *split-screen*: sisi kiri menampilkan citra malam asli, dan sisi kanan menampilkan citra yang telah ditingkatkan kontrasnya beserta hasil deteksi bounding box dan indikator FPS real-time.  
> * **Luaran Akademik:** Laporan teknis lengkap berstandar IEEE 2-kolom, repositori kode modular yang *reproducible*, serta analisis kegagalan (*failure cases*)."

---

## SLIDE 9: Pembagian Peran & Roadmap Eksekusi
**[Action: Ganti ke Slide 9. Tunjukkan pembagian tugas tim yang seimbang dan jadwal]**

> **Naskah:**  
> "Agar proyek ini dapat selesai tepat waktu menuju Ujian Akhir Semester (UAS), kami telah membagi tanggung jawab kerja secara jelas:  
> 
> * **Krisantus Linardi:** Memimpin *Data Strategy & Preprocessing Pipeline*—fokus pada audit data, penanganan class imbalance, implementasi modul CLAHE dan Gamma, serta validasi pencegahan *data leakage*.  
> * **Kristofer Leonard:** Memimpin *Model Architecture & Deployment*—fokus pada eksekusi pelatihan 4 varian eksperimen di Colab, benchmarking throughput FPS, dan pembangunan prototype antarmuka Streamlit.  
> 
> Di bagian bawah terlihat **Roadmap Eksekusi** kami:  
> * **Fase 1 (UTS):** Telah selesai 100%, meliputi audit dataset 2.400 citra, penyusunan proposal IEEE, dan setup repositori awal.  
> * **Fase 2 (Minggu 8–11):** Menjalankan pelatihan keempat eksperimen (B0 hingga Exp-C) dan pencatatan log metrik.  
> * **Fase 3 (Minggu 12–14):** Analisis kasus gagal (*error slices*), integrasi prototype demo, dan pengujian latency CPU.  
> * **Fase 4 (UAS):** Finalisasi laporan teknis lengkap dan demonstrasi aplikasi."

---

## SLIDE 10: Mengapa Proyek Ini Sangat Layak Diterima? (Why Approve)
**[Action: Ganti ke Slide 10. Bicara dengan intonasi mantap dan meyakinkan]**

> **Naskah:**  
> "Sebagai kesimpulan dari proposal kami, ada **4 pilar utama mengapa proyek ini sangat layak untuk disetujui**:  
> 
> 1. **Relevansi dan Urgensi Masalah:** Kami menyelesaikan masalah nyata keselamatan transportasi (*safety-critical task*) yang memiliki dampak sosial tinggi untuk mencegah kecelakaan fatal malam hari.  
> 2. **Metodologi Ilmiah yang Kokoh:** Dataset telah diaudit tuntas, isolasi data murni tanpa kebocoran, dan desain 4 eksperimen *Ablation Study* dirancang bersih dan terukur.  
> 3. **Realistis dan Bebas Over-Engineering:** Kami tidak memaksakan model generatif yang berat. Pilihan arsitektur YOLOv8-Nano (~3,2M parameter) dan CLAHE sangat terukur, efisien, dan siap dioperasikan di CPU laptop tanpa membutuhkan infrastruktur komputasi raksasa.  
> 4. **Luaran Terukur dan Jelas:** Parameter keberhasilan kami terdefinisi konkret, yaitu kenaikan Recall pejalan kaki di atas 10%, kecepatan inferensi minimal 20 FPS, dan prototipe aplikasi yang siap didemokan langsung di depan penguji."

---

## SLIDE 11: Penutup & Tanya Jawab (Q&A)
**[Action: Ganti ke Slide 11. Senyum, lakukan kontak mata hangat ke seluruh penguji]**

> **Naskah:**  
> "Demikian pemaparan proposal tugas akhir dari Kelompok 9.  
> 
> Kami sangat berterima kasih atas perhatian Bapak dan Ibu dewan penguji. Sekarang, kami sangat siap dan terbuka untuk menerima pertanyaan, saran, serta masukan yang membangun demi kesempurnaan proyek ini.  
> 
> Terima kasih, selamat pagi/siang."

---

## 💡 Contekan Cepat Menjawab Pertanyaan Sulit (Cheat Sheet Q&A)

1. **Pertanyaan: "Kenapa tidak pakai model pencerahan malam yang lebih canggih seperti EnlightenGAN atau Zero-DCE?"**  
   > *Jawaban:* "Model deep generatif seperti EnlightenGAN memiliki ukuran model puluhan megabyte dengan latensi pemrosesan di atas 200 ms per frame (< 5 FPS), sehingga tidak realistis untuk sistem ADAS bergerak yang butuh respon milidetik. Prapemrosesan CLAHE L-channel berbasis operasi citra teroptimasi yang hanya membutuhkan hitungan milidetik, sehingga target 20–25 FPS di CPU tetap tercapai."

2. **Pertanyaan: "Kenapa memilih YOLOv8-Nano, bukan varian Small, Medium, atau Large?"**  
   > *Jawaban:* "Sesuai prinsip anti-over-engineering, kendala utama deteksi pejalan kaki malam hari bukan pada kekurangan kapasitas representasi neural network, melainkan pada hilangnya kontras fitur visual pada citra masukan. Dengan memperbaiki kontras siluet di tahap prapemrosesan, model ringan (~3,2M parameter) sudah sangat cukup untuk mengekstraksi bounding box secara akurat dan jauh lebih cepat."

3. **Pertanyaan: "Dataset sangat timpang (mobil 88%, orang 11%), bagaimana mencegah model overfit ke mobil?"**  
   > *Jawaban:* "Itulah alasan kami tidak memakai mAP agregat sebagai tolok ukur utama. Kami membedah evaluasi per kelas dan mematok target keberhasilan pada **Recall kelas Person**. Pada fungsi loss YOLOv8 (CIoU + DFL + BCE), bobot gradien regresi bounding box pejalan kaki yang kontrasnya sudah ditingkatkan oleh CLAHE akan membantu model memprioritaskan fitur tubuh pejalan kaki."
