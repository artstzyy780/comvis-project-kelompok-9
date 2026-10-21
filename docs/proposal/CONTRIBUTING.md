# Kontrak Kerja Kelompok 09 (Checkpoint CP00)

> Dokumen kerja bersama untuk tiap checkpoint proyek akhir Visi Komputer (IF0PP006).
> Mengisi bukti minimum CP00: nama anggota, kanal komunikasi, peran awal, aturan
> rapat, dan mekanisme penyelesaian konflik.

## 1. Anggota Kelompok

| Nama Lengkap | NIM | Peran | Fokus Tugas Utama |
|:---|:---|:---|:---|
| **Krisantus Amka Ginting** | 412024025 | Data & Preprocessing Lead | Audit dataset, deteksi kebocoran scene, pipeline CLAHE/Gamma, kurasi class imbalance |
| **Linardi Halim** | 212024005 | Documentation & Reporting Lead | Proposal, laporan teknis, naskah presentasi, rekap tracker dan logbook |
| **Kristofer Ryan Giggs Eka Saputra** | 412024025 | Model & Training Lead | Eksperimen terkontrol B0/A/B/C, benchmark latensi dan FPS |
| **Leonard Franscois** | 412024035 | Demo & Deployment Lead | Demo inferensi, dokumentasi repository, video cadangan |

> Catatan: daftar anggota pada dokumen proposal sebelumnya perlu diselaraskan
> dengan tabel di atas (empat anggota), bukan dua nama gabungan.

## 2. Kanal Komunikasi

| Kanal | Kegunaan | Aturan |
|:---|:---|:---|
| Grup WhatsApp kelompok | Koordinasi harian dan pengumuman cepat | Pesan penting di-pin, respons maksimal 12 jam |
| Google Drive bersama | Penyimpanan artefak dan bukti checkpoint | Struktur folder mengikuti `docs/CP_EVIDENCE_MAP.md` |
| GitHub (`artstzyy780/comvis-project-kelompok-9`) | Source code, konfigurasi, laporan otomatis | Commit mengikuti *Conventional Commits* |
| Tracker Excel (`Kelompok_09_Tracker_Proyek_Akhir.xlsm`) | Status checkpoint, verifikasi, skor logbook | Diperbarui sebelum pemeriksaan mingguan |

## 3. Aturan Rapat

| Poin | Ketentuan |
|:---|:---|
| Frekuensi | Minimal satu kali daring per minggu, ditambah pertemuan tambahan bila ada tenggat CP |
| Durasi | Maksimal 90 menit, agenda dikirim H-1 |
| Notulen | Whoever yang menulis notulen wajib mengisi baris logbook minggu tersebut |
| Kuorum | Minimal tiga anggota untuk keputusan yang mengubah scope atau teknologi |
| Keputusan teknis | Diambil berdasarkan hasil eksperimen, bukan preferensi, dan dicatat di logbook |

## 4. Pembagian Tugas dan Bukti

Setiap anggota wajib mengisi tab logbook miliknya sendiri pada tracker
(minggu 3 sampai 15) dengan: pekerjaan nyata, hasil atau keputusan, tautan
bukti, kendala, dan rencana berikutnya. Penilaian tidak berdasarkan jumlah jam
kerja, melainkan bukti yang dapat diperiksa dosen.

## 5. Penyelesaian Konflik

1. Didiskusikan langsung pada rapat mingguan dengan tenggat respons 24 jam.
2. Jika belum selesai, kedua pihak membandingkan data eksperimen, bukan pendapat.
3. Keputusan akhir diambil oleh seluruh anggota dan dicatat di logbook kedua pihak.
4. Isu yang masih buntu dibawa ke dosen atau asisten pada jam konsultasi.

## 6. Prinsip Teknis Anti-Bocoran Data

| Prinsip | Implementasi |
|:---|:---|
| Split data sebelum augmentasi | `src/rebuild_split_dedup.py` (grouped by source scene) |
| Augmentasi hanya pada data latih | `src/prepare_experiments.py` |
| Seed tetap untuk semua eksperimen | `configs/experiments.yaml` (seed 42) |
| Satu split untuk seluruh perbandingan | `src/evaluate_experiments.py --split test` |
| Artefak beku dan deteksi perubahan | `src/freeze_artifacts.py` |
