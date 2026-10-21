# Laporan Rebuild Split Bebas-Leakage (Grouped by Source Scene)

> Bukti untuk Checkpoint CP07 (repository & pipeline data) dan CP11 (hipotesis + satu perbaikan yang diuji).

## 1. Masalah pada Split Asli

Eksport Roboflow menempatkan augmented copy dari satu scene ke beberapa split sekaligus, sehingga foto yang sama muncul pada train dan test.

| Split | File total | File dari scene bocor | Persentase |
|:---|:---:|:---:|:---:|
| **train** | 2,100 | 468 | 22.3% |
| **valid** | 200 | 104 | 52.0% |
| **test** | 100 | 55 | 55.0% |

- Scene sumber unik pada dataset: **687**
- Scene yang muncul pada lebih dari satu split: **145**

## 2. Strategi Rebuild

- **Unit split**: *source scene* (bukan file), sehingga seluruh augmented copy dari satu scene selalu berada pada split yang sama.
- **Stratifikasi**: scene dikelompokkan berdasarkan keberadaan pejalan kaki agar sinyal kelas minoritas tersebar merata.
- **Seed**: `42` sehingga pembagian bersifat reproducible.
- **Rasio target**: {"train": 0.7, "valid": 0.15, "test": 0.15}

## 3. Hasil Split Baru

| Split | Citra | Scene | Box | Vehicle | person | Citra berpejalan kaki |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **train** | 1,643 | 481 | 5,336 | 4,687 | 649 | 336 |
| **valid** | 384 | 103 | 1,368 | 1,190 | 178 | 73 |
| **test** | 373 | 103 | 1,073 | 994 | 79 | 57 |

- **Total**: 2,400 citra | 7,777 box (Vehicle 6,871 / person 906).
- **Jumlah scene unik**: 687.

## 4. Verifikasi Anti-Leakage

Setiap scene hanya boleh muncul pada satu split. Pemeriksaan ulang dilakukan oleh `src/scene_utils.detect_cross_split_leakage`.

| Split | File total | File dari scene bocor | Status |
|:---|:---:|:---:|:---:|
| **train** | 1,643 | 0 | ✅ bebas bocor |
| **valid** | 384 | 0 | ✅ bebas bocor |
| **test** | 373 | 0 | ✅ bebas bocor |

**Total scene bocor setelah rebuild: 0** (sebelumnya 145).

## 5. Mengapa Ini Penting

Dengan split bebas-leakage, selisih Recall antar varian eksperimen benar-benar mencerminkan perbedaan model dan bukan karena scene yang sama sudah terlihat saat pelatihan. Selain itu, jumlah kotak pejalan kaki pada split uji naik dari 39 menjadi 79, sehingga interval keyakinan Recall menjadi jauh lebih sempit.