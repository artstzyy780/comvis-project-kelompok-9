# 🔒 Daftar Artefak Beku (Frozen Deliverables)

- **Waktu pembekuan:** 2026-10-20T14:39:04.904992+00:00
- **Commit Git:** `4869ddd`
- **Jumlah artefak:** 8

| Artefak | Pemenuhan Checkpoint | Ukuran | SHA-256 (12 char) | Status |
|:---|:---|:---:|:---:|:---:|
| `K09_DeteksiPedestrianLowLight_Proposal_v02.pdf` | CP06 - Proposal UTS (versi beku) | 132.7 KB | `30201ac7b073` | ✅ ada |
| `docs/proposal/K09_Sidang_Proposal_DeteksiPedestrianLowLight.pptx` | CP06 - Slide Sidang UTS | 3427.0 KB | `6f507c6022b1` | ✅ ada |
| `docs/proposal/Naskah_Presentasi_Sidang_Proposal_K09.md` | CP06 - Naskah presentasi | 16.5 KB | `d37bf1cecbf0` | ✅ ada |
| `results/audit/audit_report.md` | CP03 - Laporan audit dataset | 4.1 KB | `9a8870f1ba9f` | ✅ ada |
| `results/audit/dedup_split_report.md` | CP07 - Split bebas-leakage (scene level) | 2.2 KB | `630724084ef7` | ✅ ada |
| `results/experiments/unified_results.md` | CP10 - Tabel hasil terpadu (satu split) | - | `-` | ❌ belum ada |
| `results/experiments/benchmark_latency_report.md` | CP12 - Benchmark latensi & FPS | 3.6 KB | `bd8c2716d1e1` | ✅ ada |
| `README.md` | CP13 - README final | 10.9 KB | `14891fe70a45` | ✅ ada |

## Verifikasi Integritas

Jalankan `python src/freeze_artifacts.py --check` untuk memastikan artefak tidak berubah setelah pembekuan. Perbedaan digest menandakan berkas diedit setelah freeze.

> **Peringatan:** 1 artefak belum tersedia:

> - `results/experiments/unified_results.md`
