# Ringkasan Eksperimen Genetic Algorithm

## Eksperimen A — populasi tetap, variasi banyak iterasi

| Konfigurasi | Run | Seed | Nilai awal | Nilai akhir | Terbaik selama run | Iterasi | Durasi (s) |
|---|---|---|---|---|---|---|---|
| pop=50, iter=50 | 1 | 411583 | 765 | 740 | 765 | 50 | 0.52 |
| pop=50, iter=50 | 2 | 411584 | 847 | 847 | 847 | 50 | 0.51 |
| pop=50, iter=50 | 3 | 411585 | 819 | 756 | 819 | 50 | 0.51 |
| pop=50, iter=100 | 1 | 411683 | 869 | 641 | 869 | 100 | 0.62 |
| pop=50, iter=100 | 2 | 411684 | 826 | 634 | 826 | 100 | 0.60 |
| pop=50, iter=100 | 3 | 411685 | 826 | 777 | 826 | 100 | 0.60 |
| pop=50, iter=200 | 1 | 411783 | 819 | 575 | 819 | 200 | 0.78 |
| pop=50, iter=200 | 2 | 411784 | 819 | 597 | 819 | 200 | 0.75 |
| pop=50, iter=200 | 3 | 411785 | 765 | 742 | 765 | 200 | 0.84 |

| Konfigurasi | Rata-rata | Std | Max | Min | Rata-rata durasi (s) |
|---|---|---|---|---|---|
| pop=50, iter=50 | 781.00 | 47.12 | 847 | 740 | 0.51 |
| pop=50, iter=100 | 684.00 | 65.82 | 777 | 634 | 0.61 |
| pop=50, iter=200 | 638.00 | 74.09 | 742 | 575 | 0.79 |

## Eksperimen B — iterasi tetap, variasi jumlah populasi

| Konfigurasi | Run | Seed | Nilai awal | Nilai akhir | Terbaik selama run | Iterasi | Durasi (s) |
|---|---|---|---|---|---|---|---|
| pop=20, iter=100 | 1 | 411583 | 745 | 745 | 745 | 100 | 0.25 |
| pop=20, iter=100 | 2 | 411584 | 819 | 531 | 819 | 100 | 0.24 |
| pop=20, iter=100 | 3 | 411585 | 779 | 742 | 779 | 100 | 0.25 |
| pop=50, iter=100 | 1 | 411683 | 869 | 641 | 869 | 100 | 0.68 |
| pop=50, iter=100 | 2 | 411684 | 826 | 634 | 826 | 100 | 0.60 |
| pop=50, iter=100 | 3 | 411685 | 826 | 777 | 826 | 100 | 0.57 |
| pop=100, iter=100 | 1 | 411783 | 847 | 611 | 847 | 100 | 1.19 |
| pop=100, iter=100 | 2 | 411784 | 819 | 767 | 819 | 100 | 1.22 |
| pop=100, iter=100 | 3 | 411785 | 870 | 740 | 870 | 100 | 1.21 |

| Konfigurasi | Rata-rata | Std | Max | Min | Rata-rata durasi (s) |
|---|---|---|---|---|---|
| pop=20, iter=100 | 672.67 | 100.18 | 745 | 531 | 0.25 |
| pop=50, iter=100 | 684.00 | 65.82 | 777 | 634 | 0.62 |
| pop=100, iter=100 | 706.00 | 68.07 | 767 | 611 | 1.20 |
