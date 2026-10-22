# IF3070_GXX_DasarAgamaIslam

Tugas Besar 1 IF3070 Dasar Inteligensi Artifisial 2026/2027

## Deskripsi Singkat

Program ini menyelesaikan masalah penataan kendaraan pada geladak kapal kargo menggunakan algoritma *local search*. Tujuannya adalah memilih kendaraan yang diangkut serta menentukan posisi dan orientasinya agar total `ShippingFee` maksimum, dengan tetap memenuhi batasan berikut:

- Kendaraan tidak boleh saling bertumpuk (no overlapping)
- Seluruh bagian kendaraan harus berada di dalam batas geladak
- Total berat kendaraan tidak boleh melebihi kapasitas kapal

Algoritma yang diimplementasikan:

1. Hill-climbing (varian: `<isi varian yang dipilih>`)
2. Simulated Annealing
3. Genetic Algorithm

## Struktur Repository

```
.
├── src/          # source code
├── docs/         # laporan (.pdf)
├── output/       # hasil run algoritma
├── requirements.txt
└── README.md
```
