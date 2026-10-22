# IF3070_G27_DasarAgamaIslam

Tugas Besar 1 IF3070 Dasar Inteligensi Artifisial 2026/2027

Anggota Kelompok:
- Muhammad Azka Firdaus (18224036)
- Muhammad Almer Zain Farisseno (18224070)
- Fatin Sajid Darwis (18224074)

## Deskripsi Singkat

Program ini menyelesaikan masalah penataan kendaraan pada geladak kapal kargo menggunakan algoritma *local search*. Tujuannya adalah memilih kendaraan yang diangkut serta menentukan posisi dan orientasinya agar total `ShippingFee` maksimum, dengan tetap memenuhi batasan berikut:

- Kendaraan tidak boleh saling bertumpuk (no overlapping)
- Seluruh bagian kendaraan harus berada di dalam batas geladak
- Total berat kendaraan tidak boleh melebihi kapasitas kapal

Algoritma yang diimplementasikan:

1. Hill-climbing 
2. Simulated Annealing
3. Genetic Algorithm

## Struktur Repository

```
.
├── data/
│   └── dataset.json
├── docs/
│   └── report.txt
├── output/
├── src/
│   ├── genetic_algorithm.py
│   ├── hill_climbing_basics.py
│   ├── hill_climbing_random_restart.py
│   ├── hill_climbing_sideways.py
│   ├── hill_climbing_steepest.py
│   ├── hill_climbing_stochastic.py
│   ├── main.py
│   ├── simulated_annealing.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## Prasyarat & Instalasi

Pastikan telah menginstal **Python 3.10+**.

1. **Clone repository dan masuk ke direktori proyek:**
   ```bash
   git clone https://github.com/Almerosaurus/IF3070_G27_DasarAgamaIslam.git
   cd IF3070_G27_DasarAgamaIslam
   ```

2. **(Opsional) Buat dan aktifkan virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan Program (Cara Run)

Program utama dijalankan melalui file `src/main.py`. Terdapat beberapa cara menjalankan program:

### 1. Mode Menu Interaktif (Default)

Jalankan perintah berikut untuk memilih algoritma dan varian secara interaktif melalui terminal menggunakan parameter default:

```bash
python src/main.py
```

Anda akan diminta memilih algoritma:
- `1. Hill-climbing` (dilanjutkan memilih varian: Steepest Ascent, Sideways Move, Random Restart, atau Stochastic)
- `2. Simulated Annealing`
- `3. Genetic Algorithm`

### 2. Mode Kustomisasi Parameter

Tambahkan argumen `--custom` untuk menyesuaikan parameter algoritma (misalnya temperatur awal SA, ukuran populasi GA, batasan sideways move, dll.):

```bash
python src/main.py --custom
```

### 3. Mode Langsung (Command-Line Arguments)

Anda dapat melewati menu interaktif dengan langsung menentukan opsi via CLI:

- **Steepest Ascent Hill-Climbing:**
  ```bash
  python src/main.py --search hc --variant steepest
  ```

- **Hill-Climbing with Sideways Move:**
  ```bash
  python src/main.py --search hc --variant sideways
  ```

- **Random Restart Hill-Climbing:**
  ```bash
  python src/main.py --search hc --variant random_restart
  ```

- **Stochastic Hill-Climbing:**
  ```bash
  python src/main.py --search hc --variant stochastic
  ```

- **Simulated Annealing:**
  ```bash
  python src/main.py --search sa
  ```

- **Genetic Algorithm:**
  ```bash
  python src/main.py --search ga
  ```

#### Opsi Tambahan CLI

| Flag | Nilai | Deskripsi |
|---|---|---|
| `--search` | `hc`, `sa`, `ga` | Menentukan algoritma utama |
| `--variant` | `steepest`, `sideways`, `random_restart`, `stochastic` | Menentukan varian Hill-climbing (wajib jika `--search hc` tanpa interaktif) |
| `--custom` | *(flag)* | Mengaktifkan prompt pengaturan parameter manual |
| `--runs` | Integer (default: `3`) | Jumlah pengujian run independen yang dieksekusi |
| `--seed` | Integer | Menentukan seed dasar untuk reproduksibilitas hasil pencarian |

*Contoh menjalankan 3 run Simulated Annealing dengan seed tertentu:*
```bash
python src/main.py --search sa --seed 42 --runs 3
```

## Pembagian Tugas

| NIM | Nama | Pembagian Tugas |
|:---:|:---|:---|
| 18224036 | Muhammad Azka Firdaus | Seluruh algoritma Hill Climbing, Laporan  |
| 18224070 | Muhammad Almer Zain Farisseno | Simulated Annealing, Main, Laporan|
| 18224074 | Fatin Sajid Darwis | Genetic Algorithm, Visualization, Laporan|
