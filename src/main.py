#!/usr/bin/env python3

from hill_climbing_basics import Vehicle, Ship, compute_fitness
from hill_climbing_steepest import steepest_ascent


# ============================================================
# 1. FUNGSI INPUT
# ============================================================

def input_integer(prompt, minimum=0):
    while True:
        try:
            value = int(input(prompt))
            if value < minimum:
                print(f"Nilai harus minimal {minimum}.")
                continue
            return value
        except ValueError:
            print("Input tidak valid. Masukkan bilangan bulat.")


def input_vehicle_data():
    jumlah_kendaraan = input_integer("Masukkan jumlah kendaraan: ", minimum=1)
    vehicles = []
    ids = set()

    for i in range(jumlah_kendaraan):
        print(f"\nData kendaraan ke-{i + 1}")
        while True:
            vehicle_id = input("ID kendaraan: ").strip()
            if not vehicle_id:
                print("ID kendaraan tidak boleh kosong.")
            elif vehicle_id in ids:
                print("ID kendaraan harus unik.")
            else:
                ids.add(vehicle_id)
                break

        width = input_integer("Lebar kendaraan (grid): ", minimum=1)
        length = input_integer("Panjang kendaraan (grid): ", minimum=1)
        fee = input_integer("ShippingFee: ", minimum=0)
        weight = input_integer("Berat kendaraan: ", minimum=0)
        eta = input_integer("ETA (hari): ", minimum=0)

        vehicles.append(Vehicle(vehicle_id, width, length, fee, weight, eta))

    return vehicles


def input_ship_data():
    print("Masukkan ukuran geladak kapal dan kapasitas angkut.")
    width = input_integer("Lebar geladak (grid): ", minimum=1)
    length = input_integer("Panjang geladak (grid): ", minimum=1)
    max_capacity = input_integer("Kapasitas maksimum berat: ", minimum=1)
    return Ship(width, length, max_capacity)


# ============================================================
# 2. FUNGSI KELUARAN
# ============================================================

def print_state(title, state, vehicles):
    print(f"\n{title}")
    for i, entry in enumerate(state):
        vehicle = vehicles[i]
        if entry is None:
            print(f"  {vehicle.id}: di luar kapal")
        else:
            x, y, orientation = entry
            orientation_name = "Horizontal" if orientation == 'H' else "Vertical"
            print(f"  {vehicle.id}: posisi=({x}, {y}), orientasi={orientation_name}")


def print_trial_result(trial, vehicles, ship, result):
    print("\n" + "=" * 60)
    print(f"PERCOBAAN KE-{trial}")
    print("=" * 60)

    initial, final, value, history, iterations, duration = result
    print_state("State awal:", initial, vehicles)
    print_state("State akhir:", final, vehicles)
    print("Banyak iterasi:", iterations)

    print("Nilai objective akhir:", value)
    print("Durasi proses pencarian:", round(duration, 6), "detik")
    print("Riwayat objective:", history)
    print("Nilai objective awal:", compute_fitness(initial, vehicles, ship))
    return value, iterations, duration


# ============================================================
# 3. PROGRAM UTAMA
# ============================================================

def main():
    ship = input_ship_data()
    vehicles = input_vehicle_data()
    max_iter = input_integer("Masukkan maksimum iterasi: ", minimum=1)

    print("\nProgram menjalankan algoritma sebanyak 3 percobaan.")
    all_results = []

    for trial in range(1, 4):
        result = steepest_ascent(vehicles, ship, max_iter=max_iter)
        all_results.append(
            print_trial_result(trial, vehicles, ship, result)
        )

    print("\n" + "=" * 60)
    print("RINGKASAN 3 PERCOBAAN")
    print("=" * 60)
    for i, (value, iterations, duration) in enumerate(all_results, start=1):
        print(f"Percobaan {i}: objective={value}, iterasi={iterations}, "
              f"durasi={duration:.6f} detik")


if __name__ == "__main__":
    main()
