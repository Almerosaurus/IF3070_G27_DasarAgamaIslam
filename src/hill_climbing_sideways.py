#!/usr/bin/env python3
import random
import time

from hill_climbing_basics import (
    compute_fitness,
    generate_successors,
    random_initial_state,
    state_is_fully_valid,
)

def sideways_move(vehicles, ship, initial_state=None, max_sideways=100,
                  printall=False, seed=None):
    if max_sideways < 0:
        raise ValueError("max_sideways harus bernilai non-negatif")

    rng = random.Random(seed)
    start_time = time.perf_counter()
    if initial_state is None:
        initial_state = random_initial_state(vehicles, ship, rng=rng)
    elif not state_is_fully_valid(initial_state, vehicles, ship):
        raise ValueError("initial_state harus valid dan berisi satu entry per kendaraan")

    initial_state = initial_state[:]
    state = initial_state[:]
    current_value = compute_fitness(state, vehicles, ship)
    obj_history = [current_value]
    iterasi = 0
    sideways_count = 0
    # Banyak sideways move berturut-turut sejak perbaikan nilai terakhir.

    while True:
        if printall:
            print("Nilai objective saat ini:", current_value)

        successors = generate_successors(state, vehicles, ship)
        # Menghasilkan semua state tetangga yang mungkin dari state saat ini.
        if not successors:
            break

        if printall:
            print("State dengan successor:")
        successor_values = [compute_fitness(s, vehicles, ship) for s in successors]
        if printall:
            for successor, value in zip(successors, successor_values):
                print(successor, "->", value)

        neighbor_value = max(successor_values)
        # Mengambil nilai objektif tertinggi dari semua state tetangga.
        if neighbor_value < current_value:
            if printall:
                print("Tidak ada state tetangga yang lebih baik atau sama.")
            break

        if neighbor_value == current_value:
            # Sideways move: berpindah ke state tetangga dengan nilai yang sama.
            if sideways_count >= max_sideways:
                if printall:
                    print("Banyak sideways move sudah mencapai maksimum.")
                break
            sideways_count += 1
        else:
            sideways_count = 0

        neighbor_candidates = [s for s, value in zip(successors, successor_values)
                               if value == neighbor_value]
        # Memilih secara acak jika beberapa state tetangga memiliki nilai terbaik.
        state = rng.choice(neighbor_candidates)
        current_value = neighbor_value
        obj_history.append(current_value)
        iterasi += 1
        # Menambah jumlah langkah yang berhasil diterima.

        if printall:
            print("Berpindah ke state:", state)
            print("Nilai objective baru:", current_value)

    duration = time.perf_counter() - start_time
    if printall:
        print("State akhir:", state)
        print("Nilai objective akhir:", current_value)
        print("Banyak iterasi:", iterasi)

    return initial_state, state, current_value, obj_history, iterasi, duration
