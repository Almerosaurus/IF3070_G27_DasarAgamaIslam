#!/usr/bin/env python3
import random
import time

from hill_climbing_basics import (
    compute_fitness,
    generate_successors,
    random_initial_state,
    state_is_fully_valid,
)

def stochastic_hill_climbing(vehicles, ship, initial_state=None, max_iter=None,
                             printall=False, seed=None):
    if max_iter is not None and max_iter < 0:
        raise ValueError("max_iter harus bernilai non-negatif")

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

    while max_iter is None or iterasi < max_iter:
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

        uphill_candidates = [(s, value) for s, value in zip(successors, successor_values)
                             if value > current_value]
        # Mengambil semua state tetangga yang lebih baik dari state saat ini.
        if not uphill_candidates:
            if printall:
                print("Tidak ada state tetangga yang lebih baik.")
            break

        # Memilih secara acak salah satu state tetangga yang lebih baik,
        # tidak harus yang terbaik.
        state, current_value = rng.choice(uphill_candidates)
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
