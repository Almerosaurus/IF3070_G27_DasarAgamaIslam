#!/usr/bin/env python3
import random
import time

from hill_climbing_basics import (
    compute_fitness,
    random_initial_state,
    random_successor,
    state_is_fully_valid,
)

def stochastic_hill_climbing(vehicles, ship, nmax=1000, initial_state=None,
                             seed=None):
    if nmax < 0:
        raise ValueError("nmax harus bernilai non-negatif")

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

    for _ in range(nmax):
        neighbor = random_successor(state, vehicles, ship, rng=rng)
        # Ambil satu successor acak untuk dievaluasi.
        if neighbor is None:
            break

        neighbor_value = compute_fitness(neighbor, vehicles, ship)
        # Berpindah hanya jika nilai successor lebih tinggi.
        if neighbor_value > current_value:
            state = neighbor
            current_value = neighbor_value

        # Hitung setiap successor yang dicoba, termasuk yang ditolak.
        # Simpan nilai objective setelah setiap percobaan.
        obj_history.append(current_value)
        iterasi += 1

        if iterasi % 10 == 0:
            print(f"\r  Iterasi {iterasi:>7d}  cur={current_value:.1f}  "
                  f"best={current_value:.1f}", end="", flush=True)

    duration = time.perf_counter() - start_time
    if iterasi >= 10:
        print()
    print("State akhir:", state)
    print("Nilai objective akhir:", current_value)
    print("Banyak iterasi:", iterasi)

    return initial_state, state, current_value, obj_history, iterasi, duration
