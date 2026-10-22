#!/usr/bin/env python3
import random
import time

from hill_climbing_basics import (
    compute_fitness,
    generate_successors,
    random_initial_state,
    state_is_fully_valid,
)

def steepest_ascent(vehicles, ship, initial_state=None, seed=None):
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

    while True:
        successors = generate_successors(state, vehicles, ship)
        # Generate semua state tetangga
        if not successors:
            break

        neighbor_value = max(compute_fitness(s, vehicles, ship) for s in successors)
        # Pilih value tertinggi
        if neighbor_value <= current_value:
            break

        neighbor_candidates = [s for s in successors
                               if compute_fitness(s, vehicles, ship) == neighbor_value]
        # Pilih secara acak jika beberapa memiliki nilai terbaik.
        state = rng.choice(neighbor_candidates)
        current_value = neighbor_value
        obj_history.append(current_value)
        iterasi += 1
        # Catat langkah yang berhasil diterima

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
