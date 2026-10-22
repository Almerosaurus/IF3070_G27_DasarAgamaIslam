import random
import time

from hill_climbing_basics import max_possible_fitness
from hill_climbing_steepest import steepest_ascent

def random_restart(vehicles, ship, max_restart=10, printall=False, seed=None):
    if max_restart < 0:
        raise ValueError("max_restart harus bernilai non-negatif")

    rng = random.Random(seed)
    start_time = time.perf_counter()
    upper_bound = max_possible_fitness(vehicles)

    initial_state = None
    best_state = None
    best_value = None
    obj_history = []
    iterasi_per_restart = []
    restart = 0

    while True:
        if printall:
            print(f"--- Percobaan ke-{restart + 1} (restart ke-{restart}) ---")

        run_initial, run_state, run_value, run_history, run_iterasi, _ = steepest_ascent(
            vehicles, ship, printall=printall, seed=rng.randrange(2**32))

        if initial_state is None:
            initial_state = run_initial
        obj_history.extend(run_history)
        iterasi_per_restart.append(run_iterasi)

        if best_value is None or run_value > best_value:
            best_state, best_value = run_state, run_value

        if printall:
            print("Nilai objective percobaan ini:", run_value)
            print("Nilai objective terbaik sejauh ini:", best_value)

        if best_value >= upper_bound:
            if printall:
                print("Nilai objective sudah mencapai batas atas.")
            break
        if restart >= max_restart:
            if printall:
                print("Banyak restart sudah mencapai maksimum.")
            break
        restart += 1

    duration = time.perf_counter() - start_time
    if printall:
        print("State akhir:", best_state)
        print("Nilai objective akhir:", best_value)
        print("Banyak restart:", restart)
        print("Banyak iterasi per restart:", iterasi_per_restart)

    return (initial_state, best_state, best_value, obj_history,
            restart, iterasi_per_restart, duration)
