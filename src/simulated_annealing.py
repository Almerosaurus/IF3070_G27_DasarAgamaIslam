#!/usr/bin/env python3
import random
import math
import json
import sys
import itertools
from typing import List, Optional, Tuple, Dict, Set

# ============================================================
# 1. MODEL DATA
# ============================================================

class Vehicle:
    def __init__(self, vid, w, l, fee, weight, eta):
        self.id     = vid
        self.w      = w      # width  (grid)
        self.l      = l      # length (grid)
        self.fee    = fee    # ShippingFee (juta rupiah)
        self.weight = weight
        self.eta    = eta
    def __repr__(self):
        return f"Vehicle(id={self.id!r}, {self.w}x{self.l}, fee={self.fee})"


class Ship:
    def __init__(self, w, l, max_capacity):
        self.w       = w
        self.l       = l
        self.max_cap = max_capacity
    def __repr__(self):
        return f"Ship({self.w}x{self.l}, cap={self.max_cap})"


# ============================================================
# 2. FUNGSI BANTUAN
# ============================================================

def occupied_cells(x, y, ori, veh):
    """
    Set koordinat (col,row) yang ditempati 'veh' di (x,y) ori.
    H : lebar = veh.w (ke arah x), tinggi = veh.l (ke arah y)
    V : lebar = veh.l (ke arah x), tinggi = veh.w (ke arah y)
    """
    cw, ch = (veh.w, veh.l) if ori == 'H' else (veh.l, veh.w)
    return {(x + dx, y + dy) for dx in range(cw) for dy in range(ch)}


def is_placement_valid(state, veh_idx, vehicles, ship):
    entry = state[veh_idx]
    if entry is None:
        return True
    x, y, ori = entry
    veh = vehicles[veh_idx]
    cw, ch = (veh.w, veh.l) if ori == 'H' else (veh.l, veh.w)

    if x < 0 or y < 0:
        return False
    if x + cw > ship.w or y + ch > ship.l:
        return False

    my_cells = occupied_cells(x, y, ori, veh)
    for j, ej in enumerate(state):
        if j == veh_idx or ej is None:
            continue
        if my_cells & occupied_cells(ej[0], ej[1], ej[2], vehicles[j]):
            return False

    if total_weight(state, vehicles) > ship.max_cap:
        return False
    return True


def state_is_fully_valid(state, vehicles, ship):
    return all(is_placement_valid(state, i, vehicles, ship)
               for i in range(len(vehicles)))


def total_weight(state, vehicles):
    return sum(vehicles[i].weight for i, e in enumerate(state)
               if e is not None)


def compute_fitness(state, vehicles, ship):
    if not state_is_fully_valid(state, vehicles, ship):
        return 0.0
    return sum(vehicles[i].fee for i, e in enumerate(state)
               if e is not None)

def schedule(t, T0, alpha=0.995, T_min=0.001):
    T = T0 * alpha ** (t - 1)
    return T if T >= T_min else 0.0


# ============================================================
# 3. RANDOM INITIAL STATE
# ============================================================

def random_initial_state(vehicles, ship, tries=500):
    state = [None] * len(vehicles)
    order = list(range(len(vehicles)))
    random.shuffle(order)

    for i in order:
        veh = vehicles[i]
        for _ in range(tries):
            ori = random.choice(['H', 'V'])
            cw, ch = (veh.w, veh.l) if ori == 'H' else (veh.l, veh.w)
            if cw > ship.w or ch > ship.l:
                continue
            x = random.randint(0, ship.w - cw)
            y = random.randint(0, ship.l - ch)
            cand = state[:]
            cand[i] = (x, y, ori)
            if state_is_fully_valid(cand, vehicles, ship):
                state[i] = (x, y, ori)
                break
    return state


# ============================================================
# 4. NEIGHBOR GENERATION (3 Tipe Move)
# ============================================================

def move_swap(state, vehicles, ship):
    n = len(vehicles)
    if n < 2:
        return state[:]
    i, j = random.sample(range(n), 2)
    new_state = state[:]
    new_state[i], new_state[j] = state[j], state[i]
    ok_move = (is_placement_valid(new_state, i, vehicles, ship) and
          is_placement_valid(new_state, j, vehicles, ship) and
          total_weight(new_state, vehicles) <= ship.max_cap)
    return new_state if ok_move else state[:]


def move_relocate(state, vehicles, ship):
    n = len(vehicles)
    i = random.randint(0, n - 1)
    veh = vehicles[i]
    new_state = state[:]

    ori = random.choice(['H', 'V'])
    cw, ch = (veh.w, veh.l) if ori == 'H' else (veh.l, veh.w)
    if cw > ship.w or ch > ship.l:
        new_state[i] = None
        return new_state

    x = random.randint(0, ship.w - cw)
    y = random.randint(0, ship.l - ch)
    cand = new_state[:]
    cand[i] = (x, y, ori)
    if (is_placement_valid(cand, i, vehicles, ship) and
            total_weight(cand, vehicles) <= ship.max_cap):
        new_state[i] = (x, y, ori)
    else:
        new_state[i] = None
    return new_state


def move_rotate(state, vehicles, ship):
    inside = [i for i, e in enumerate(state) if e is not None]
    if not inside:
        return state[:]
    i = random.choice(inside)
    x, y, ori = state[i]
    cand = state[:]
    cand[i] = (x, y, 'V' if ori == 'H' else 'H')
    if (is_placement_valid(cand, i, vehicles, ship) and
            total_weight(cand, vehicles) <= ship.max_cap):
        return cand
    return state[:]


# ============================================================
# 5. SIMULATED ANNEALING
# ============================================================

def simulated_annealing(vehicles, ship, initial_state=None, T0=50.0, seed=None):
    if seed is not None:
        random.seed(seed)

    if initial_state is None:
        initial_state = random_initial_state(vehicles, ship)

    state   = initial_state[:]
    cur_val = compute_fitness(state, vehicles, ship)
    best_state = state[:]
    best_val   = cur_val

    move_funcs = [move_swap, move_relocate, move_rotate]

    for t in itertools.count(1):
        T = schedule(t, T0)
        if T == 0:
                break

        new_state = random.choice(move_funcs)(state, vehicles, ship)
        new_val   = compute_fitness(new_state, vehicles, ship)
        delta     = new_val - cur_val

        if delta >= 0 or random.random() < math.exp(delta / T):
            state, cur_val = new_state, new_val

        if cur_val > best_val:
            best_val   = cur_val
            best_state = state[:]

        if (t + 1) % 10 == 0:
            print(f"\r  Iterasi {t+1:>7d} " f"T={T:.2f}  " f"cur={cur_val:.1f}  best={best_val:.1f}  " , end="", flush=True)

    print()
    return best_state, best_val, initial_state

# ============================================================
# 6. VISUALISASI (ASCII)
# ============================================================

def draw_solution(state, vehicles, ship, title="SOLUSI"):
    H, W = ship.l, ship.w
    grid = [['.'] * W for _ in range(H)]

    for i, entry in enumerate(state):
        if entry is None:
            continue
        x, y, ori = entry
        lbl = vehicles[i].id[0].upper() if vehicles[i].id else chr(65 + i)
        if lbl in ('.', '#'):
            lbl = chr(65 + i)
        for cx, cy in occupied_cells(x, y, ori, vehicles[i]):
            if 0 <= cx < W and 0 <= cy < H:
                grid[cy][cx] = lbl

    # ---- header ----
    bar = "=" * 60
    print("\n" + bar)
    print(f"  {title}")
    print(bar)
    print(f"  Geladak : {W} (lebar) x {H} (panjang)")
    print(f"  Kapasitas Maksimal : {ship.max_cap}")

    # ---- grid ----
    print("\n  y\\x ", end="")
    for cx in range(W):
        print(f"{cx:2d}", end="")
    print()
    print("  " + "-" * (2 * W + 4))
    for cy in range(H):
        print(f"  {cy:2d} ", end="")
        for cx in range(W):
            print(f" {grid[cy][cx]}", end="")
        print()
    print("  " + "-" * (2 * W + 4))

    # ---- kendaraan di dalam ----
    inside = [(i, e) for i, e in enumerate(state) if e is not None]
    outside = [i for i, e in enumerate(state) if e is None]

    print("\n  --- Kendaraan di dalam geladak ---")
    print(f"  {'Idx':>4s} {'ID':<10s} {'Pos(x,y)':<12s} "
          f"{'Ori':<5s} {'Size(wxl)':<10s} {'Fee':>8s} {'Weight':>8s}")
    print("  " + "-" * 60)
    if not inside:
        print("  (tidak ada)")
    for i, (x, y, ori) in inside:
        v = vehicles[i]
        print(f"  {i:>4d} {v.id:<10s} ({x:>2d},{y:>2d})      "
              f"{ori:<5s} {v.w}x{v.l:<7d} "
              f"{v.fee:>8.1f} {v.weight:>8.1f}")

    # ---- kendaraan di luar ----
    print("\n  --- Kendaraan di luar geladak ---")
    if not outside:
        print("  (semua kendaraan berhasil diangkut)")
    else:
        for i in outside:
            v = vehicles[i]
            print(f"  {i:>4d} {v.id:<10s} "
                  f"size={v.w}x{v.l}  fee={v.fee}  "
                  f"weight={v.weight}  ETA={v.eta}")

    # ---- statistik ----
    total_fee = sum(vehicles[i].fee for i, e in enumerate(state)
                   if e is not None)
    total_wt  = total_weight(state, vehicles)
    used_area = sum(vehicles[i].w * vehicles[i].l
                   for i, e in enumerate(state) if e is not None)
    deck_area = W * H

    print("\n  --- Statistik ---")
    print(f"  Total kendaraan       : {len(vehicles)}")
    print(f"  Di dalam geladak      : {len(inside)}")
    print(f"  Di luar geladak       : {len(outside)}")
    print(f"  Total ShippingFee     : {total_fee:.1f} juta rupiah")
    print(f"  Total berat           : {total_wt:.1f} / {ship.max_cap} unit")
    print(f"  Area terpakai         : {used_area} / {deck_area} "
          f"({used_area/deck_area*100:.1f}%)")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    vehicles = [
        Vehicle("Udin",  w=3, l=4, fee=200, weight=100, eta=3),
        Vehicle("Dino",  w=2, l=3, fee=100, weight=40, eta=5),
        Vehicle("Lia", w=4, l=2, fee=50, weight=20, eta=2),
        Vehicle("Nia",  w=1, l=5, fee=35, weight=10, eta=7),
        Vehicle("Bon",   w=3, l=3, fee=25, weight=5, eta=4),
        Vehicle("Tin", w=2, l=2, fee=40, weight=10, eta=6),
        Vehicle("Mer",  w=5, l=1, fee=30, weight=7, eta=1),
        Vehicle("Ska",  w=2, l=4, fee=60, weight=20, eta=8),
    ]
    ship = Ship(w=10, l=8, max_capacity=500)

    # ---- BUILD INITIAL STATE (acak) ----
    random.seed(42)               # reproducibility; set None jika ingin beda
    initial_state = random_initial_state(vehicles, ship)
    draw_solution(initial_state, vehicles, ship,
                 title="INITIAL STATE (state acak awal)")

    # ---- RUN SIMULATED ANNEALING ----
    best_state, best_val, _ = simulated_annealing(
        vehicles, ship,
        initial_state=initial_state,
        T0=20.0, seed=None
    )

    # ---- VISUALISASI HASIL ----
    print()
    draw_solution(best_state, vehicles, ship,
                 title="FINAL STATE (hasil optimal Simulated Annealing)")

    # ---- FINAL STATE SEBAGAI JSON ----
    state_list = []
    for i, entry in enumerate(best_state):
        base = {"index": i, "vehicle_id": vehicles[i].id}
        if entry is None:
            base["on_ship"] = False
        else:
            x, y, ori = entry
            base.update({"on_ship": True, "x": x, "y": y,
                         "orientation": "Horizontal" if ori == 'H'
                                         else "Vertical"})
        state_list.append(base)

if __name__ == "__main__":
    main()
