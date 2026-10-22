#!/usr/bin/env python3
import random
import math
import itertools

# 1. Model Data

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


# 2. Fungsi Bantuan

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
    if total_weight(state, vehicles) > ship.max_cap:
        return False
    occupied = set()
    for veh, entry in zip(vehicles, state):
        if entry is None:
            continue
        x, y, ori = entry
        cw, ch = (veh.w, veh.l) if ori == 'H' else (veh.l, veh.w)
        if x < 0 or y < 0 or x + cw > ship.w or y + ch > ship.l:
            return False
        for dx in range(cw):
            for dy in range(ch):
                cell = (x + dx, y + dy)
                if cell in occupied:
                    return False
                occupied.add(cell)
    return True


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


# 3. Random Initial State

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


# 4. Neighbor Generation (3 Aksi)

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


# 5. Simulated Annealing

def simulated_annealing(vehicles, ship, initial_state=None, T0=50.0, seed=None, record=None):
    if seed is not None:
        random.seed(seed)

    if initial_state is None:
        initial_state = random_initial_state(vehicles, ship)

    state   = initial_state[:]
    cur_val = compute_fitness(state, vehicles, ship)
    best_state = state[:]
    best_val   = cur_val

    move_funcs = [move_swap, move_relocate, move_rotate]
    if record is not None:
        record.update(iteration=[0], value=[cur_val], best_so_far=[best_val],
                      acceptance_probability=[1.0], temperature=[T0])

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

        if record is not None:
            record["iteration"].append(t)
            record["value"].append(cur_val)
            record["best_so_far"].append(best_val)
            record["acceptance_probability"].append(1.0 if delta >= 0 else math.exp(delta / T))
            record["temperature"].append(T)

        if (t + 1) % 10 == 0:
            print(f"\r  Iterasi {t+1:>7d} " f"T={T:.2f}  " f"cur={cur_val:.1f}  best={best_val:.1f}  " , end="", flush=True)

    print()
    if record is not None:
        record["final_current_state"] = state[:]
    return best_state, best_val, initial_state
