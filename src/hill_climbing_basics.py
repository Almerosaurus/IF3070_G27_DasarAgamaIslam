#!/usr/bin/env python3
import random


# ============================================================
# 1. MODEL DATA
# ============================================================

class Vehicle:
    def __init__(self, vid, w, l, fee, weight, eta=0):
        self.id     = vid
        self.w      = w      # lebar (grid)
        self.l      = l      # panjang (grid)
        self.fee    = fee    # tarif pengiriman (juta rupiah)
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
# 2. FUNGSI UMUM
# ============================================================

def occupied_cells(x, y, ori, veh):
    """Himpunan koordinat (kolom, baris) yang ditempati kendaraan di (x,y)."""
    cw, ch = (veh.w, veh.l) if ori == 'H' else (veh.l, veh.w)
    return {(x + dx, y + dy) for dx in range(cw) for dy in range(ch)}


def total_weight(state, vehicles):
    return sum(vehicles[i].weight for i, entry in enumerate(state)
               if entry is not None)


def is_placement_valid(state, veh_idx, vehicles, ship):
    entry = state[veh_idx]
    if entry is None:
        return total_weight(state, vehicles) <= ship.max_cap

    x, y, ori = entry
    if ori not in ('H', 'V'):
        return False
    veh = vehicles[veh_idx]
    cw, ch = (veh.w, veh.l) if ori == 'H' else (veh.l, veh.w)

    if x < 0 or y < 0 or x + cw > ship.w or y + ch > ship.l:
        return False

    my_cells = occupied_cells(x, y, ori, veh)
    for j, other_entry in enumerate(state):
        if j == veh_idx or other_entry is None:
            continue
        ox, oy, other_ori = other_entry
        if my_cells & occupied_cells(ox, oy, other_ori, vehicles[j]):
            return False

    return total_weight(state, vehicles) <= ship.max_cap


def state_is_fully_valid(state, vehicles, ship):
    return (len(state) == len(vehicles) and
            all(is_placement_valid(state, i, vehicles, ship)
                for i in range(len(vehicles))))


def compute_fitness(state, vehicles, ship):
    if not state_is_fully_valid(state, vehicles, ship):
        return 0
    return sum(vehicles[i].fee for i, entry in enumerate(state)
               if entry is not None)


def possible_placements(veh, ship):
    placements = [None]
    for ori in ('H', 'V'):
        cw, ch = (veh.w, veh.l) if ori == 'H' else (veh.l, veh.w)
        if cw > ship.w or ch > ship.l:
            continue
        placements.extend((x, y, ori)
                          for x in range(ship.w - cw + 1)
                          for y in range(ship.l - ch + 1))
    return placements


def random_initial_state(vehicles, ship, tries=500, rng=None):
    rng = rng or random.Random()
    state = [None] * len(vehicles)
    order = list(range(len(vehicles)))
    rng.shuffle(order)

    for i in order:
        veh = vehicles[i]
        for _ in range(tries):
            ori = rng.choice(['H', 'V'])
            cw, ch = (veh.w, veh.l) if ori == 'H' else (veh.l, veh.w)
            if cw > ship.w or ch > ship.l:
                continue
            x = rng.randint(0, ship.w - cw)
            y = rng.randint(0, ship.l - ch)
            cand = state[:]
            cand[i] = (x, y, ori)
            if state_is_fully_valid(cand, vehicles, ship):
                state[i] = (x, y, ori)
                break
    return state


def generate_successors(state, vehicles, ship):
    """Bangkitkan semua perpindahan satu kendaraan dan pertukaran posisi yang valid."""
    successors = []
    seen = set()

    # Memindahkan, memuat, mengeluarkan, atau mengubah orientasi satu kendaraan.
    for i, veh in enumerate(vehicles):
        for placement in possible_placements(veh, ship):
            if placement == state[i]:
                continue
            cand = state[:]
            cand[i] = placement
            key = tuple(cand)
            if key not in seen and state_is_fully_valid(cand, vehicles, ship):
                seen.add(key)
                successors.append(cand)

    # Menukar penempatan dua kendaraan.
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            if state[i] == state[j]:
                continue
            cand = state[:]
            cand[i], cand[j] = cand[j], cand[i]
            key = tuple(cand)
            if key not in seen and state_is_fully_valid(cand, vehicles, ship):
                seen.add(key)
                successors.append(cand)

    return successors