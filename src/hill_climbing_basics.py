import random

# Model Data

class Vehicle:
    def __init__(self, vehicle_id, width, length, shipping_fee, weight, eta=0):
        self.id           = vehicle_id
        self.width        = width
        self.length       = length
        self.shipping_fee = shipping_fee
        self.weight = weight
        self.eta    = eta

    def __repr__(self):
        return (f"Vehicle(id={self.id!r}, {self.width}x{self.length}, "
                f"shipping_fee={self.shipping_fee})")


class Ship:
    def __init__(self, width, length, max_capacity):
        self.width        = width
        self.length       = length
        self.max_capacity = max_capacity

    def __repr__(self):
        return f"Ship({self.width}x{self.length}, cap={self.max_capacity})"


# Fungsi Bantuan

def occupied_cells(x, y, orientation, vehicle):
    """Himpunan koordinat (kolom, baris) yang ditempati kendaraan di (x,y)."""
    cell_width, cell_length = (
        (vehicle.width, vehicle.length)
        if orientation == 'H'
        else (vehicle.length, vehicle.width)
    )
    return {
        (x + dx, y + dy)
        for dx in range(cell_width)
        for dy in range(cell_length)
    }


def total_weight(state, vehicles):
    return sum(
        vehicle.weight
        for vehicle, placement in zip(vehicles, state)
        if placement is not None
    )


def is_placement_valid(state, vehicle_index, vehicles, ship):
    placement = state[vehicle_index]
    if placement is None:
        return total_weight(state, vehicles) <= ship.max_capacity

    x, y, orientation = placement
    if orientation not in ('H', 'V'):
        return False
    vehicle = vehicles[vehicle_index]
    cell_width, cell_length = (
        (vehicle.width, vehicle.length)
        if orientation == 'H'
        else (vehicle.length, vehicle.width)
    )

    if (x < 0 or y < 0 or x + cell_width > ship.width
            or y + cell_length > ship.length):
        return False

    vehicle_cells = occupied_cells(x, y, orientation, vehicle)
    for other_index, other_placement in enumerate(state):
        if other_index == vehicle_index or other_placement is None:
            continue
        other_x, other_y, other_orientation = other_placement
        other_vehicle_cells = occupied_cells(
            other_x, other_y, other_orientation, vehicles[other_index]
        )
        if vehicle_cells & other_vehicle_cells:
            return False

    return total_weight(state, vehicles) <= ship.max_capacity


def _state_is_fully_valid_original(state, vehicles, ship):
    return (len(state) == len(vehicles) and
            all(is_placement_valid(state, i, vehicles, ship)
                for i in range(len(vehicles))))


def _cell_size(vehicle, orientation):
    return ((vehicle.width, vehicle.length) if orientation == 'H'
            else (vehicle.length, vehicle.width))


def state_is_fully_valid(state, vehicles, ship):
    if len(state) != len(vehicles):
        return False
    if total_weight(state, vehicles) > ship.max_capacity:
        return False
    occupied = set()
    for vehicle, placement in zip(vehicles, state):
        if placement is None:
            continue
        x, y, orientation = placement
        if orientation not in ('H', 'V'):
            return False
        cell_width, cell_length = _cell_size(vehicle, orientation)
        if (x < 0 or y < 0 or x + cell_width > ship.width
                or y + cell_length > ship.length):
            return False
        for dx in range(cell_width):
            for dy in range(cell_length):
                cell = (x + dx, y + dy)
                if cell in occupied:
                    return False
                occupied.add(cell)
    return True

_FITNESS_CACHE = {"owner": None, "values": {}}


def max_possible_fitness(vehicles):
    return sum(vehicle.shipping_fee for vehicle in vehicles)


def compute_fitness(state, vehicles, ship):
    if _FITNESS_CACHE["owner"] == (id(vehicles), id(ship)):
        cached = _FITNESS_CACHE["values"].get(tuple(state))
        if cached is not None:
            return cached
    if not state_is_fully_valid(state, vehicles, ship):
        return 0
    return sum(
        vehicle.shipping_fee
        for vehicle, placement in zip(vehicles, state)
        if placement is not None
    )


def possible_placements(vehicle, ship):
    placements = [None]
    for orientation in ('H', 'V'):
        cell_width, cell_length = (
            (vehicle.width, vehicle.length)
            if orientation == 'H'
            else (vehicle.length, vehicle.width)
        )
        if cell_width > ship.width or cell_length > ship.length:
            continue
        placements.extend(
            (x, y, orientation)
            for x in range(ship.width - cell_width + 1)
            for y in range(ship.length - cell_length + 1)
        )
    return placements


def random_initial_state(vehicles, ship, tries=500, rng=None):
    rng = rng or random.Random()
    state = [None] * len(vehicles)
    order = list(range(len(vehicles)))
    rng.shuffle(order)

    for i in order:
        vehicle = vehicles[i]
        for _ in range(tries):
            orientation = rng.choice(['H', 'V'])
            cell_width, cell_length = (
                (vehicle.width, vehicle.length)
                if orientation == 'H'
                else (vehicle.length, vehicle.width)
            )
            if cell_width > ship.width or cell_length > ship.length:
                continue
            x = rng.randint(0, ship.width - cell_width)
            y = rng.randint(0, ship.length - cell_length)
            cand = state[:]
            cand[i] = (x, y, orientation)
            if state_is_fully_valid(cand, vehicles, ship):
                state[i] = (x, y, orientation)
                break
    return state


def generate_successors(state, vehicles, ship):
    if not state_is_fully_valid(state, vehicles, ship):
        return _generate_successors_original(state, vehicles, ship)

    owner = {}
    for index, (vehicle, placement) in enumerate(zip(vehicles, state)):
        if placement is not None:
            for cell in occupied_cells(*placement, vehicle):
                owner[cell] = index
    current_weight = total_weight(state, vehicles)
    current_fee = sum(v.shipping_fee for v, p in zip(vehicles, state) if p is not None)

    _FITNESS_CACHE["owner"] = (id(vehicles), id(ship))
    cache = _FITNESS_CACHE["values"] = {}
    successors = []
    seen = set()

    for i, vehicle in enumerate(vehicles):
        inside = state[i] is not None
        base_weight = current_weight - (vehicle.weight if inside else 0)
        base_fee = current_fee - (vehicle.shipping_fee if inside else 0)
        for placement in possible_placements(vehicle, ship):
            if placement == state[i]:
                continue
            if placement is not None:
                if base_weight + vehicle.weight > ship.max_capacity:
                    continue
                x, y, orientation = placement
                cell_width, cell_length = _cell_size(vehicle, orientation)
                if any(owner.get((x + dx, y + dy), i) != i
                       for dx in range(cell_width) for dy in range(cell_length)):
                    continue
            cand = state[:]
            cand[i] = placement
            key = tuple(cand)
            if key not in seen:
                seen.add(key)
                successors.append(cand)
                cache[key] = base_fee + (vehicle.shipping_fee if placement is not None else 0)

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
                cache[key] = sum(v.shipping_fee for v, p in zip(vehicles, cand) if p is not None)

    return successors


def _generate_successors_original(state, vehicles, ship):
    successors = []
    seen = set()

    for i, vehicle in enumerate(vehicles):
        for placement in possible_placements(vehicle, ship):
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


def random_successor(state, vehicles, ship, rng=None):
    rng = rng or random
    actions = []

    for vehicle_index, vehicle in enumerate(vehicles):
        for placement in possible_placements(vehicle, ship):
            if placement != state[vehicle_index]:
                actions.append(("place", vehicle_index, placement))

    for first_index in range(len(vehicles)):
        for second_index in range(first_index + 1, len(vehicles)):
            if state[first_index] != state[second_index]:
                actions.append(("swap", first_index, second_index))

    rng.shuffle(actions)
    for action in actions:
        candidate = state[:]
        if action[0] == "place":
            _, vehicle_index, placement = action
            candidate[vehicle_index] = placement
        else:
            _, first_index, second_index = action
            candidate[first_index], candidate[second_index] = (
                candidate[second_index], candidate[first_index]
            )

        if state_is_fully_valid(candidate, vehicles, ship):
            return candidate

    return None
