import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

H, V = "H", "V"


# ============================================================
# 1. DATA PERMASALAHAN
# ============================================================

class Problem:

    def __init__(self, vehicles, ship):
        get = lambda obj, *names: next(getattr(obj, n) for n in names if hasattr(obj, n))  # noqa: E731
        self.vehicles = list(vehicles)
        self.n = len(self.vehicles)
        self.W = int(get(ship, "w", "width")) 
        self.L = int(get(ship, "l", "length")) 
        self.capacity = float(get(ship, "max_capacity", "max_cap"))
        self.vw = [int(get(v, "w", "width")) for v in self.vehicles]
        self.vl = [int(get(v, "l", "length")) for v in self.vehicles]
        self.fee = np.array([float(get(v, "shipping_fee", "fee")) for v in self.vehicles])
        self.weight = np.array([float(v.weight) for v in self.vehicles])
        self.max_fitness = self.fee.sum() 

    def size(self, i, ori):
        """(lebar ke arah x, tinggi ke arah y) kendaraan i."""
        return (self.vw[i], self.vl[i]) if ori == H else (self.vl[i], self.vw[i])


# ============================================================
# 2. FITNESS FUNCTION
# ============================================================

def fitness(prob, state):
    grid = np.zeros((prob.L, prob.W), dtype=int)
    fee = weight = 0.0
    for i, gene in enumerate(state):
        if gene is None:
            continue
        x, y, ori = gene
        dx, dy = prob.size(i, ori)
        if x < 0 or y < 0 or x + dx > prob.W or y + dy > prob.L:
            return 0.0  
        grid[y:y + dy, x:x + dx] += 1
        fee += prob.fee[i]
        weight += prob.weight[i]
    if grid.max(initial=0) > 1 or weight > prob.capacity:
        return 0.0 
    return fee


# ============================================================
# 3. OPERATOR GA
# ============================================================

def random_state(prob, rng, tries=500):
    state = [None] * prob.n
    occupied = np.zeros((prob.L, prob.W), dtype=bool)
    weight = 0.0
    for i in rng.permutation(prob.n):
        i = int(i)
        if weight + prob.weight[i] > prob.capacity:
            continue
        for _ in range(tries):
            ori = H if rng.random() < 0.5 else V
            dx, dy = prob.size(i, ori)
            if dx > prob.W or dy > prob.L:
                continue
            x, y = int(rng.integers(0, prob.W - dx + 1)), int(rng.integers(0, prob.L - dy + 1))
            if not occupied[y:y + dy, x:x + dx].any():
                occupied[y:y + dy, x:x + dx] = True
                weight += prob.weight[i]
                state[i] = (x, y, ori)
                break
    return state


def random_selection(population, fit, rng):
    if fit.sum() <= 0:
        return population[int(rng.integers(len(population)))]
    return population[int(rng.choice(len(population), p=fit / fit.sum()))]


def reproduce(x, y, rng):
    c = int(rng.integers(1, len(x))) if len(x) > 1 else len(x)
    return x[:c] + y[c:]


def mutate(prob, child, rng, out_prob=0.2):
    child = child[:]
    i = int(rng.integers(prob.n))
    gene = None
    if rng.random() >= out_prob:
        for ori in ((H, V) if rng.random() < 0.5 else (V, H)):
            dx, dy = prob.size(i, ori)
            if dx <= prob.W and dy <= prob.L:
                gene = (int(rng.integers(0, prob.W - dx + 1)), int(rng.integers(0, prob.L - dy + 1)), ori)
                break
    child[i] = gene
    return child


def repair(prob, state):
    state = state[:]
    occupied = np.zeros((prob.L, prob.W), dtype=bool)
    weight = 0.0
    density = prob.fee / np.array([w * l for w, l in zip(prob.vw, prob.vl)])
    for i in sorted((k for k, g in enumerate(state) if g is not None), key=lambda k: -density[k]):
        x, y, ori = state[i]
        state[i] = None
        if weight + prob.weight[i] > prob.capacity:
            continue
        candidates = []
        for rank, o in enumerate((ori, V if ori == H else H)):
            dx, dy = prob.size(i, o)
            if dx > prob.W or dy > prob.L:
                continue
            sat = np.pad(occupied.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            free = (sat[dy:, dx:] - sat[:-dy, dx:] - sat[dy:, :-dx] + sat[:-dy, :-dx]) == 0
            candidates += [(abs(cx - x) + abs(cy - y), rank, cy, cx, o) for cy, cx in zip(*np.nonzero(free))]
        if candidates:
            _, _, cy, cx, o = min(candidates)
            cx, cy = int(cx), int(cy)
            dx, dy = prob.size(i, o)
            occupied[cy:cy + dy, cx:cx + dx] = True
            weight += prob.weight[i]
            state[i] = (cx, cy, o)
    return state


# ============================================================
# 4. GENETIC ALGORITHM
# ============================================================

@dataclass
class GAResult:
    params: dict
    initial_state: list 
    final_state: list  
    initial_value: float
    final_value: float
    best_ever_value: float   
    iterations_done: int
    duration_seconds: float
    history: dict = field(default_factory=dict)  

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")


def genetic_algorithm(vehicles, ship, population_size=50, max_iterations=100, seed=None,
                      mutation_rate=0.1, mutation_out_prob=0.2, elitism=0, use_repair=False):
    prob, rng = Problem(vehicles, ship), np.random.default_rng(seed)
    t0 = time.perf_counter()

    population = [random_state(prob, rng) for _ in range(population_size)]
    fit = np.array([fitness(prob, s) for s in population])
    initial_state, initial_value = population[int(fit.argmax())], float(fit.max())
    history = {"iteration": [0], "max": [float(fit.max())], "mean": [float(fit.mean())]}

    gen = 0
    while gen < max_iterations and fit.max() < prob.max_fitness: 
        gen += 1
        new_population = [population[i] for i in np.argsort(-fit)[:elitism]]
        while len(new_population) < population_size:
            x = random_selection(population, fit, rng)
            y = random_selection(population, fit, rng)
            child = reproduce(x, y, rng)
            if rng.random() < mutation_rate:
                child = mutate(prob, child, rng, mutation_out_prob)
            if use_repair:
                child = repair(prob, child)
            new_population.append(child)
        population = new_population
        fit = np.array([fitness(prob, s) for s in population])
        history["iteration"].append(gen)
        history["max"].append(float(fit.max()))
        history["mean"].append(float(fit.mean()))

    params = dict(population_size=population_size, max_iterations=max_iterations, seed=seed,
                  mutation_rate=mutation_rate, elitism=elitism, use_repair=use_repair)
    return GAResult(params, list(initial_state), list(population[int(fit.argmax())]), initial_value,
                    float(fit.max()), max(history["max"]), gen, time.perf_counter() - t0, history)


# ============================================================
# 5. EKSPERIMEN (sesuai spesifikasi)
# ============================================================
def run_ga_experiment(vehicles, ship, control_population=50, iteration_variations=(50, 100, 200),
                      control_iterations=100, population_variations=(20, 50, 100),
                      runs=3, base_seed=0, on_result=None, **ga_params):
    """Return {"A": {label: [GAResult, ...]}, "B": {...}}. on_result(group, label, k, r) dipanggil tiap run."""
    plans = {"A": [(control_population, it) for it in iteration_variations],
             "B": [(p, control_iterations) for p in population_variations]}
    out = {}
    for group, configs in plans.items():
        print("[A] populasi tetap, variasi iterasi" if group == "A" else "[B] iterasi tetap, variasi populasi")
        out[group] = {}
        for c, (pop, iters) in enumerate(configs):
            label = f"pop={pop}, iter={iters}"
            out[group][label] = []
            for k in range(1, runs + 1):
                seed = base_seed + 100 * c + k - 1
                r = genetic_algorithm(vehicles, ship, pop, iters, seed, **ga_params)
                out[group][label].append(r)
                if on_result:
                    on_result(group, label, k, r)
                print(f"  [{label}] run {k}/{runs} seed={seed}: awal={r.initial_value:g} "
                      f"akhir={r.final_value:g} durasi={r.duration_seconds:.2f}s")
    return out


def experiment_summary(experiments):
    titles = {"A": "A — populasi tetap, variasi banyak iterasi", "B": "B — iterasi tetap, variasi jumlah populasi"}
    lines = ["# Ringkasan Eksperimen Genetic Algorithm", ""]
    for group, configs in experiments.items():
        lines += [f"## Eksperimen {titles[group]}", "",
                  "| Konfigurasi | Run | Seed | Nilai awal | Nilai akhir | Terbaik selama run | Iterasi | Durasi (s) |",
                  "|---|---|---|---|---|---|---|---|"]
        for label, results in configs.items():
            for k, r in enumerate(results, start=1):
                lines.append(f"| {label} | {k} | {r.params['seed']} | {r.initial_value:g} | {r.final_value:g} | "
                             f"{r.best_ever_value:g} | {r.iterations_done} | {r.duration_seconds:.2f} |")
        lines += ["", "| Konfigurasi | Rata-rata | Std | Max | Min | Rata-rata durasi (s) |", "|---|---|---|---|---|---|"]
        for label, results in configs.items():
            v = np.array([r.final_value for r in results])
            d = np.array([r.duration_seconds for r in results])
            lines.append(f"| {label} | {v.mean():.2f} | {v.std():.2f} | {v.max():g} | {v.min():g} | {d.mean():.2f} |")
        lines.append("")
    return "\n".join(lines)
