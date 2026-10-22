from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

H, V = "H", "V"


# ============================================================
# 1. AKSES DATA (kompatibel dengan main.py & simulated_annealing.py)
# ============================================================

def vehicle_fee(v) -> float:
    return float(v.shipping_fee if hasattr(v, "shipping_fee") else v.fee)


def ship_capacity(ship) -> float:
    return float(ship.max_capacity if hasattr(ship, "max_capacity") else ship.max_cap)


def footprint(v, ori: str) -> tuple[int, int]:
    """(lebar ke arah x, tinggi ke arah y) — sama dengan simulated_annealing.py"""
    return (v.w, v.l) if ori == H else (v.l, v.w)


class Problem:
    """Pembungkus vehicles + ship, sekaligus menghitung bobot penalti."""

    def __init__(self, vehicles, ship):
        self.vehicles = list(vehicles)
        self.ship = ship
        self.n = len(self.vehicles)
        self.W = int(ship.w)          # banyak kolom (sumbu x)
        self.L = int(ship.l)          # banyak baris (sumbu y)
        self.capacity = ship_capacity(ship)
        self.fee = np.array([vehicle_fee(v) for v in self.vehicles], dtype=float)
        self.weight = np.array([float(v.weight) for v in self.vehicles], dtype=float)
        self.area = np.array([v.w * v.l for v in self.vehicles], dtype=float)
        # Penalti dipilih supaya melanggar constraint TIDAK PERNAH lebih untung daripada mematuhinya:
        #   1 sel overlap / keluar geladak  >= fee kendaraan termahal
        #   1 unit kelebihan berat          >= rasio fee/berat terbesar
        self.penalty_cell = float(self.fee.max()) + 1.0
        self.penalty_weight = float((self.fee / np.maximum(self.weight, 1e-9)).max()) + 1.0

    def to_dict(self) -> dict:
        return {
            "ship": {"w": self.W, "l": self.L, "max_capacity": self.capacity},
            "vehicles": [{"id": v.id, "w": v.w, "l": v.l, "shipping_fee": vehicle_fee(v),
                          "weight": float(v.weight), "eta": getattr(v, "eta", None)} for v in self.vehicles],
        }


# ============================================================
# 2. OBJECTIVE FUNCTION
# ============================================================
#   value = total_fee
#           - penalty_cell   * (sel_overlap + sel_di_luar_geladak)
#           - penalty_weight * kelebihan_berat
#   Untuk state yang valid, value == total_fee.

@dataclass
class Evaluation:
    value: float
    total_fee: float
    total_weight: float
    capacity: float
    overlap_cells: int
    out_of_bounds_cells: int
    overweight: float
    n_inside: int
    used_cells: int
    deck_cells: int

    @property
    def is_valid(self) -> bool:
        return self.overlap_cells == 0 and self.out_of_bounds_cells == 0 and self.overweight <= 1e-9

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_valid"] = self.is_valid
        d["utilization"] = round(self.used_cells / self.deck_cells, 4)
        return d


def evaluate(prob: Problem, state: list) -> Evaluation:
    count = np.zeros((prob.L, prob.W), dtype=np.int16)
    fee = weight = 0.0
    oob = n_inside = 0
    for i, e in enumerate(state):
        if e is None:
            continue
        x, y, ori = e
        n_inside += 1
        fee += prob.fee[i]
        weight += prob.weight[i]
        dx, dy = footprint(prob.vehicles[i], ori)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + dx, prob.W), min(y + dy, prob.L)
        inside_area = max(0, x1 - x0) * max(0, y1 - y0)
        oob += dx * dy - inside_area
        if inside_area:
            count[y0:y1, x0:x1] += 1
    overlap = int(np.maximum(count - 1, 0).sum())
    overweight = max(0.0, weight - prob.capacity)
    value = fee - prob.penalty_cell * (overlap + oob) - prob.penalty_weight * overweight
    return Evaluation(float(value), float(fee), float(weight), prob.capacity, overlap, int(oob),
                      float(overweight), n_inside, int(np.count_nonzero(count)), prob.W * prob.L)


def objective(prob: Problem, state: list) -> float:
    return evaluate(prob, state).value


# ============================================================
# 3. INISIALISASI RANDOM
# ============================================================

def random_entry(prob: Problem, i: int, rng: np.random.Generator, inside_prob: float = 0.5):
    """Entry acak untuk kendaraan i: None (di luar) atau (x, y, ori) di dalam batas geladak."""
    if rng.random() >= inside_prob:
        return None
    v = prob.vehicles[i]
    for ori in rng.permutation([H, V]):
        dx, dy = footprint(v, ori)
        if dx <= prob.W and dy <= prob.L:
            return (int(rng.integers(0, prob.W - dx + 1)), int(rng.integers(0, prob.L - dy + 1)), str(ori))
    return None  # tidak muat sama sekali


def random_state(prob: Problem, rng: np.random.Generator, inside_prob: float = 0.5) -> list:
    """State awal random (belum tentu valid)."""
    return [random_entry(prob, i, rng, inside_prob) for i in range(prob.n)]


# ============================================================
# 4. MOVE (dipakai sebagai operator MUTASI)
# ============================================================
#   swap    : tukar entry kendaraan i dan j (bisa dalam-dalam / dalam-luar)
#   move    : pindahkan kendaraan i ke koordinat acak lain (atau keluarkan dari kapal)
#   rotate  : ubah orientasi kendaraan i

def move_swap(prob: Problem, state: list, rng: np.random.Generator) -> list:
    s = state[:]
    if prob.n < 2:
        return s
    i, j = (int(t) for t in rng.choice(prob.n, size=2, replace=False))
    s[i], s[j] = s[j], s[i]
    return s


def move_relocate(prob: Problem, state: list, rng: np.random.Generator, out_prob: float = 0.1) -> list:
    s = state[:]
    i = int(rng.integers(prob.n))
    if s[i] is not None and rng.random() < out_prob:
        s[i] = None
        return s
    ori = s[i][2] if s[i] is not None else str(rng.choice([H, V]))
    dx, dy = footprint(prob.vehicles[i], ori)
    if dx > prob.W or dy > prob.L:
        return s
    s[i] = (int(rng.integers(0, prob.W - dx + 1)), int(rng.integers(0, prob.L - dy + 1)), ori)
    return s


def move_rotate(prob: Problem, state: list, rng: np.random.Generator) -> list:
    s = state[:]
    inside = [k for k, e in enumerate(s) if e is not None]
    if inside:
        i = inside[int(rng.integers(len(inside)))]
        x, y, ori = s[i]
        s[i] = (x, y, V if ori == H else H)
    return s


MOVES = (move_swap, move_relocate, move_rotate)


# ============================================================
# 5. HEURISTIK REPAIR (greedy)
# ============================================================
#   1. Kendaraan di dalam kapal diproses dari fee per sel terbesar.
#   2. Jika menambah kendaraan membuat berat > kapasitas  -> dikeluarkan.
#   3. Diletakkan di posisinya (di-clamp ke dalam geladak, dicoba juga orientasi lain).
#      Jika bentrok -> dipindah ke posisi kosong TERDEKAT (jarak Manhattan); jika tidak ada -> dikeluarkan.
#   4. (fill=True) kendaraan di luar dicoba dimasukkan ke ruang kosong selama kapasitas cukup.
#   Ruang kosong dicari dengan summed-area table sehingga cek semua posisi cukup O(W*L).

def _free_mask(occ: np.ndarray, dx: int, dy: int):
    rows, cols = occ.shape
    if dx > cols or dy > rows:
        return None
    sat = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    sat[1:, 1:] = occ.cumsum(0).cumsum(1)
    return (sat[dy:, dx:] - sat[:-dy, dx:] - sat[dy:, :-dx] + sat[:-dy, :-dx]) == 0


def repair(prob: Problem, state: list, fill: bool = False) -> list:
    s = state[:]
    occ = np.zeros((prob.L, prob.W), dtype=np.int8)
    weight = 0.0
    density = prob.fee / prob.area

    def place(i, x, y, ori):
        nonlocal weight
        dx, dy = footprint(prob.vehicles[i], ori)
        occ[y:y + dy, x:x + dx] = 1
        weight += prob.weight[i]
        s[i] = (int(x), int(y), ori)

    for i in sorted((k for k, e in enumerate(s) if e is not None), key=lambda k: -density[k]):
        x, y, ori = s[i]
        s[i] = None
        if weight + prob.weight[i] > prob.capacity + 1e-9:
            continue
        v = prob.vehicles[i]
        placed = False
        for o in (ori, V if ori == H else H):                      # (a) posisi sekarang
            dx, dy = footprint(v, o)
            if dx > prob.W or dy > prob.L:
                continue
            cx, cy = min(max(x, 0), prob.W - dx), min(max(y, 0), prob.L - dy)
            if not occ[cy:cy + dy, cx:cx + dx].any():
                place(i, cx, cy, o)
                placed = True
                break
        if placed:
            continue
        best = None                                                 # (b) posisi kosong terdekat
        for rank, o in enumerate((ori, V if ori == H else H)):
            mask = _free_mask(occ, *footprint(v, o))
            if mask is None or not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            dist = np.abs(xs - x) + np.abs(ys - y)
            k = int(np.argmin(dist))
            cand = (int(dist[k]), rank, int(xs[k]), int(ys[k]), o)
            best = cand if best is None or cand < best else best
        if best is not None:
            place(i, best[2], best[3], best[4])

    if fill:
        for i in sorted((k for k, e in enumerate(s) if e is None), key=lambda k: -density[k]):
            if weight + prob.weight[i] > prob.capacity + 1e-9:
                continue
            for o in (H, V):
                mask = _free_mask(occ, *footprint(prob.vehicles[i], o))
                if mask is not None and mask.any():
                    ys, xs = np.nonzero(mask)
                    k = int(np.lexsort((ys, xs))[0])
                    place(i, int(xs[k]), int(ys[k]), o)
                    break
    return s


# ============================================================
# 6. GENETIC ALGORITHM
# ============================================================

@dataclass
class GAConfig:
    population_size: int = 50
    max_iterations: int = 100          # banyak generasi
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3         # peluang seorang anak dimutasi
    mutation_steps: int = 2            # maksimum banyak move pada satu mutasi
    tournament_size: int = 3
    elitism: int = 2
    crossover: str = "spatial"         # "spatial" | "uniform"
    use_repair: bool = True
    fill: bool = False
    init_inside_prob: float = 0.5
    seed: int | None = None
    record_snapshots: bool = True
    verbose: bool = False


@dataclass
class GAResult:
    params: dict
    problem: dict
    initial_state: list                # individu terbaik populasi awal
    final_state: list                  # individu terbaik selama pencarian
    initial_eval: dict
    final_eval: dict
    duration_seconds: float
    history: dict                      # iteration, max, mean, min, best_so_far (per generasi)
    snapshots: list = field(default_factory=list)   # state terbaik tiap kali membaik (untuk animasi)
    best_found_at_iteration: int = 0
    algorithm: str = "genetic_algorithm"

    @property
    def final_value(self) -> float:
        return self.final_eval["value"]

    @property
    def initial_value(self) -> float:
        return self.initial_eval["value"]

    def summary(self) -> str:
        fe = self.final_eval
        return "\n".join([
            f"Jumlah populasi  : {self.params['population_size']}",
            f"Banyak iterasi   : {self.params['max_iterations']}",
            f"Seed             : {self.params['seed']}",
            f"Nilai awal       : {self.initial_value:.2f}  (individu terbaik populasi awal)",
            f"Nilai akhir      : {self.final_value:.2f}  (valid={fe['is_valid']})",
            f"Berat            : {fe['total_weight']:g} / {fe['capacity']:g}",
            f"Kendaraan masuk  : {fe['n_inside']} / {len(self.final_state)}",
            f"Utilisasi deck   : {fe['utilization'] * 100:.1f}%",
            f"Best ditemukan   : iterasi ke-{self.best_found_at_iteration}",
            f"Durasi           : {self.duration_seconds:.3f} s",
        ])

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

    @staticmethod
    def load(path) -> "GAResult":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        def fix(st):
            return [tuple(e) if e is not None else None for e in st]

        d["initial_state"], d["final_state"] = fix(d["initial_state"]), fix(d["final_state"])
        for snap in d.get("snapshots", []):
            snap["state"] = fix(snap["state"])
        return GAResult(**d)


class GeneticAlgorithm:
    def __init__(self, vehicles, ship, config: GAConfig | None = None, **overrides):
        self.prob = Problem(vehicles, ship)
        cfg = config or GAConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"Parameter GA tidak dikenal: {k}")
            setattr(cfg, k, v)
        if cfg.population_size < 2:
            raise ValueError("population_size minimal 2")
        if cfg.crossover not in ("spatial", "uniform"):
            raise ValueError("crossover harus 'spatial' atau 'uniform'")
        cfg.elitism = min(cfg.elitism, cfg.population_size)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ---------- individu ----------
    def _individual(self, state: list) -> tuple[list, Evaluation]:
        if self.cfg.use_repair:
            state = repair(self.prob, state, fill=self.cfg.fill)
        return state, evaluate(self.prob, state)

    # ---------- seleksi ----------
    def _tournament(self, pop: list) -> list:
        k = min(self.cfg.tournament_size, len(pop))
        idx = self.rng.choice(len(pop), size=k, replace=False)
        return max((pop[i] for i in idx), key=lambda ind: ind[1].value)[0]

    # ---------- crossover ----------
    def _crossover(self, a: list, b: list) -> tuple[list, list]:
        if self.rng.random() >= self.cfg.crossover_rate:
            return a[:], b[:]
        if self.cfg.crossover == "uniform":
            mask = self.rng.random(self.prob.n) < 0.5
            return ([ea if m else eb for ea, eb, m in zip(a, b, mask)],
                    [eb if m else ea for ea, eb, m in zip(a, b, mask)])
        # spatial: garis potong vertikal di kolom `cut`; anak mewarisi kendaraan parent pertama yang
        # berada di kiri garis dan kendaraan parent kedua yang berada di kanan garis.
        cut = float(self.rng.integers(1, self.prob.W)) if self.prob.W > 1 else 0.5

        def center_x(i, e):
            return e[0] + footprint(self.prob.vehicles[i], e[2])[0] / 2.0

        def child(left, right):
            out = []
            for i, (el, er) in enumerate(zip(left, right)):
                if el is not None and center_x(i, el) < cut:
                    out.append(el)
                elif er is not None and center_x(i, er) >= cut:
                    out.append(er)
                else:
                    out.append(None)
            return out

        return child(a, b), child(b, a)

    # ---------- mutasi ----------
    def _mutate(self, s: list) -> list:
        if self.rng.random() >= self.cfg.mutation_rate:
            return s
        for _ in range(int(self.rng.integers(1, self.cfg.mutation_steps + 1))):
            s = MOVES[int(self.rng.integers(len(MOVES)))](self.prob, s, self.rng)
        return s

    # ---------- main loop ----------
    def run(self) -> GAResult:
        cfg, prob = self.cfg, self.prob
        t0 = time.perf_counter()
        pop = [self._individual(random_state(prob, self.rng, cfg.init_inside_prob))
               for _ in range(cfg.population_size)]
        best = max(pop, key=lambda ind: ind[1].value)
        initial_best = best
        history = {"iteration": [], "max": [], "mean": [], "min": [], "best_so_far": []}
        snapshots = []

        def record(gen):
            nonlocal best
            fits = np.array([ind[1].value for ind in pop])
            gen_best = pop[int(np.argmax(fits))]
            improved = gen == 0 or gen_best[1].value > best[1].value
            if gen_best[1].value > best[1].value:
                best = gen_best
            history["iteration"].append(gen)
            history["max"].append(float(fits.max()))
            history["mean"].append(float(fits.mean()))
            history["min"].append(float(fits.min()))
            history["best_so_far"].append(float(best[1].value))
            if cfg.record_snapshots and improved:
                snapshots.append({"iteration": gen, "value": best[1].value, "state": list(best[0])})
            if cfg.verbose and (gen % max(1, cfg.max_iterations // 10) == 0 or gen == cfg.max_iterations):
                print(f"  gen {gen:4d} | max {fits.max():10.2f} | mean {fits.mean():10.2f} | best {best[1].value:10.2f}")

        record(0)
        for gen in range(1, cfg.max_iterations + 1):
            pop.sort(key=lambda ind: ind[1].value, reverse=True)
            new_pop = pop[:cfg.elitism]
            while len(new_pop) < cfg.population_size:
                for child in self._crossover(self._tournament(pop), self._tournament(pop)):
                    if len(new_pop) < cfg.population_size:
                        new_pop.append(self._individual(self._mutate(child)))
            pop = new_pop
            record(gen)

        params = asdict(cfg)
        params.pop("verbose")
        params.pop("record_snapshots")
        return GAResult(
            params=params, problem=prob.to_dict(),
            initial_state=list(initial_best[0]), final_state=list(best[0]),
            initial_eval=initial_best[1].to_dict(), final_eval=best[1].to_dict(),
            duration_seconds=time.perf_counter() - t0, history=history, snapshots=snapshots,
            best_found_at_iteration=int(np.argmax(history["best_so_far"])),
        )


def genetic_algorithm(vehicles, ship, population_size: int = 50, max_iterations: int = 100,
                      seed: int | None = None, **params) -> GAResult:
    """Shortcut fungsi (gaya sama dengan simulated_annealing(...))."""
    return GeneticAlgorithm(vehicles, ship, population_size=population_size,
                            max_iterations=max_iterations, seed=seed, **params).run()


# ============================================================
# 7. EKSPERIMEN (sesuai spesifikasi)
# ============================================================
#   A. jumlah populasi sebagai kontrol, 3 variasi banyak iterasi, masing-masing 3 run
#   B. banyak iterasi sebagai kontrol, 3 variasi jumlah populasi, masing-masing 3 run
#   Run ke-k pada konfigurasi ke-c memakai seed = base_seed + 100*c + k (tiap run independen).

def run_ga_experiment(vehicles, ship, control_population: int = 50, iteration_variations=(50, 100, 200),
                      control_iterations: int = 100, population_variations=(20, 50, 100),
                      runs: int = 3, base_seed: int = 0, verbose: bool = True, on_result=None,
                      **ga_params) -> dict:
    """
    Return {"A": {label: [GAResult, ...]}, "B": {...}}.
    on_result(group, label, run_index, result) dipanggil setelah tiap run (mis. untuk menyimpan file).
    """
    plans = {
        "A": [(control_population, it) for it in iteration_variations],
        "B": [(p, control_iterations) for p in population_variations],
    }
    out = {}
    for group, configs in plans.items():
        if verbose:
            print("[A] populasi tetap, variasi iterasi" if group == "A" else "[B] iterasi tetap, variasi populasi")
        out[group] = {}
        for c, (pop, iters) in enumerate(configs):
            label = f"pop={pop}, iter={iters}"
            out[group][label] = []
            for k in range(runs):
                seed = base_seed + 100 * c + k
                r = genetic_algorithm(vehicles, ship, pop, iters, seed, **ga_params)
                out[group][label].append(r)
                if on_result:
                    on_result(group, label, k + 1, r)
                if verbose:
                    print(f"  [{label}] run {k + 1}/{runs} seed={seed}: awal={r.initial_value:.2f} "
                          f"akhir={r.final_value:.2f} valid={r.final_eval['is_valid']} "
                          f"durasi={r.duration_seconds:.2f}s")
    return out


def experiment_summary(experiments: dict) -> str:
    """Tabel markdown ringkasan hasil run_ga_experiment."""
    titles = {"A": "A — populasi tetap, variasi banyak iterasi", "B": "B — iterasi tetap, variasi jumlah populasi"}
    lines = ["# Ringkasan Eksperimen Genetic Algorithm", ""]
    for group, configs in experiments.items():
        lines += [f"## Eksperimen {titles.get(group, group)}", "",
                  "| Konfigurasi | Run | Seed | Nilai awal | Nilai akhir | Valid | Best di iterasi | Durasi (s) |",
                  "|---|---|---|---|---|---|---|---|"]
        for label, results in configs.items():
            for k, r in enumerate(results, start=1):
                lines.append(f"| {label} | {k} | {r.params['seed']} | {r.initial_value:.2f} | {r.final_value:.2f} | "
                             f"{r.final_eval['is_valid']} | {r.best_found_at_iteration} | {r.duration_seconds:.2f} |")
        lines += ["", "| Konfigurasi | Rata-rata | Std | Max | Min | Rata-rata durasi (s) |", "|---|---|---|---|---|---|"]
        for label, results in configs.items():
            v = np.array([r.final_value for r in results])
            d = np.array([r.duration_seconds for r in results])
            lines.append(f"| {label} | {v.mean():.2f} | {v.std():.2f} | {v.max():.2f} | {v.min():.2f} | {d.mean():.2f} |")
        lines.append("")
    return "\n".join(lines)
