import argparse
import contextlib
import io
import itertools
import math
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # gambar langsung disimpan ke file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import hill_climbing_basics as hcb  # noqa: E402
import simulated_annealing as sam  # noqa: E402
import visualization as viz  # noqa: E402
from genetic_algorithm import experiment_summary, run_ga_experiment  # noqa: E402
from hill_climbing_random_restart import random_restart  # noqa: E402
from hill_climbing_sideways import sideways_move  # noqa: E402
from hill_climbing_steepest import steepest_ascent  # noqa: E402
from hill_climbing_stochastic import stochastic_hill_climbing  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = ROOT / "data" / "dataset.json"
OUTPUT_DIR = ROOT / "output"

ALGORITHM_MENU = [("hc", "Hill-climbing"), ("sa", "Simulated Annealing"), ("ga", "Genetic Algorithm")]
HC_MENU = [("steepest", "Steepest Ascent Hill-climbing (hill-climbing dasar)"),
           ("sideways", "Hill-climbing with Sideways Move"),
           ("random_restart", "Random Restart Hill-climbing"),
           ("stochastic", "Stochastic Hill-climbing")]
NAMES = {"steepest": "Steepest Ascent Hill-climbing", "sideways": "Hill-climbing with Sideways Move",
         "random_restart": "Random Restart Hill-climbing", "stochastic": "Stochastic Hill-climbing",
         "sa": "Simulated Annealing", "ga": "Genetic Algorithm"}
DEFAULT_PARAMS = {
    "steepest": {},
    "sideways": {"max_sideways": 100},
    "random_restart": {"max_restart": 10},
    "stochastic": {"nmax": 1000},
    "sa": {"T0": 20.0, "stuck_window": 100},
    "ga": {"control_population": 50, "iteration_variations": [50, 100, 200],
           "control_iterations": 100, "population_variations": [20, 50, 100],
           "mutation_rate": 0.1, "elitism": 0, "use_repair": False},
}


# Model Data

@dataclass
class Vehicle:
    id: str
    w: int
    l: int
    shipping_fee: float
    weight: float
    eta: int


@dataclass
class Ship:
    w: int
    l: int
    max_capacity: float


def load_problem(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    s = data["ship"]
    ship = Ship(s["dimensions"]["w"], s["dimensions"]["l"], s["max_capacity"])
    vehicles = [Vehicle(v["id"], v["dimensions"]["w"], v["dimensions"]["l"], v["shipping_fee"],
                        v["weight"], v.get("eta", 0)) for v in data["vehicles"]]
    return vehicles, ship


# Setiap modul punya class Vehicle/Ship sendiri, jadi data dikonversi ke class milik modulnya.
def to_hc_model(vehicles, ship):
    return ([hcb.Vehicle(v.id, v.w, v.l, v.shipping_fee, v.weight, v.eta) for v in vehicles],
            hcb.Ship(ship.w, ship.l, ship.max_capacity))


def to_sa_model(vehicles, ship):
    return ([sam.Vehicle(v.id, v.w, v.l, v.shipping_fee, v.weight, v.eta) for v in vehicles],
            sam.Ship(ship.w, ship.l, ship.max_capacity))


def state_value(vehicles, ship, state):
    st = viz.state_stats(vehicles, ship, state)
    return st["fee"] if st["valid"] else 0.0


def ascii_state(vehicles, ship, state, title):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        viz.draw_solution(list(state), vehicles, ship, title)
    return buf.getvalue()


# Hill-climbing & Simulated Annealing

@dataclass
class RunResult:
    algorithm: str
    params: dict
    seed: int
    initial_state: list
    final_state: list
    initial_value: float
    final_value: float
    duration_seconds: float
    history: dict
    extra: dict = field(default_factory=dict)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")


def run_hill_climbing(variant, vehicles, ship, params, seed):
    hv, hs = to_hc_model(vehicles, ship)
    if variant == "random_restart":
        init, final, value, hist, restarts, per_restart, dur = random_restart(
            hv, hs, max_restart=params["max_restart"], seed=seed)
        extra = {"restarts": restarts, "iterations_per_restart": per_restart, "max_restart": params["max_restart"]}
    else:
        func, kwargs = {"steepest": (steepest_ascent, {}),
                        "sideways": (sideways_move, {"max_sideways": params.get("max_sideways")}),
                        "stochastic": (stochastic_hill_climbing, {"nmax": params.get("nmax")})}[variant]
        init, final, value, hist, iters, dur = func(hv, hs, seed=seed, **kwargs)
        extra = {"iterations": iters, **kwargs}
    history = {"iteration": list(range(len(hist))), "value": list(hist),
               "best_so_far": list(np.maximum.accumulate(hist))}
    return RunResult(variant, params, seed, list(init), list(final), state_value(vehicles, ship, init),
                     float(value), dur, history, extra)


def count_stuck(values, window):
    stuck = same = 0
    for prev, cur in zip(values, values[1:]):
        same = same + 1 if cur == prev else 0
        stuck += same == window
    return stuck


def simulated_annealing_with_record(sv, ss, T0, seed):
    random.seed(seed)
    initial_state = sam.random_initial_state(sv, ss)
    state, cur_val = initial_state[:], sam.compute_fitness(initial_state, sv, ss)
    best_state, best_val = state[:], cur_val
    move_funcs = [sam.move_swap, sam.move_relocate, sam.move_rotate]
    record = {"iteration": [0], "value": [cur_val], "best_so_far": [best_val],
              "acceptance_probability": [1.0], "temperature": [T0]}
    for t in itertools.count(1):
        T = sam.schedule(t, T0)
        if T == 0:
            break
        new_state = random.choice(move_funcs)(state, sv, ss)
        new_val = sam.compute_fitness(new_state, sv, ss)
        delta = new_val - cur_val
        if delta >= 0 or random.random() < math.exp(delta / T):
            state, cur_val = new_state, new_val
        if cur_val > best_val:
            best_state, best_val = state[:], cur_val
        for key, val in (("iteration", t), ("value", cur_val), ("best_so_far", best_val),
                         ("acceptance_probability", 1.0 if delta >= 0 else math.exp(delta / T)), ("temperature", T)):
            record[key].append(val)
    return initial_state, state, best_state, best_val, record


def run_simulated_annealing(vehicles, ship, params, seed):
    sv, ss = to_sa_model(vehicles, ship)
    t0 = time.perf_counter()
    init, final, _, best_value, record = simulated_annealing_with_record(sv, ss, params["T0"], seed)
    dur = time.perf_counter() - t0
    # state akhir = current saat T = 0 (return value di PPT)
    extra = {"iterations": record["iteration"][-1], "stuck_count": count_stuck(record["value"], params["stuck_window"]),
             "stuck_window": params["stuck_window"], "best_value": float(best_value)}
    return RunResult("sa", params, seed, list(init), list(final), state_value(vehicles, ship, init),
                     state_value(vehicles, ship, final), dur, record, extra)


def describe(r):
    e = r.extra
    if r.algorithm == "random_restart":
        return f"banyak restart = {e['restarts']} (maks {e['max_restart']}), iterasi per restart = {e['iterations_per_restart']}"
    if r.algorithm == "sa":
        return (f"iterasi = {e['iterations']}, stuck = {e['stuck_count']} kali (>= {e['stuck_window']} iterasi "
                f"tanpa perubahan), nilai terbaik selama run = {e['best_value']:g}")
    if r.algorithm == "sideways":
        return f"iterasi = {e['iterations']} (maks sideways {e['max_sideways']})"
    return f"iterasi = {e['iterations']}"


def save_run(r, vehicles, ship, out_dir, k):
    name = NAMES[r.algorithm]
    r.save(out_dir / f"run{k}.json")
    plt.close(viz.plot_initial_final(vehicles, ship, r.initial_state, r.final_state,
                                     f"{name} — run {k} (seed {r.seed})  |  durasi {r.duration_seconds:.2f} s",
                                     out_dir / f"run{k}_states.png", r.initial_value, r.final_value))
    fig = viz.plot_objective(r.history, f"{name} — objective function vs iterasi (run {k})",
                             info=f"{describe(r)}  |  durasi {r.duration_seconds:.2f} s")
    if r.algorithm == "random_restart":                # garis batas tiap restart
        for pos in np.cumsum([n + 1 for n in r.extra["iterations_per_restart"][:-1]]):
            fig.axes[0].axvline(pos - 0.5, color=viz.MUTED, linewidth=0.8, linestyle=":")
    viz.save(fig, out_dir / f"run{k}_objective.png")
    plt.close(fig)
    (out_dir / f"run{k}_states.txt").write_text(
        ascii_state(vehicles, ship, r.initial_state, f"STATE AWAL — {name} run {k}") +
        ascii_state(vehicles, ship, r.final_state, f"STATE AKHIR — {name} run {k}"), encoding="utf-8")


def reset_output_dir(path):
    if path.exists():
        for p in sorted(path.rglob("*"), key=lambda q: len(q.parts), reverse=True):
            try:
                p.unlink() if p.is_file() else p.rmdir()
            except OSError:
                pass
    path.mkdir(parents=True, exist_ok=True)


def run_local_search(algo, vehicles, ship, params, runs, base_seed):
    out_dir = OUTPUT_DIR / ("sa" if algo == "sa" else f"hc_{algo}")
    reset_output_dir(out_dir)                        
    results = []
    for k in range(1, runs + 1):
        seed = base_seed + k - 1
        print(f"\n>>> {NAMES[algo]} — run {k}/{runs} (seed {seed})")
        r = (run_simulated_annealing(vehicles, ship, params, seed) if algo == "sa"
             else run_hill_climbing(algo, vehicles, ship, params, seed))
        save_run(r, vehicles, ship, out_dir, k)
        print(f"    nilai awal = {r.initial_value:g}  ->  nilai akhir = {r.final_value:g}  |  "
              f"{describe(r)}  |  durasi = {r.duration_seconds:.2f} s")
        results.append(r)

    plt.close(viz.plot_experiment({f"run {k}": [r] for k, r in enumerate(results, 1)}, f"{NAMES[algo]} — {runs} run",
                                  out_dir / "experiment.png", series_key="value", second_key="best_so_far"))
    v = np.array([r.final_value for r in results])
    lines = [f"# {NAMES[algo]}", "", f"Parameter: {params}", "",
             "| Run | Seed | Nilai awal | Nilai akhir | Durasi (s) | Keterangan |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {r.seed} | {r.initial_value:g} | {r.final_value:g} | {r.duration_seconds:.2f} | {describe(r)} |"
              for k, r in enumerate(results, 1)]
    lines += ["", f"Rata-rata nilai akhir = {v.mean():.2f}, std = {v.std():.2f}, max = {v.max():g}, min = {v.min():g}",
              f"Rata-rata durasi = {np.mean([r.duration_seconds for r in results]):.2f} s"]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    best = max(results, key=lambda r: r.final_value)
    k = results.index(best) + 1
    print(ascii_state(vehicles, ship, best.initial_state, f"STATE AWAL — run terbaik (run {k})"))
    print(ascii_state(vehicles, ship, best.final_state, f"STATE AKHIR — run terbaik (run {k})"))
    return out_dir


# Genetic Algorithm

def run_genetic_algorithm(vehicles, ship, params, runs, base_seed):
    out = OUTPUT_DIR / "ga"
    reset_output_dir(out)                             
    folders = {"A": "A_variasi_iterasi", "B": "B_variasi_populasi"}

    def on_result(group, label, k, r):
        p = r.params
        d = out / folders[group] / f"pop{p['population_size']}_iter{p['max_iterations']}"
        r.save(d / f"run{k}.json")
        plt.close(viz.plot_initial_final(
            vehicles, ship, r.initial_state, r.final_state,
            f"Genetic Algorithm — populasi {p['population_size']}, iterasi {p['max_iterations']}, "
            f"seed {p['seed']}  |  durasi {r.duration_seconds:.2f} s",
            d / f"run{k}_states.png", r.initial_value, r.final_value))
        plt.close(viz.plot_objective(
            r.history, "Genetic Algorithm — objective function vs iterasi", d / f"run{k}_objective.png",
            info=f"Populasi {p['population_size']}  |  {p['max_iterations']} iterasi  |  nilai akhir "
                 f"{r.final_value:g}  |  durasi {r.duration_seconds:.2f} s"))
        (d / f"run{k}_states.txt").write_text(
            ascii_state(vehicles, ship, r.initial_state, f"STATE AWAL — GA {label} run {k}") +
            ascii_state(vehicles, ship, r.final_state, f"STATE AKHIR — GA {label} run {k}"), encoding="utf-8")

    exp = run_ga_experiment(vehicles, ship, params["control_population"], params["iteration_variations"],
                            params["control_iterations"], params["population_variations"], runs, base_seed,
                            on_result=on_result, mutation_rate=params["mutation_rate"],
                            elitism=params["elitism"], use_repair=params["use_repair"])
    titles = {"A": f"GA — populasi tetap {params['control_population']}, variasi banyak iterasi",
              "B": f"GA — iterasi tetap {params['control_iterations']}, variasi jumlah populasi"}
    for g, groups in exp.items():
        plt.close(viz.plot_experiment(groups, titles[g], out / folders[g] / "experiment.png"))
    (out / "summary.md").write_text(experiment_summary(exp), encoding="utf-8")

    best = max((r for groups in exp.values() for rs in groups.values() for r in rs), key=lambda r: r.final_value)
    print(f"\nRun GA terbaik: populasi {best.params['population_size']}, iterasi {best.params['max_iterations']}, "
          f"seed {best.params['seed']}")
    print(ascii_state(vehicles, ship, best.initial_state, "STATE AWAL — run GA terbaik"))
    print(ascii_state(vehicles, ship, best.final_state, "STATE AKHIR — run GA terbaik"))
    return out


# Menu

def choose(title, options):
    print(f"\n=== {title} ===")
    for k, (_, label) in enumerate(options, start=1):
        print(f"  {k}. {label}")
    while True:
        ans = input(f"Pilih [1-{len(options)}]: ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1][0]
        print("  Pilihan tidak valid, coba lagi.")


def ask(prompt, default, cast=int):
    while True:
        ans = input(f"  {prompt} [{default}]: ").strip()
        try:
            return cast(ans) if ans else default
        except ValueError:
            print("  Input tidak valid, coba lagi.")


def ask_params(algo, p):
    print("\n--- Atur parameter (Enter = pakai nilai default) ---")
    p = dict(p)
    ints = lambda s: [int(t) for t in s.replace(",", " ").split()]  # noqa: E731
    questions = {
        "sideways": [("max_sideways", "Maximum sideways move", int)],
        "random_restart": [("max_restart", "Maximum restart", int)],
        "stochastic": [("nmax", "nmax (banyak iterasi)", int)],
        "sa": [("T0", "Temperatur awal T0", float),
               ("stuck_window", "Iterasi tanpa perubahan yang dihitung 'stuck'", int)],
        "ga": [("control_population", "Jumlah populasi kontrol", int),
               ("iteration_variations", "3 variasi banyak iterasi", ints),
               ("control_iterations", "Banyak iterasi kontrol", int),
               ("population_variations", "3 variasi jumlah populasi", ints),
               ("mutation_rate", "Peluang mutasi", float),
               ("elitism", "Elitism (heuristik tambahan, 0 = sesuai PPT)", int),
               ("use_repair", "Repair (heuristik tambahan, y/n)", lambda s: s.lower().startswith("y"))],
    }
    for key, prompt, cast in questions.get(algo, []):
        p[key] = ask(prompt, p[key], cast)
    return p


def main():
    ap = argparse.ArgumentParser(description="IF3070 — local search penataan kendaraan di geladak kapal")
    ap.add_argument("--search", choices=["hc", "sa", "ga"], help="lewati menu pilihan algoritma")
    ap.add_argument("--variant", choices=[k for k, _ in HC_MENU], help="varian hill-climbing")
    ap.add_argument("--custom", action="store_true", help="atur parameter sendiri")
    ap.add_argument("--runs", type=int, default=3, help="banyak run tiap konfigurasi (spesifikasi: 3)")
    ap.add_argument("--seed", type=int, default=None, help="seed dasar (default: acak setiap eksekusi)")
    a = ap.parse_args()

    print("=" * 64 + "\n  IF3070 — Local Search Penataan Kendaraan di Geladak Kapal\n" + "=" * 64)
    algo = a.search or choose("Pilih algoritma local search", ALGORITHM_MENU)
    if algo == "hc":
        algo = a.variant or choose("Pilih jenis Hill-climbing", HC_MENU)
    params = ask_params(algo, DEFAULT_PARAMS[algo]) if a.custom else dict(DEFAULT_PARAMS[algo])
    vehicles, ship = load_problem(DATASET_FILE)
    base_seed = a.seed if a.seed is not None else random.SystemRandom().randrange(1_000_000)

    print(f"\nAlgoritma : {NAMES[algo]}")
    print(f"Data set  : geladak {ship.w} x {ship.l}, kapasitas {ship.max_capacity:g}, {len(vehicles)} kendaraan")
    print(f"Parameter : {params}")
    print(f"Seed dasar: {base_seed}  (ulangi hasil yang sama dengan: --seed {base_seed})")

    t0 = time.perf_counter()
    if algo == "ga":
        out = run_genetic_algorithm(vehicles, ship, params, a.runs, base_seed)
    else:
        out = run_local_search(algo, vehicles, ship, params, a.runs, base_seed)
    print(f"\nSelesai dalam {time.perf_counter() - t0:.1f} s. Semua hasil ada di: {out}")


if __name__ == "__main__":
    sys.exit(main())
