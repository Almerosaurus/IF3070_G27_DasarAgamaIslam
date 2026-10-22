import random
import math
#import utils
#tambahan Genetic Algorithm
from dataclasses import dataclass
import argparse
import json
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")   

import matplotlib.pyplot as plt  

from genetic_algorithm import GAResult, genetic_algorithm, run_ga_experiment, experiment_summary  
import visualization as viz 

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
#-----------------------------------

HORIZONTAL = "Horizontal"
VERTICAL = "Vertical"

@dataclass
class Vehicle:
    id: str
    w: int
    l: int
    shipping_fee: int
    weight: int
    eta: int


@dataclass
class Ship:
    w: int
    l: int
    max_capacity: int


@dataclass
class Placement:
    x: int = 0
    y: int = 0
    orientation: str = VERTICAL

    @property
    def inside(self) -> bool:
        return self.x is not None and self.y is not None

#tambahan Genetic Algorithm
EXAMPLE_VEHICLES = [
    Vehicle("Udin", w=3, l=4, shipping_fee=200, weight=100, eta=3),
    Vehicle("Dino", w=2, l=3, shipping_fee=100, weight=40, eta=5),
    Vehicle("Lia", w=4, l=2, shipping_fee=50, weight=20, eta=2),
    Vehicle("Nia", w=1, l=5, shipping_fee=35, weight=10, eta=7),
    Vehicle("Bon", w=3, l=3, shipping_fee=25, weight=5, eta=4),
    Vehicle("Tin", w=2, l=2, shipping_fee=40, weight=10, eta=6),
    Vehicle("Mer", w=5, l=1, shipping_fee=30, weight=7, eta=1),
    Vehicle("Ska", w=2, l=4, shipping_fee=60, weight=20, eta=8),
]
EXAMPLE_SHIP = Ship(w=10, l=8, max_capacity=500)


def load_problem(path=None):
    """
    Baca problem dari JSON -> (list[Vehicle], Ship). Tanpa path -> data contoh.
    Format: {"ship": {"w": 10, "l": 20, "max_capacity": 500},
             "vehicles": [{"id": "Owen", "w": 3, "l": 4, "shipping_fee": 60, "weight": 5, "eta": 3}, ...]}
    ("dimensions": {"w":..,"l":..} juga diterima, begitu pula nama key gaya spesifikasi: ShippingFee, MaxCapacity, ...)
    """
    if path is None:
        return list(EXAMPLE_VEHICLES), EXAMPLE_SHIP

    def pick(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f"key {keys} tidak ditemukan di {d}")

    def dims(d):
        dd = d.get("dimensions", d.get("Dimensions", d))
        return int(pick(dd, "w", "width")), int(pick(dd, "l", "length"))

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    s = pick(data, "ship", "Ship")
    sw, sl = dims(s)
    ship = Ship(sw, sl, pick(s, "max_capacity", "MaxCapacity", "capacity"))
    vehicles = []
    for v in pick(data, "vehicles", "Vehicles"):
        w, l = dims(v)
        vehicles.append(Vehicle(str(pick(v, "id", "Id", "ID")), w, l,
                                pick(v, "shipping_fee", "ShippingFee", "fee"),
                                pick(v, "weight", "Weight"), v.get("eta", v.get("ETA", 0))))
    return vehicles, ship


def to_placements(state):
    """State GA (None / (x, y, 'H'|'V')) -> list[Placement] milik main.py (di luar kapal = x, y None)."""
    return [Placement(None, None, VERTICAL) if e is None
            else Placement(e[0], e[1], HORIZONTAL if e[2] == "H" else VERTICAL) for e in state]


def print_placements(vehicles, state, title):
    print(f"\n--- {title} ---")
    for v, p in zip(vehicles, to_placements(state)):
        pos = f"({p.x:>2}, {p.y:>2}) {p.orientation}" if p.inside else "di luar kapal"
        print(f"  {v.id:<12} {v.w}x{v.l}  fee={v.shipping_fee:<6g} weight={v.weight:<6g} -> {pos}")


def save_ga_outputs(result, vehicles, ship, out_dir, prefix, animate=False):
    """Simpan JSON + gambar state awal/akhir + plot objective (max & rata-rata populasi)."""
    out_dir = Path(out_dir)
    paths = {"json": result.save(out_dir / f"{prefix}.json")}
    p = result.params
    fig = viz.plot_initial_final(
        vehicles, ship, result.initial_state, result.final_state,
        f"Genetic Algorithm — populasi {p['population_size']}, iterasi {p['max_iterations']}, seed {p['seed']}"
        f"  |  durasi {result.duration_seconds:.2f} s",
        out_dir / f"{prefix}_states.png", result.initial_value, result.final_value)
    plt.close(fig)
    paths["states"] = out_dir / f"{prefix}_states.png"
    fig = viz.plot_objective(
        result.history, "Genetic Algorithm — objective function vs iterasi", out_dir / f"{prefix}_objective.png",
        info=f"Populasi {p['population_size']}  |  {p['max_iterations']} iterasi  |  nilai akhir "
             f"{result.final_value:,.2f}  |  durasi {result.duration_seconds:.2f} s")
    plt.close(fig)
    paths["objective"] = out_dir / f"{prefix}_objective.png"
    if animate:
        paths["animation"] = viz.animate_snapshots(vehicles, ship, result.snapshots,
                                                   out_dir / f"{prefix}_animation.gif", "GA —")
    return paths


def ga_params_from_args(a):
    return dict(crossover_rate=a.crossover_rate, mutation_rate=a.mutation_rate, elitism=a.elitism,
                tournament_size=a.tournament, crossover=a.crossover, use_repair=not a.no_repair, fill=a.fill)


def cmd_ga(a):
    vehicles, ship = load_problem(a.input)
    name = Path(a.input).stem if a.input else "contoh"
    print(f"Genetic Algorithm: {name}, populasi={a.population}, iterasi={a.iterations}, seed={a.seed}")
    result = genetic_algorithm(vehicles, ship, a.population, a.iterations, a.seed, verbose=True,
                               **ga_params_from_args(a))
    print_placements(vehicles, result.initial_state, "STATE AWAL")
    print_placements(vehicles, result.final_state, "STATE AKHIR")
    print("\n" + result.summary())
    out_dir = Path(a.out) if a.out else OUTPUT_DIR / "ga"
    prefix = f"{name}_pop{a.population}_iter{a.iterations}_seed{a.seed}"
    for k, v in save_ga_outputs(result, vehicles, ship, out_dir, prefix, a.animate).items():
        print(f"{k:<10}: {v}")
    if a.show:
        plt.show()


def cmd_ga_experiment(a):
    vehicles, ship = load_problem(a.input)
    name = Path(a.input).stem if a.input else "contoh"
    out = Path(a.out) if a.out else OUTPUT_DIR / "ga_experiment" / name
    folders = {"A": "A_variasi_iterasi", "B": "B_variasi_populasi"}

    def on_result(group, label, k, r):
        cfg = f"pop{r.params['population_size']}_iter{r.params['max_iterations']}"
        save_ga_outputs(r, vehicles, ship, out / folders[group] / cfg, f"run{k}")

    exp = run_ga_experiment(vehicles, ship, a.control_population, a.iteration_variations,
                            a.control_iterations, a.population_variations, a.runs, a.seed,
                            on_result=on_result, **ga_params_from_args(a))
    titles = {"A": f"GA — populasi tetap {a.control_population}, variasi banyak iterasi",
              "B": f"GA — iterasi tetap {a.control_iterations}, variasi jumlah populasi"}
    for g, groups in exp.items():
        plt.close(viz.plot_experiment(groups, titles[g], out / folders[g] / "experiment.png"))
    (out / "summary.md").write_text(experiment_summary(exp), encoding="utf-8")
    print(f"Selesai. Hasil di {out}")


def cmd_ga_visualize(a):
    r = GAResult.load(a.result)
    vehicles = [Vehicle(v["id"], v["w"], v["l"], v["shipping_fee"], v["weight"], v["eta"])
                for v in r.problem["vehicles"]]
    s = r.problem["ship"]
    ship = Ship(s["w"], s["l"], s["max_capacity"])
    print(r.summary())
    for k, v in save_ga_outputs(r, vehicles, ship, Path(a.result).parent, Path(a.result).stem, a.animate).items():
        print(f"{k:<10}: {v}")
    if a.show:
        plt.show()


def build_parser():
    ap = argparse.ArgumentParser(description="IF3070 — local search penataan kendaraan di geladak kapal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def ga_args(p):
        p.add_argument("--input", "-i", default=None, help="file problem JSON (default: data contoh)")
        p.add_argument("--out", "-o", default=None, help="folder output")
        p.add_argument("--crossover-rate", type=float, default=0.9)
        p.add_argument("--mutation-rate", type=float, default=0.3)
        p.add_argument("--elitism", type=int, default=2)
        p.add_argument("--tournament", type=int, default=3)
        p.add_argument("--crossover", choices=["spatial", "uniform"], default="spatial")
        p.add_argument("--no-repair", action="store_true", help="tanpa heuristik repair (hanya penalti)")
        p.add_argument("--fill", action="store_true", help="repair juga mengisi ruang kosong secara greedy")

    g = sub.add_parser("ga", help="jalankan Genetic Algorithm sekali")
    ga_args(g)
    g.add_argument("--population", "-p", type=int, default=50, help="jumlah populasi")
    g.add_argument("--iterations", "-n", type=int, default=100, help="banyak iterasi (generasi)")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--animate", action="store_true", help="simpan animasi GIF")
    g.add_argument("--show", action="store_true", help="tampilkan jendela gambar")

    e = sub.add_parser("ga-experiment", help="eksperimen GA sesuai spesifikasi")
    ga_args(e)
    e.add_argument("--control-population", type=int, default=50)
    e.add_argument("--iteration-variations", type=int, nargs=3, default=[50, 100, 200])
    e.add_argument("--control-iterations", type=int, default=100)
    e.add_argument("--population-variations", type=int, nargs=3, default=[20, 50, 100])
    e.add_argument("--runs", type=int, default=3)
    e.add_argument("--seed", type=int, default=0, help="seed dasar")

    v = sub.add_parser("ga-visualize", help="gambar ulang dari file JSON hasil GA")
    v.add_argument("result")
    v.add_argument("--animate", action="store_true")
    v.add_argument("--show", action="store_true")
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    {"ga": cmd_ga, "ga-experiment": cmd_ga_experiment, "ga-visualize": cmd_ga_visualize}[args.cmd](args)
