from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

# ---------------- gaya ----------------
BG, TEXT, TEXT_2, MUTED, GRID, DECK, RED = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#f1efe9", "#d03b3b")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CMAP = LinearSegmentedColormap.from_list("fee", ["#cfe0f5", "#7aaee8", "#2a78d6", "#174a8a"])
SECONDARY = ("acceptance_probability", "temperature")      # diplot di subplot terpisah
LABELS = {"max": "Maksimum populasi", "mean": "Rata-rata populasi", "best_so_far": "Terbaik sejauh ini",
          "value": "Objective value", "acceptance_probability": r"$e^{\Delta E / T}$",
          "temperature": "Temperatur"}
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "legend.frameon": False, "font.size": 10, "lines.linewidth": 2,
})


# ---------------- util ----------------
def fee_of(v):
    return float(v.shipping_fee if hasattr(v, "shipping_fee") else v.fee)


def capacity_of(ship):
    return float(ship.max_capacity if hasattr(ship, "max_capacity") else ship.max_cap)


def size_of(v, ori):
    return (v.w, v.l) if ori == "H" else (v.l, v.w)


def save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")


def state_stats(vehicles, ship, state):
    """Total fee, berat, sel terpakai, dan pelanggaran constraint sebuah state."""
    count = np.zeros((ship.l, ship.w), dtype=int)
    fee = weight = out_of_bounds = 0
    for v, e in zip(vehicles, state):
        if e is None:
            continue
        x, y, ori = e
        dx, dy = size_of(v, ori)
        fee += fee_of(v)
        weight += v.weight
        out_of_bounds += x < 0 or y < 0 or x + dx > ship.w or y + dy > ship.l
        count[max(y, 0):y + dy, max(x, 0):x + dx] += 1
    overlap = int(np.maximum(count - 1, 0).sum())
    valid = overlap == 0 and out_of_bounds == 0 and weight <= capacity_of(ship)
    return {"fee": fee, "weight": weight, "capacity": capacity_of(ship), "count": count,
            "overlap": overlap, "out_of_bounds": out_of_bounds, "valid": valid,
            "n_inside": sum(e is not None for e in state), "used": int(np.count_nonzero(count)),
            "cells": ship.w * ship.l}


# ---------------- gambar state ----------------
def _vehicle(ax, x, y, dx, dy, color, label, faded=False):
    ax.add_patch(Rectangle((x + 0.06, y + 0.06), dx - 0.12, dy - 0.12, facecolor=color,
                           alpha=0.45 if faded else 0.9, edgecolor="none"))
    dark = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) < 0.55 and not faded
    ax.text(x + dx / 2, y + dy / 2, label if min(dx, dy) >= 2 else label.split("\n")[0], ha="center",
            va="center", fontsize=7.5, color="white" if dark else TEXT, rotation=90 if dx == 1 < dy else 0)


def draw_state(ax, vehicles, ship, state, title, norm, value=None):
    """Geladak (atas) dan kendaraan di luar kapal (bawah) pada satu Axes."""
    st = state_stats(vehicles, ship, state)
    W, L = ship.w, ship.l
    color = lambda v: CMAP(norm(fee_of(v) / (v.w * v.l)))  # noqa: E731

    ax.add_patch(Rectangle((0, 0), W, L, facecolor=DECK, edgecolor=TEXT_2, linewidth=1.5, zorder=0))
    for c in range(1, W):
        ax.plot([c, c], [0, L], color="white", linewidth=0.8, zorder=0.5)
    for r in range(1, L):
        ax.plot([0, W], [r, r], color="white", linewidth=0.8, zorder=0.5)
    for i, (v, e) in enumerate(zip(vehicles, state)):
        if e is not None:
            _vehicle(ax, e[0], e[1], *size_of(v, e[2]), color(v), f"#{i}\n{fee_of(v):g}")
    for r, c in zip(*np.nonzero(st["count"] > 1)):             # sel overlap diarsir merah
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor="none", edgecolor=RED, hatch="xxx"))

    # kendaraan di luar kapal, disusun berderet di bawah geladak
    outside = sorted((i for i, e in enumerate(state) if e is None), key=lambda k: -vehicles[k].w * vehicles[k].l)
    top = L + 1.8
    ax.text(W / 2, top - 0.5, f"Di luar kapal ({len(outside)} kendaraan)", ha="center", fontsize=9,
            color=TEXT_2, fontweight="bold")
    cx, cy, row_h = 0.0, top, 0.0
    for i in outside:
        v = vehicles[i]
        dx, dy = max(v.w, v.l), min(v.w, v.l)
        if cx + dx > W and cx > 0:
            cx, cy, row_h = 0.0, cy + row_h + 0.4, 0.0
        _vehicle(ax, cx, cy, dx, dy, color(v), f"#{i}\n{fee_of(v):g}", faded=True)
        cx, row_h = cx + dx + 0.4, max(row_h, dy)

    ax.set_xlim(-0.5, W + 0.5)
    ax.set_ylim((cy + row_h if outside else top) + 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(W) + 0.5, range(W), fontsize=7)
    ax.set_yticks(np.arange(L) + 0.5, range(L), fontsize=7)
    ax.tick_params(length=0, colors=MUTED)
    ax.xaxis.tick_top()

    status = "VALID" if st["valid"] else (f"TIDAK VALID — overlap {st['overlap']} sel, "
                                          f"{st['out_of_bounds']} kendaraan keluar batas")
    info = (f"Value = {st['fee'] if value is None else value:,.0f}   |   Berat = {st['weight']:g} / {st['capacity']:g}"
            f"   |   Masuk {st['n_inside']} / {len(vehicles)}   |   Utilisasi {st['used'] / st['cells']:.0%}   |   {status}")
    ax.set_title(title, fontsize=13, pad=36)
    ax.annotate(info, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 18), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color=TEXT_2 if st["valid"] else RED)
    return st


def plot_initial_final(vehicles, ship, initial_state, final_state, title="", save_path=None,
                       initial_value=None, final_value=None):
    """State awal & akhir berdampingan; judul, panel, dan colorbar dijajarkan di tengah."""
    d = [fee_of(v) / (v.w * v.l) for v in vehicles]
    norm = Normalize(min(d), max(d) if max(d) > min(d) else min(d) + 1)
    fig, axes = plt.subplots(1, 2)
    draw_state(axes[0], vehicles, ship, initial_state, "State Awal", norm, initial_value)
    draw_state(axes[1], vehicles, ship, final_state, "State Akhir", norm, final_value)

    # samakan tinggi kedua panel, lalu sesuaikan ukuran gambar dengan bentuk geladak
    span_x = max(abs(a.get_xlim()[1] - a.get_xlim()[0]) for a in axes)
    span_y = max(abs(a.get_ylim()[0] - a.get_ylim()[1]) for a in axes)
    for a in axes:
        a.set_ylim(a.get_ylim()[1] + span_y, a.get_ylim()[1])
        a.set_anchor("N")
    panel_h = 7.0
    panel_w = panel_h * span_x / span_y
    fig.set_size_inches(2 * panel_w + 2.5, panel_h + 3.0)
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.13, top=1 - 1.25 / (panel_h + 3.0), wspace=0.15)
    fig.suptitle(title, y=0.985, fontsize=14, fontweight="bold")
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_label("Fee per sel (juta rupiah / sel)   —   label: #index kendaraan, angka bawah = ShippingFee",
                 color=TEXT_2, fontsize=9)
    cb.outline.set_visible(False)
    cax.grid(False)
    save(fig, save_path)
    return fig


# ---------------- objective vs iterasi ----------------
def plot_objective(history, title="Objective function vs iterasi", save_path=None, info=None):
    """history = {"iteration": [...], <nama deret>: [...]}. Deret e^(dE/T) & temperatur di subplot sendiri."""
    it = history["iteration"]
    primary = [k for k in history if k not in ("iteration", *SECONDARY)]
    secondary = [k for k in history if k in SECONDARY]
    fig, axes = plt.subplots(1 + len(secondary), 1, figsize=(11, 4.6 + 2.8 * len(secondary)), sharex=True,
                             squeeze=False, layout="constrained")
    ax = axes[0][0]
    marker = "o" if len(it) <= 60 else None             # titik agar history pendek tetap terlihat
    for ci, k in enumerate(k for k in primary if k != "best_so_far"):
        ax.plot(it, history[k], color=SERIES[ci], label=LABELS.get(k, k), marker=marker, markersize=5)
        ax.annotate(f"{history[k][-1]:,.1f}", (it[-1], history[k][-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=TEXT_2)
    if "best_so_far" in history:
        ax.plot(it, history["best_so_far"], color=MUTED, linewidth=1.4, linestyle="--", label=LABELS["best_so_far"])
    if len(it) == 1:
        ax.set_xlim(-1, 1)
        ax.set_xticks([0])
        ax.text(0.5, 0.8, "Hanya 1 titik: tidak ada successor yang lebih baik dari state awal",
                transform=ax.transAxes, ha="center", fontsize=9, color=TEXT_2)
    ax.set_title(title, fontsize=12, pad=20 if info else 6)
    if info:                                           # keterangan di bawah judul
        ax.text(0.5, 1.015, info, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=TEXT_2)
    ax.set_ylabel("Objective function")
    ax.legend(loc="lower right", ncol=len(primary))
    for r, k in enumerate(secondary, start=1):
        a = axes[r][0]
        if k == "acceptance_probability":
            a.scatter(it, history[k], color=SERIES[r], s=3, alpha=0.6, linewidths=0)
            a.set_ylim(-0.03, 1.03)
        else:
            a.plot(it, history[k], color=SERIES[r], linewidth=1.2)
        a.set_title(f"{LABELS[k]} vs iterasi", fontsize=11)
        a.set_ylabel(LABELS[k])
    axes[-1][0].set_xlabel("Iterasi")
    save(fig, save_path)
    return fig


# ---------------- eksperimen ----------------
def plot_experiment(groups, title, save_path=None, series_key="max", second_key="mean"):
    """
    groups = {label: [hasil, ...]}; setiap hasil punya .history, .final_value, .duration_seconds.
    GA: series_key="max", second_key="mean".  HC/SA: series_key="value", second_key="best_so_far".
    """
    labels = list(groups)
    fig = plt.figure(figsize=(15, 9), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    ax1, ax2 = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2])
    ax_val, ax_dur, ax_tab = (fig.add_subplot(gs[1, i]) for i in range(3))
    rows = []
    for ci, lab in enumerate(labels):
        c, res = SERIES[ci % len(SERIES)], groups[lab]
        hs = [r.history for r in res]
        n = min(len(h["iteration"]) for h in hs)
        marker = "o" if n <= 60 else None
        for h in hs:
            ax1.plot(h["iteration"], h[series_key], color=c, alpha=0.3, linewidth=1, marker=marker, markersize=3)
        x = hs[0]["iteration"][:n]
        ax1.plot(x, np.mean([h[series_key][:n] for h in hs], axis=0), color=c, linewidth=2.2, label=lab,
                 marker=marker, markersize=5)
        ax2.plot(x, np.mean([h[second_key][:n] for h in hs], axis=0), color=c, linewidth=2)
        vals = np.array([r.final_value for r in res])
        durs = np.array([r.duration_seconds for r in res])
        ax_val.scatter(ci + np.linspace(-0.12, 0.12, len(vals)), vals, color=c, s=46, zorder=3,
                       edgecolor="white", linewidth=1.5)
        ax_val.hlines(vals.mean(), ci - 0.28, ci + 0.28, color=c, linewidth=2.5)
        ax_dur.bar(ci, durs.mean(), color=c, width=0.6, zorder=2)
        ax_dur.annotate(f"{durs.mean():.2f}s", (ci, durs.mean()), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8.5, color=TEXT_2)
        rows.append([lab, f"{vals.mean():.1f}", f"{vals.std():.1f}", f"{vals.max():.1f}", f"{durs.mean():.2f}"])

    ax1.set_title(f"{LABELS[series_key]} vs iterasi (tipis = tiap run, tebal = rata-rata)", fontsize=11)
    ax1.set(xlabel="Iterasi", ylabel="Objective function")
    ax1.legend(loc="lower right")
    ax2.set_title(f"{LABELS[second_key]} vs iterasi", fontsize=11)
    ax2.set_xlabel("Iterasi")
    for a, t in ((ax_val, "Nilai akhir per run"), (ax_dur, "Durasi rata-rata (s)")):
        a.set_title(t, fontsize=11)
        a.set_xticks(range(len(labels)), labels, fontsize=8.5)
        a.grid(axis="x", visible=False)
    ax_tab.axis("off")
    ax_tab.set_title("Ringkasan nilai akhir", fontsize=11)
    tab = ax_tab.table(cellText=rows, colLabels=["Konfigurasi", "Rata-rata", "Std", "Max", "Durasi"],
                       loc="center", cellLoc="center", colWidths=[0.4, 0.15, 0.15, 0.15, 0.15])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.6)
    for (r, _), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_text_props(fontweight="bold" if r == 0 else "normal")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    save(fig, save_path)
    return fig


# ---------------- ASCII ----------------
def draw_solution(state, vehicles, ship, title="SOLUSI"):
    """Cetak geladak dalam bentuk teks (A = kendaraan #0, B = #1, ...) beserta statistik state."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    grid = [["."] * ship.w for _ in range(ship.l)]
    for i, e in enumerate(state):
        if e is None:
            continue
        dx, dy = size_of(vehicles[i], e[2])
        for cx in range(e[0], min(e[0] + dx, ship.w)):
            for cy in range(e[1], min(e[1] + dy, ship.l)):
                grid[cy][cx] = letters[i % len(letters)]

    st = state_stats(vehicles, ship, state)
    line = "  " + "-" * (3 * ship.w + 5)
    print("\n" + "=" * 60 + f"\n  {title}\n" + "=" * 60)
    print(f"  Geladak : {ship.w} (lebar) x {ship.l} (panjang)\n  Kapasitas Maksimal : {st['capacity']:g}\n")
    print("  y\\x " + "".join(f"{c:3d}" for c in range(ship.w)))
    print(line)
    for cy, row in enumerate(grid):
        print(f"  {cy:2d}  " + "".join(f"  {c}" for c in row))
    print(line)
    print("\n  --- Statistik ---")
    print(f"  Total kendaraan       : {len(vehicles)}")
    print(f"  Di dalam geladak      : {st['n_inside']}")
    print(f"  Di luar geladak       : {len(vehicles) - st['n_inside']}")
    print(f"  Total ShippingFee     : {st['fee']:.1f} juta rupiah")
    print(f"  Total berat           : {st['weight']:.1f} / {st['capacity']:g} unit")
    print(f"  Area terpakai         : {st['used']} / {st['cells']} ({st['used'] / st['cells']:.1%})")
    print(f"  Status                : {'VALID' if st['valid'] else 'TIDAK VALID'}")