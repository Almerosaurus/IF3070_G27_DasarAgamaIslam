"""
Visualisasi (matplotlib) — dipakai bersama oleh semua algoritma.

Konvensi state sama dengan simulated_annealing.py & genetic_algorithm.py:
    state[i] = None (di luar kapal)  atau  (x, y, 'H'/'V')
    'H' -> veh.w sel ke arah x, veh.l sel ke arah y ; 'V' -> sebaliknya
    geladak: x = 0..ship.w-1 (kolom), y = 0..ship.l-1 (baris)

Fungsi utama:
    plot_state(vehicles, ship, state, title, save_path)
    plot_initial_final(vehicles, ship, initial_state, final_state, title, save_path)
    plot_objective(history, title, save_path)          # history = {"iteration": [...], "<nama>": [...], ...}
    plot_experiment(groups, title, save_path)          # groups = {label: [hasil, ...]}
    animate_snapshots(vehicles, ship, snapshots, save_path)

Warna kendaraan = fee per sel (biru terang = murah, biru gelap = mahal).
Sel overlap diberi arsir merah; kendaraan yang keluar batas geladak bergaris merah putus-putus.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

# ---------------- gaya ----------------
SURFACE, TEXT, TEXT_2, MUTED, GRID, DECK, CRITICAL = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#f1efe9", "#d03b3b")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DENSITY_CMAP = LinearSegmentedColormap.from_list("fee_density", ["#cfe0f5", "#7aaee8", "#2a78d6", "#174a8a"])
SECONDARY_SERIES = ("acceptance_probability", "temperature")
LABELS = {
    "max": "Maksimum populasi", "mean": "Rata-rata populasi", "min": "Minimum populasi",
    "best_so_far": "Terbaik sejauh ini", "value": "Objective value",
    "acceptance_probability": r"$e^{\Delta E / T}$", "temperature": "Temperatur",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT, "legend.frameon": False,
    "font.size": 10, "lines.linewidth": 2,
})


# ---------------- util ----------------
def _fee(v) -> float:
    return float(v.shipping_fee if hasattr(v, "shipping_fee") else v.fee)


def _cap(ship) -> float:
    return float(ship.max_capacity if hasattr(ship, "max_capacity") else ship.max_cap)


def _fp(v, ori):
    return (v.w, v.l) if ori == "H" else (v.l, v.w)


def _get(r, key):
    """Ambil atribut dari objek hasil (GAResult, dll.) atau dict."""
    return r[key] if isinstance(r, dict) else getattr(r, key)


def state_stats(vehicles, ship, state) -> dict:
    W, L = ship.w, ship.l
    count = np.zeros((L, W), dtype=int)
    fee = weight = 0.0
    oob = 0
    for v, e in zip(vehicles, state):
        if e is None:
            continue
        x, y, ori = e
        dx, dy = _fp(v, ori)
        fee += _fee(v)
        weight += v.weight
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + dx, W), min(y + dy, L)
        a = max(0, x1 - x0) * max(0, y1 - y0)
        oob += dx * dy - a
        if a:
            count[y0:y1, x0:x1] += 1
    overlap = int(np.maximum(count - 1, 0).sum())
    over = max(0.0, weight - _cap(ship))
    return {"fee": fee, "weight": weight, "capacity": _cap(ship), "overlap": overlap, "oob": int(oob),
            "overweight": over, "n_inside": sum(e is not None for e in state),
            "used": int(np.count_nonzero(count)), "cells": W * L, "count": count,
            "valid": overlap == 0 and oob == 0 and over <= 1e-9}


def density_norm(vehicles) -> Normalize:
    d = [_fee(v) / (v.w * v.l) for v in vehicles]
    return Normalize(min(d), max(d) if max(d) > min(d) else min(d) + 1)


def _text_color(rgba) -> str:
    return "white" if (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]) < 0.55 else TEXT


def _draw_vehicle(ax, x, y, dx, dy, color, label, sub, edge=None, alpha=1.0):
    ax.add_patch(Rectangle((x + 0.06, y + 0.06), dx - 0.12, dy - 0.12, facecolor=color, alpha=alpha,
                           edgecolor=edge or "none", linewidth=1.6 if edge else 0, linestyle="--" if edge else "-"))
    txt = label if min(dx, dy) < 2 else f"{label}\n{sub}"
    ax.text(x + dx / 2, y + dy / 2, txt, ha="center", va="center", fontsize=7.5 if min(dx, dy) >= 2 else 6.5,
            color=_text_color(color) if alpha >= 0.8 else TEXT, rotation=90 if (dx == 1 and dy > 1) else 0,
            linespacing=1.1)


# ---------------- state ----------------
def draw_state(ax, vehicles, ship, state, title=None, norm=None, value=None) -> dict:
    """Gambar geladak + kendaraan di luar kapal + info value pada sebuah Axes."""
    st = state_stats(vehicles, ship, state)
    norm = norm or density_norm(vehicles)
    W, L = ship.w, ship.l

    ax.add_patch(Rectangle((0, 0), W, L, facecolor=DECK, edgecolor=TEXT_2, linewidth=1.5, zorder=0))
    for c in range(1, W):
        ax.plot([c, c], [0, L], color="white", linewidth=0.8, zorder=0.5)
    for r in range(1, L):
        ax.plot([0, W], [r, r], color="white", linewidth=0.8, zorder=0.5)

    max_x = W
    for i, (v, e) in enumerate(zip(vehicles, state)):
        if e is None:
            continue
        x, y, ori = e
        dx, dy = _fp(v, ori)
        oob = x < 0 or y < 0 or x + dx > W or y + dy > L
        max_x = max(max_x, x + dx)
        _draw_vehicle(ax, x, y, dx, dy, DENSITY_CMAP(norm(_fee(v) / (v.w * v.l))), f"#{i}", f"{_fee(v):g}",
                      edge=CRITICAL if oob else None, alpha=0.9)
    for r, c in zip(*np.nonzero(st["count"] > 1)):
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor="none", edgecolor=CRITICAL, hatch="xxx", linewidth=0.8, zorder=3))

    # kendaraan di luar kapal (disusun berderet hanya untuk tampilan):
    # di bawah geladak bila geladak melebar, di kanan geladak bila geladak memanjang ke bawah
    outside = [i for i, e in enumerate(state) if e is None]
    side = L > W
    x0 = max_x + 1.5 if side else 0.0
    top = 0.6 if side else L + 1.8
    ax.text(x0, top - 0.5, f"Di luar kapal ({len(outside)} kendaraan)", fontsize=9, color=TEXT_2,
            va="bottom", fontweight="bold")
    limit = max(W, 12)
    cx, cy, shelf = x0, top, 0.0
    for i in sorted(outside, key=lambda k: -vehicles[k].w * vehicles[k].l):
        v = vehicles[i]
        dx, dy = max(v.w, v.l), min(v.w, v.l)
        if cx + dx > x0 + limit and cx > x0:
            cx, cy, shelf = x0, cy + shelf + 0.4, 0.0
        _draw_vehicle(ax, cx, cy, dx, dy, DENSITY_CMAP(norm(_fee(v) / (v.w * v.l))), f"#{i}", f"{_fee(v):g}", alpha=0.45)
        cx += dx + 0.4
        shelf = max(shelf, dy)
    bottom = max(cy + shelf if outside else top, L if side else 0)

    ax.set_xlim(-0.6, (x0 + limit if side else max(max_x, limit)) + 0.6)
    ax.set_ylim(bottom + 0.6, -0.6)
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(W) + 0.5, [str(c) if W <= 12 or c % 2 == 0 else "" for c in range(W)], fontsize=7)
    ax.set_yticks(np.arange(L) + 0.5, [str(r) if L <= 12 or r % 2 == 0 else "" for r in range(L)], fontsize=7)
    ax.tick_params(length=0, colors=MUTED)
    ax.xaxis.tick_top()

    val = st["fee"] if value is None else value
    status = "VALID" if st["valid"] else (f"TIDAK VALID — overlap {st['overlap']} sel, keluar batas {st['oob']} sel, "
                                          f"kelebihan berat {st['overweight']:g}")
    info = (f"Value = {val:,.2f}  |  Total fee = {st['fee']:,.0f} jt  |  Berat = {st['weight']:g}/{st['capacity']:g}  |  "
            f"Masuk {st['n_inside']}/{len(vehicles)}  |  Utilisasi {st['used'] / st['cells'] * 100:.1f}%\n{status}")
    if title:
        ax.set_title(title, loc="left", pad=52)
    ax.annotate(info, xy=(0, 1), xycoords="axes fraction", xytext=(0, 16), textcoords="offset points",
                fontsize=8.5, color=TEXT_2 if st["valid"] else CRITICAL, va="bottom", ha="left")
    return st


def _colorbar(fig, axes, norm):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DENSITY_CMAP), ax=axes, orientation="horizontal",
                      fraction=0.03, pad=0.03, aspect=50)
    cb.set_label("Fee per sel (juta rupiah / sel)  —  label: #index kendaraan, angka bawah = ShippingFee",
                 color=TEXT_2, fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=TEXT_2)


def _save(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")


def plot_state(vehicles, ship, state, title="State", save_path=None, value=None):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    norm = density_norm(vehicles)
    draw_state(ax, vehicles, ship, state, title, norm, value)
    _colorbar(fig, [ax], norm)
    _save(fig, save_path)
    return fig


def plot_initial_final(vehicles, ship, initial_state, final_state, title="", save_path=None,
                       initial_value=None, final_value=None):
    """State awal & akhir berdampingan."""
    norm = density_norm(vehicles)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.8))
    draw_state(axes[0], vehicles, ship, initial_state, "State Awal", norm, initial_value)
    draw_state(axes[1], vehicles, ship, final_state, "State Akhir", norm, final_value)
    if title:
        fig.suptitle(title, x=0.02, y=1.04, ha="left", fontsize=13, fontweight="bold")
    _colorbar(fig, list(axes), norm)
    _save(fig, save_path)
    return fig


# ---------------- objective vs iterasi ----------------
def plot_objective(history: dict, title="Objective function vs iterasi", save_path=None,
                   hide=("min",), info: str | None = None):
    """
    history = {"iteration": [...], "<deret>": [...], ...}
    Deret "acceptance_probability" / "temperature" (SA) otomatis diplot di subplot terpisah.
    """
    it = history["iteration"]
    primary = [k for k in history if k != "iteration" and k not in SECONDARY_SERIES and k not in hide]
    secondary = [k for k in history if k in SECONDARY_SERIES]
    fig, axes = plt.subplots(1 + len(secondary), 1, figsize=(11, 4.6 + 2.8 * len(secondary)),
                             sharex=True, squeeze=False)
    ax = axes[0][0]
    ci = 0
    for k in primary:
        if k == "best_so_far":
            ax.plot(it, history[k], color=MUTED, linewidth=1.4, linestyle="--", label=LABELS.get(k, k))
            continue
        ax.plot(it, history[k], color=SERIES[ci % len(SERIES)], label=LABELS.get(k, k))
        ax.annotate(f"{history[k][-1]:,.1f}", (it[-1], history[k][-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=TEXT_2)
        ci += 1
    ax.set_ylabel("Objective function")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", ncol=max(1, len(primary)))
    if info:
        ax.text(0.01, 0.97, info, transform=ax.transAxes, va="top", fontsize=9, color=TEXT_2)
    for r, k in enumerate(secondary, start=1):
        a = axes[r][0]
        col = SERIES[(ci + r - 1) % len(SERIES)]
        if k == "acceptance_probability":
            a.scatter(it, history[k], color=col, s=3, alpha=0.6, linewidths=0)
            a.set_ylim(-0.03, 1.03)
        else:
            a.plot(it, history[k], color=col, linewidth=1.2)
        a.set_ylabel(LABELS.get(k, k))
        a.set_title(LABELS.get(k, k) + " vs iterasi", loc="left", fontsize=11)
    axes[-1][0].set_xlabel("Iterasi")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------- eksperimen ----------------
def plot_experiment(groups: dict, title: str, save_path=None, series_key: str = "max",
                    second_key: str = "mean"):
    """
    groups = {label: [hasil, ...]}; hasil = objek/dict yang punya `history`, `final_value`,
    `duration_seconds`. Untuk GA: series_key="max", second_key="mean".
    Untuk SA/HC: series_key="value", second_key="best_so_far".
    """
    labels = list(groups)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    ax1, ax2 = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2])
    ax_val, ax_dur, ax_tab = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    rows = []
    for ci, lab in enumerate(labels):
        c = SERIES[ci % len(SERIES)]
        res = groups[lab]
        hs = [_get(r, "history") for r in res]
        for h in hs:
            ax1.plot(h["iteration"], h[series_key], color=c, alpha=0.3, linewidth=1)
        n = min(len(h["iteration"]) for h in hs)
        ax1.plot(hs[0]["iteration"][:n], np.mean([h[series_key][:n] for h in hs], axis=0), color=c,
                 linewidth=2.2, label=lab)
        if second_key in hs[0]:
            ax2.plot(hs[0]["iteration"][:n], np.mean([h[second_key][:n] for h in hs], axis=0), color=c, linewidth=2)
        vals = np.array([_get(r, "final_value") for r in res])
        durs = np.array([_get(r, "duration_seconds") for r in res])
        jit = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.zeros(1)
        ax_val.scatter(ci + jit, vals, color=c, s=46, zorder=3, edgecolor="white", linewidth=1.5)
        ax_val.hlines(vals.mean(), ci - 0.28, ci + 0.28, color=c, linewidth=2.5)
        ax_dur.bar(ci, durs.mean(), color=c, width=0.6, zorder=2)
        ax_dur.errorbar(ci, durs.mean(), yerr=durs.std(), color=TEXT_2, capsize=4, linewidth=1)
        ax_dur.annotate(f"{durs.mean():.2f}s", (ci, durs.mean() + durs.std()), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8.5, color=TEXT_2)
        rows.append([lab, f"{vals.mean():.1f}", f"{vals.std():.1f}", f"{vals.max():.1f}", f"{durs.mean():.2f}"])

    ax1.set_title(f"{LABELS.get(series_key, series_key)} vs iterasi (tipis = tiap run, tebal = rata-rata)",
                  loc="left", fontsize=11)
    ax1.set_xlabel("Iterasi")
    ax1.set_ylabel("Objective function")
    ax1.legend(loc="lower right")
    ax2.set_title(f"{LABELS.get(second_key, second_key)} vs iterasi", loc="left", fontsize=11)
    ax2.set_xlabel("Iterasi")
    for a, t in ((ax_val, "Nilai akhir per run"), (ax_dur, "Durasi rata-rata (s)")):
        a.set_title(t, loc="left", fontsize=11)
        a.set_xticks(range(len(labels)), labels, fontsize=8.5)
        a.grid(axis="x", visible=False)
    ax_val.set_xlim(-0.6, len(labels) - 0.4)
    ax_tab.axis("off")
    tab = ax_tab.table(cellText=rows, colLabels=["Konfigurasi", "Rata-rata", "Std", "Max", "Durasi"],
                       loc="center", cellLoc="center", colWidths=[0.4, 0.15, 0.15, 0.15, 0.15])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.6)
    for (r, _), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax_tab.set_title("Ringkasan nilai akhir", loc="left", fontsize=11)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------- animasi ----------------
def animate_snapshots(vehicles, ship, snapshots: list, save_path, title="", fps: int = 3):
    """snapshots = [{"iteration": t, "value": v, "state": state}, ...] -> GIF"""
    norm = density_norm(vehicles)
    fig, ax = plt.subplots(figsize=(11, 8.5))

    def update(k):
        ax.clear()
        s = snapshots[k]
        draw_state(ax, vehicles, ship, s["state"], f"{title} iterasi {s['iteration']}".strip(), norm, s.get("value"))

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // fps)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(save_path), writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return save_path
