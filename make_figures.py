"""Grafiki aprakstam no results/*.csv  ->  report/img/*.png"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "report" / "img"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def read(name: str) -> list[dict]:
    with open(RES / name) as fh:
        return list(csv.DictReader(fh))


def fig_results(instances: list[dict], runs: list[dict]) -> None:
    main = [r for r in runs if r["experiment"] == "main"]
    best_ts = {i["instance"]: min(int(r["best_k"]) for r in main if r["instance"] == i["instance"])
               for i in instances}
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for y, inst in enumerate(instances):
        name = inst["instance"]
        ds, ts = int(inst["dsatur_k"]), best_ts[name]
        lb, bk = int(inst["clique_lb"]), int(inst["best_known"])
        ax.plot([ts, ds], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter(lb, y, s=130, facecolors="none", edgecolors=INK2, lw=1.2, zorder=6,
                   label="Apakšējā robeža (klika)" if y == 0 else None)
        ax.scatter(ds, y, s=46, color=ORANGE, edgecolors=SURFACE, lw=1.5, zorder=3,
                   label="DSatur (sākums)" if y == 0 else None)
        ax.scatter(ts, y, s=46, color=BLUE, edgecolors=SURFACE, lw=1.5, zorder=4,
                   label="Tabu meklēšana" if y == 0 else None)
        ax.scatter(bk, y, s=120, marker="|", color=INK, lw=1.6, zorder=5,
                   label="Labākais zināmais" if y == 0 else None)
    ax.set_yticks(range(len(instances)))
    ax.set_yticklabels([f"{i['instance']}  ({i['exams']})" for i in instances])
    ax.invert_yaxis()
    ax.set_xlabel("Laika slotu skaits sesijā")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, fontsize=8, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_rezultati.png", dpi=200)
    plt.close(fig)


def ecdf(times: list[float], horizon: float) -> tuple[list[float], list[float]]:
    xs, ys = [0.0], [0.0]
    ok = sorted(t for t in times if not math.isnan(t))
    for i, t in enumerate(ok, 1):
        xs += [t, t]
        ys += [ys[-1], 100 * i / len(times)]
    xs.append(horizon)
    ys.append(ys[-1])
    return xs, ys


def fig_restarts(runs: list[dict]) -> None:
    target = 30
    groups = {
        "Ar restartiem (50 000 iter.)": [r for r in runs if r["experiment"] == "main" and r["instance"] == "uta-s-92"],
        "Bez restartiem": [r for r in runs if r["experiment"] == "restart"],
    }
    horizon = max(float(r["elapsed"]) for g in groups.values() for r in g)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for (label, rows), color in zip(groups.items(), [BLUE, ORANGE]):
        times = [float(r["time_to_best"]) if int(r["best_k"]) <= target else float("nan") for r in rows]
        xs, ys = ecdf(times, horizon)
        ax.plot(xs, ys, color=color, lw=2, solid_joinstyle="round", label=label)
        ax.annotate(f"{ys[-1]:.0f}%", (horizon, ys[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    ax.set_xlim(0, horizon * 1.06)
    ax.set_ylim(-3, 105)
    ax.set_xlabel("Laiks, s")
    ax.set_ylabel("Palaidieni ar 30 slotiem, %")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_restarti.png", dpi=200)
    plt.close(fig)


def fig_start_strategy(instances: list[dict], runs: list[dict]) -> None:
    main = defaultdict(list)
    fixed = defaultdict(list)
    for r in runs:
        if r["experiment"] == "main":
            main[r["instance"]].append(int(r["best_k"]) <= int(r["best_known"]))
        elif r["experiment"] == "fixed_k":
            fixed[r["instance"]].append(int(r["best_k"]) != -1)
    names = [i["instance"] for i in instances]
    a = [100 * sum(main[n]) / len(main[n]) for n in names]
    b = [100 * sum(fixed[n]) / len(fixed[n]) for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    w = 0.38
    xs = range(len(names))
    ax.bar([x - w / 2 - 0.01 for x in xs], a, width=w, color=BLUE, label="DSatur + pakāpeniska k samazināšana")
    ax.bar([x + w / 2 + 0.01 for x in xs], b, width=w, color=ORANGE, label="Nejaušs sākums, fiksēts k")
    for x, v in zip(xs, b):
        ax.annotate(f"{v:.0f}", (x + w / 2, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7, color=INK2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=0, fontsize=7.5)
    ax.set_ylabel("Sasniedz labāko zināmo, %")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sakuma_strategija.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instances = read("instances.csv")
    runs = read("runs.csv")
    fig_results(instances, runs)
    if any(r["experiment"] == "restart" for r in runs):
        fig_restarts(runs)
    if any(r["experiment"] == "fixed_k" for r in runs):
        fig_start_strategy(instances, runs)
    print("Grafiki saglabāti:", OUT)


if __name__ == "__main__":
    main()
