"""Testēšana: visas instances, vairāki palaidieni.

Piemērs:
    python run_experiments.py --time 60 --seeds 10 --jobs 2
Rezultāti tiek saglabāti mapē results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from clique import max_clique          # noqa: E402
from problem import load_toronto       # noqa: E402
from solver import solve, solve_fixed_k   # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Labākais zināmais minimālais slotu skaits (Qu et al., 2009 apskats;
# citēts Qu, Burke & McCollum, EJOR 2009, 9. tabula).
BEST_KNOWN = {
    "hec-s-92": 17, "sta-f-83": 13, "yor-f-83": 19, "ute-s-92": 10,
    "ear-f-83": 22, "tre-s-92": 20, "kfu-s-93": 19, "uta-s-92": 30,
}
INSTANCES = sorted(BEST_KNOWN, key=lambda n: load_toronto(DATA / f"{n}.exm").n)

_cache: dict = {}


def get_problem(name: str):
    if name not in _cache:
        p = load_toronto(DATA / f"{name}.exm")
        lb = len(max_clique(p, time_limit=60)[0])
        _cache[name] = (p, lb)
    return _cache[name]


def run_one(task: dict) -> dict:
    p, lb = get_problem(task["instance"])
    if task["experiment"] == "fixed_k":
        r = solve_fixed_k(p, BEST_KNOWN[p.name], time_limit=task["time"], seed=task["seed"],
                          stall_limit=task["stall_limit"])
    else:
        r = solve(
            p, time_limit=task["time"], seed=task["seed"], lower_bound=lb,
            stall_limit=task["stall_limit"],
        )
    return {
        "experiment": task["experiment"],
        "config": task["config"],
        "instance": p.name,
        "seed": r.seed,
        "initial_k": r.initial_k,
        "best_k": r.best_k,
        "lower_bound": lb,
        "best_known": BEST_KNOWN[p.name],
        "time_to_best": round(r.time_to_best, 3),
        "elapsed": round(r.elapsed, 3),
        "iterations": r.iterations,
        "restarts": r.restarts,
        "valid": p.is_valid(r.colors),
        "trace": [(round(t, 3), k) for t, k in r.trace],
        "colors": r.colors.tolist(),
    }


def write_instances() -> None:
    with open(RESULTS / "instances.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "exams", "students", "enrolments", "conflict_edges",
                    "density", "clique_lb", "dsatur_k", "best_known"])
        from dsatur import dsatur
        for name in INSTANCES:
            p, lb = get_problem(name)
            w.writerow([name, p.n, p.n_students, p.n_enrolments, p.m,
                        f"{p.density:.3f}", lb, int(dsatur(p).max()) + 1, BEST_KNOWN[name]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=60.0, help="laika limits vienam palaidienam, s")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--fixed-seeds", type=int, default=5, help="palaidieni fixed_k eksperimentam")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--experiment", choices=["main", "restart", "fixed_k", "all"], default="all")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "solutions").mkdir(exist_ok=True)
    write_instances()

    tasks = []
    if args.experiment in ("main", "all"):
        tasks += [dict(experiment="main", config="restart_50k", instance=n, seed=s,
                       time=args.time, stall_limit=50_000)
                  for n in INSTANCES for s in range(1, args.seeds + 1)]
    if args.experiment in ("restart", "all"):
        tasks += [dict(experiment="restart", config="no_restart", instance="uta-s-92", seed=s,
                       time=args.time, stall_limit=10**12)
                  for s in range(1, args.seeds + 1)]
    if args.experiment in ("fixed_k", "all"):
        # tīra TabuCol: nejaušs sākums, k = labākais zināmais (bez DSatur)
        tasks += [dict(experiment="fixed_k", config="random_start", instance=n, seed=s,
                       time=args.time, stall_limit=50_000)
                  for n in INSTANCES for s in range(1, args.fixed_seeds + 1)]

    rows = []
    t0 = time.perf_counter()
    with Pool(args.jobs) as pool:
        for row in pool.imap_unordered(run_one, tasks):
            rows.append(row)
            print(f"[{time.perf_counter() - t0:7.1f}s] {row['experiment']:8s} {row['config']:12s} "
                  f"{row['instance']:9s} seed={row['seed']:2d} k0={row['initial_k']} "
                  f"k={row['best_k']} (best known {row['best_known']}, LB {row['lower_bound']}) "
                  f"t_best={row['time_to_best']}s valid={row['valid']}", flush=True)

    rows.sort(key=lambda r: (r["experiment"], INSTANCES.index(r["instance"]), r["seed"]))
    fields = [k for k in rows[0] if k not in ("trace", "colors")]
    with open(RESULTS / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(RESULTS / "traces.json", "w") as fh:
        json.dump([{k: r[k] for k in ("experiment", "config", "instance", "seed", "trace")}
                   for r in rows], fh)

    # labākais risinājums katrai instancei -> eksāmenu saraksts (eksāmens slots)
    for name in INSTANCES:
        cand = [r for r in rows if r["instance"] == name and r["experiment"] == "main"]
        if not cand:
            continue
        best = min(cand, key=lambda r: (r["best_k"], r["time_to_best"]))
        p, _ = get_problem(name)
        with open(RESULTS / "solutions" / f"{name}.sol", "w") as fh:
            for exam, slot in zip(p.exam_ids, best["colors"]):
                fh.write(f"{exam} {slot + 1}\n")


if __name__ == "__main__":
    main()
