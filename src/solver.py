"""
1. Sākuma risinājums ar DSatur -> derīgs saraksts ar k0 slotiem.
2. Kamēr atliek laiks:  izņem vienu krāsu (mazāko klasi), tās virsotnes
   pārvieto uz krāsām ar vismazāk konfliktiem un ar TabuCol meklē derīgu
   krāsojumu ar k-1 krāsām.  Ja izdodas -> k := k-1 un atkārto.
   Ja TabuCol `stall_limit` iterācijas neuzlabo labāko f (stagnācija),
   mēģinājumu sāk no jauna ar citu nejaušu sākuma stāvokli (restarts).
3. Apstājas, ja sasniegta apakšējā robeža (klikas izmērs) - tad
   rezultāts ir pierādīti optimāls - vai beidzies laika limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dsatur import dsatur
from problem import ExamProblem
from tabucol import tabucol


@dataclass
class SolveResult:
    instance: str
    seed: int
    initial_k: int                  # DSatur rezultāts
    best_k: int                     # labākais atrastais slotu skaits
    colors: np.ndarray
    time_to_best: float             # sekundes līdz best_k atrašanai
    elapsed: float                  # kopējais darbības laiks
    iterations: int                 # kopējais TabuCol iterāciju skaits
    reached_lower_bound: bool
    restarts: int = 0               # cik reizes TabuCol restartēts stagnācijas dēļ
    trace: list[tuple[float, int]] = field(default_factory=list)   # (laiks, k)


def remove_colour(problem: ExamProblem, colors: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """No derīga k-krāsojuma izveido (parasti nederīgu) (k-1)-krāsojumu."""
    sizes = np.bincount(colors, minlength=k)
    smallest = np.flatnonzero(sizes == sizes.min())
    c_rm = int(smallest[rng.integers(smallest.size)])

    new = colors.copy()
    moved = np.flatnonzero(new == c_rm)
    new[new > c_rm] -= 1
    new[moved] = -1
    for v in rng.permutation(moved):
        nb_cols = new[problem.neighbours[v]]
        counts = np.bincount(nb_cols[nb_cols >= 0], minlength=k - 1)
        options = np.flatnonzero(counts == counts.min())
        new[v] = options[rng.integers(options.size)]
    return new


def solve(
    problem: ExamProblem,
    time_limit: float = 60.0,
    seed: int = 0,
    lower_bound: int = 1,
    max_iter_per_k: int = 10**9,
    tenure_L: int = 10,
    tenure_alpha: float = 0.6,
    stall_limit: int = 50_000,
) -> SolveResult:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    colors = dsatur(problem)
    k = int(colors.max()) + 1
    initial_k = k
    assert problem.is_valid(colors)
    trace = [(time.perf_counter() - t0, k)]
    time_to_best = trace[0][0]
    total_iter = 0

    restarts = 0
    while k > lower_bound:
        remaining = time_limit - (time.perf_counter() - t0)
        if remaining <= 0:
            break
        # jauns mēģinājums ar k-1 krāsām
        start = remove_colour(problem, colors, k, rng)
        res = tabucol(
            problem, k - 1, start, rng,
            max_iter=max_iter_per_k, time_limit=remaining,
            tenure_L=tenure_L, tenure_alpha=tenure_alpha,
            stall_limit=stall_limit,
        )
        total_iter += res.iterations
        if res.conflicts == 0:
            k -= 1
            colors = res.colors
            time_to_best = time.perf_counter() - t0
            trace.append((time_to_best, k))
        else:
            restarts += 1

    assert problem.is_valid(colors), "atrastais saraksts nav derīgs"
    return SolveResult(
        instance=problem.name,
        seed=seed,
        initial_k=initial_k,
        best_k=k,
        colors=colors,
        time_to_best=time_to_best,
        elapsed=time.perf_counter() - t0,
        iterations=total_iter,
        reached_lower_bound=(k == lower_bound),
        restarts=restarts,
        trace=trace,
    )


def solve_fixed_k(
    problem: ExamProblem,
    k: int,
    time_limit: float = 60.0,
    seed: int = 0,
    stall_limit: int = 50_000,
) -> SolveResult:
    """Tīra TabuCol pārbaude, nejaušs sākums, fiksēts k (bez DSatur).

    Atgriež best_k = k, ja derīgs saraksts atrasts, citādi best_k = -1.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    total_iter = 0
    restarts = 0
    best_colors = None
    while True:
        remaining = time_limit - (time.perf_counter() - t0)
        if remaining <= 0:
            break
        start = rng.integers(k, size=problem.n)
        res = tabucol(problem, k, start, rng, time_limit=remaining, stall_limit=stall_limit,
                      max_iter=10**12)
        total_iter += res.iterations
        if res.conflicts == 0:
            best_colors = res.colors
            break
        restarts += 1
    elapsed = time.perf_counter() - t0
    found = best_colors is not None
    if found:
        assert problem.is_valid(best_colors)
    return SolveResult(
        instance=problem.name,
        seed=seed,
        initial_k=k,
        best_k=k if found else -1,
        colors=best_colors if found else np.zeros(problem.n, dtype=np.int64),
        time_to_best=elapsed if found else float("nan"),
        elapsed=elapsed,
        iterations=total_iter,
        reached_lower_bound=False,
        restarts=restarts,
    )
