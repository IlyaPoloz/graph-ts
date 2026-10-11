"""
Domēns:      katrai virsotnei (eksāmenam) piešķirta viena no k krāsām (slotiem).
Novērtēšana: f(c) = konfliktējošo šķautņu skaits; mērķis f = 0.
Gājiens:     (v, c_jaunā) - konfliktējošai virsotnei v nomaina krāsu.
Algoritms:   katrā iterācijā izvēlas LABĀKO atļauto gājienu (arī ja tas pasliktina f),
             pēc gājiena aizliedz atgriezt v iepriekšējā krāsā uz tl iterācijām.
             Aspirācija: tabu gājienu atļauj, ja tas dod labāku f nekā līdz šim labākais.

Ātrai novērtēšanai glabā tabulu gamma[v, c] = cik v kaimiņiem ir krāsa c.
Tad gājiena (v -> c) ietekme ir  delta = gamma[v, c] - gamma[v, color[v]]
un to aprēķina O(1); pēc gājiena atjauno tikai v kaimiņu rindas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from problem import ExamProblem

BIG = 1 << 30


@dataclass
class TabuResult:
    colors: np.ndarray          # labākais atrastais krāsojums
    conflicts: int              # tā f vērtība
    iterations: int
    elapsed: float
    history: list[tuple[int, int]] = field(default_factory=list)   # (iterācija, f)


def build_gamma(problem: ExamProblem, colors: np.ndarray, k: int) -> np.ndarray:
    gamma = np.zeros((problem.n, k), dtype=np.int64)
    if problem.m:
        u, v = problem.edges[:, 0], problem.edges[:, 1]
        np.add.at(gamma, (u, colors[v]), 1)
        np.add.at(gamma, (v, colors[u]), 1)
    return gamma


def tabucol(
    problem: ExamProblem,
    k: int,
    colors: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 1_000_000,
    time_limit: float = float("inf"),
    tenure_L: int = 10,
    tenure_alpha: float = 0.6,
    history_every: int = 0,
    stall_limit: int = 10**12,
) -> TabuResult:
    """Meklē krāsojumu ar k krāsām un f = 0, sākot no colors.

    Apstājas, ja f = 0, sasniegts max_iter / time_limit vai arī
    stall_limit iterācijas pēc kārtas nav uzlabots labākais f.
    """
    t0 = time.perf_counter()
    n = problem.n
    nbrs = problem.neighbours
    ar = np.arange(n)

    colors = np.asarray(colors, dtype=np.int64).copy()
    gamma = build_gamma(problem, colors, k)
    tabu_until = np.zeros((n, k), dtype=np.int64)

    f = int(gamma[ar, colors].sum() // 2)
    best_f = f
    best_colors = colors.copy()
    history: list[tuple[int, int]] = [(0, f)] if history_every else []

    it = 0
    last_improvement = 0
    while it < max_iter and f > 0:
        if (it & 255) == 0 and time.perf_counter() - t0 > time_limit:
            break
        if it - last_improvement > stall_limit:
            break

        own = gamma[ar, colors]                      # konfliktu skaits katrai virsotnei
        conf = np.flatnonzero(own > 0)               # konfliktējošās virsotnes
        m = conf.size
        delta = gamma[conf] - own[conf, None]        # (m, k) gājienu ietekme uz f
        delta[np.arange(m), colors[conf]] = BIG      # pašreizējā krāsa nav gājiens

        allowed = (tabu_until[conf] <= it) | (f + delta < best_f)   # tabu + aspirācija
        masked = np.where(allowed, delta, BIG)
        d_min = int(masked.min())
        if d_min >= BIG:                             # visi gājieni tabu -> ņem labāko jebkuru
            masked = delta
            d_min = int(masked.min())

        candidates = np.flatnonzero(masked.ravel() == d_min)
        pick = int(candidates[rng.integers(candidates.size)])
        i, new_c = divmod(pick, k)
        v = int(conf[i])
        old_c = int(colors[v])

        # veic gājienu un atjauno gamma tikai kaimiņiem
        colors[v] = new_c
        nb = nbrs[v]
        gamma[nb, old_c] -= 1
        gamma[nb, new_c] += 1
        f += d_min

        # dinamisks tabu ilgums: tl = rand(0..L-1) + alpha * |konfliktējošās virsotnes|
        tabu_until[v, old_c] = it + int(rng.integers(tenure_L)) + int(tenure_alpha * m)

        it += 1
        if f < best_f:
            best_f = f
            best_colors = colors.copy()
            last_improvement = it
        if history_every and it % history_every == 0:
            history.append((it, f))

    if history_every and (not history or history[-1][0] != it):
        history.append((it, f))
    return TabuResult(best_colors, best_f, it, time.perf_counter() - t0, history)
