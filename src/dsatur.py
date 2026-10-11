"""
Katrā solī izvēlas vēl nenokrāsoto virsotni ar lielāko piesātinājumu
(dažādo krāsu skaitu kaimiņos); vienādības gadījumā - ar lielāko pakāpi
nenokrāsotajā apakšgrafā. Virsotnei piešķir mazāko brīvo krāsu.
"""

from __future__ import annotations

import numpy as np

from problem import ExamProblem


def dsatur(problem: ExamProblem) -> np.ndarray:
    n = problem.n
    nbrs = problem.neighbours
    colors = np.full(n, -1, dtype=np.int64)
    neighbour_colors: list[set[int]] = [set() for _ in range(n)]
    uncoloured_degree = problem.degrees().copy()
    uncoloured = set(range(n))

    while uncoloured:
        v = max(uncoloured, key=lambda x: (len(neighbour_colors[x]), uncoloured_degree[x], -x))
        used = neighbour_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        uncoloured.remove(v)
        for u in nbrs[v]:
            if colors[u] == -1:
                neighbour_colors[u].add(c)
                uncoloured_degree[u] -= 1
    return colors
