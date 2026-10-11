"""
Ja k eksāmeniem visiem ir savstarpēji kopīgi studenti (klika izmērā k),
tad sesijai vajag vismaz k slotus. Tāpēc  omega(G) <= chi(G).
Ja tabu meklēšana atrod krāsojumu ar omega(G) krāsām, tas ir pierādīti optimāls.
"""

from __future__ import annotations

import time

from problem import ExamProblem


def max_clique(problem: ExamProblem, time_limit: float = 60.0) -> tuple[list[int], bool]:
    n = problem.n
    deg = problem.degrees()
    # pārnumurē virsotnes pēc pakāpes dilstošā secībā (labāka robeža)
    order = sorted(range(n), key=lambda v: -deg[v])
    pos = {v: i for i, v in enumerate(order)}
    adj = [0] * n
    for v in range(n):
        mask = 0
        for u in problem.neighbours[v]:
            mask |= 1 << pos[int(u)]
        adj[pos[v]] = mask

    best: list[int] = []
    deadline = time.perf_counter() + time_limit
    timed_out = False

    def colour_sort(P: int) -> tuple[list[int], list[int]]:
        vertices, bounds = [], []
        colour = 0
        Q = P
        while Q:
            colour += 1
            R = Q
            while R:
                low = R & -R
                v = low.bit_length() - 1
                R &= ~adj[v] & ~low
                Q &= ~low
                vertices.append(v)
                bounds.append(colour)
        return vertices, bounds

    def expand(C: list[int], P: int) -> None:
        nonlocal best, timed_out
        if timed_out:
            return
        if time.perf_counter() > deadline:
            timed_out = True
            return
        vertices, bounds = colour_sort(P)
        for i in range(len(vertices) - 1, -1, -1):
            if len(C) + bounds[i] <= len(best):
                return
            v = vertices[i]
            C.append(v)
            newP = P & adj[v]
            if newP:
                expand(C, newP)
            elif len(C) > len(best):
                best = C.copy()
            C.pop()
            P &= ~(1 << v)

    expand([], (1 << n) - 1)
    clique = sorted(order[v] for v in best)
    return clique, not timed_out
