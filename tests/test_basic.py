"""Pamata testi (palaist: python -m pytest tests  vai  python tests/test_basic.py)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clique import max_clique            # noqa: E402
from dsatur import dsatur                # noqa: E402
from problem import from_edges, load_toronto   # noqa: E402
from solver import solve                 # noqa: E402
from tabucol import build_gamma, tabucol # noqa: E402

# Grafs no apraksta piemēra: 6 eksāmeni, hromatiskais skaitlis 3
EXAMPLE = from_edges("piemers", 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (3, 5)])


def test_example_graph_needs_three_slots():
    r = solve(EXAMPLE, time_limit=5, seed=1, lower_bound=len(max_clique(EXAMPLE)[0]))
    assert r.best_k == 3
    assert EXAMPLE.is_valid(r.colors)


def test_bipartite_cycle_two_colours():
    g = from_edges("c8", 8, [(i, (i + 1) % 8) for i in range(8)])
    r = solve(g, time_limit=5, seed=0, lower_bound=2)
    assert r.best_k == 2


def test_complete_graph():
    n = 7
    g = from_edges("k7", n, [(i, j) for i in range(n) for j in range(i + 1, n)])
    assert len(max_clique(g)[0]) == n
    assert dsatur(g).max() + 1 == n


def test_incremental_gamma_matches_full_recount():
    """gamma tabulai pēc tabu meklēšanas jāsakrīt ar pilnu pārrēķinu."""
    rng = np.random.default_rng(3)
    n = 60
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.2]
    g = from_edges("rnd", n, edges)
    k = 6
    start = rng.integers(k, size=n)
    res = tabucol(g, k, start, rng, max_iter=2000)
    assert res.conflicts == g.conflicts(res.colors)
    gamma = build_gamma(g, res.colors, k)
    assert gamma[np.arange(n), res.colors].sum() // 2 == res.conflicts


def test_toronto_instance_loads_and_solves():
    p = load_toronto(ROOT / "data" / "hec-s-92.exm")
    assert (p.n, p.n_students, p.n_enrolments) == (81, 2823, 10632)
    r = solve(p, time_limit=20, seed=1, lower_bound=17)
    assert p.is_valid(r.colors)
    assert r.best_k <= 18


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
