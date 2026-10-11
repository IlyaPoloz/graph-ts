"""Eksāmenu sesijas plānošanas problēma kā grafu krāsošanas problēma.

Ievaddati ir Toronto (Carter et al., 1996) formātā:
  * <nosaukums>.exm  - katrā rindā: eksāmena_id  studentu_skaits
  * <nosaukums>.stu  - katrā rindā: studenta_id  eksāmena_id

No tiem tiek izveidots konfliktu grafs G = (V, E):
  * virsotne  = eksāmens,
  * šķautne   = divi eksāmeni, kurus kārto vismaz viens kopīgs students,
  * krāsa     = laika slots, kurā notiek eksāmens.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ExamProblem:
    name: str
    exam_ids: list[str]                 # eksāmenu identifikatori failā
    n_students: int
    n_enrolments: int
    edges: np.ndarray                   # (m, 2) virsotņu pāri, u < v
    weights: np.ndarray                 # (m,) kopīgo studentu skaits uz šķautnes
    neighbours: list[np.ndarray] = field(repr=False)

    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return len(self.exam_ids)

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def density(self) -> float:
        return 2 * self.m / (self.n * (self.n - 1))

    def degrees(self) -> np.ndarray:
        return np.array([len(nb) for nb in self.neighbours])

    # ------------------------------------------------------------------ #
    def conflicts(self, colors: np.ndarray) -> int:
        """Konfliktējošo šķautņu skaits (abiem galiem viena krāsa)."""
        colors = np.asarray(colors)
        return int(np.sum(colors[self.edges[:, 0]] == colors[self.edges[:, 1]]))

    def is_valid(self, colors: np.ndarray) -> bool:
        return self.conflicts(colors) == 0

    def students_in_conflict(self, colors: np.ndarray) -> int:
        """Cik studentu-pāru 'sadursmes' rada nederīgs saraksts."""
        colors = np.asarray(colors)
        same = colors[self.edges[:, 0]] == colors[self.edges[:, 1]]
        return int(self.weights[same].sum())


def load_toronto(exm_path: str | Path, stu_path: str | Path | None = None) -> ExamProblem:
    """Nolasa Toronto formāta instanci un izveido konfliktu grafu."""
    exm_path = Path(exm_path)
    if stu_path is None:
        stu_path = exm_path.with_suffix(".stu")
    stu_path = Path(stu_path)

    exam_ids: list[str] = []
    with open(exm_path) as fh:
        for line in fh:
            parts = line.split()
            if parts:
                exam_ids.append(parts[0])
    index = {e: i for i, e in enumerate(exam_ids)}

    by_student: dict[str, list[int]] = defaultdict(list)
    n_enrol = 0
    with open(stu_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            by_student[parts[0]].append(index[parts[1]])
            n_enrol += 1

    pair_weight: dict[tuple[int, int], int] = defaultdict(int)
    for exams in by_student.values():
        exams = sorted(set(exams))
        for a in range(len(exams)):
            for b in range(a + 1, len(exams)):
                pair_weight[(exams[a], exams[b])] += 1

    n = len(exam_ids)
    if pair_weight:
        edges = np.array(sorted(pair_weight), dtype=np.int64)
        weights = np.array([pair_weight[tuple(e)] for e in edges], dtype=np.int64)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        weights = np.zeros(0, dtype=np.int64)

    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    neighbours = [np.array(sorted(a), dtype=np.int64) for a in adj]

    return ExamProblem(
        name=exm_path.stem,
        exam_ids=exam_ids,
        n_students=len(by_student),
        n_enrolments=n_enrol,
        edges=edges,
        weights=weights,
        neighbours=neighbours,
    )


def from_edges(name: str, n: int, edge_list: list[tuple[int, int]]) -> ExamProblem:
    """Neliela grafa izveide testiem (bez studentu datiem)."""
    edges = np.array(sorted({(min(u, v), max(u, v)) for u, v in edge_list}), dtype=np.int64)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return ExamProblem(
        name=name,
        exam_ids=[str(i) for i in range(n)],
        n_students=0,
        n_enrolments=0,
        edges=edges,
        weights=np.ones(len(edges), dtype=np.int64),
        neighbours=[np.array(sorted(a), dtype=np.int64) for a in adj],
    )
