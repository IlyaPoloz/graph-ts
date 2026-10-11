"""Eksāmenu sesijas plānošana ar tabu meklēšanu - vienas instances risināšana.

Piemēri:
    python main.py data/hec-s-92.exm
    python main.py data/uta-s-92.exm --time 120 --seed 3 --out uta.sol
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from clique import max_clique      # noqa: E402
from problem import load_toronto   # noqa: E402
from solver import solve           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimāla eksāmenu sesija (grafu krāsošana + TabuCol)")
    ap.add_argument("exm", help="ceļš uz .exm failu (.stu failam jābūt blakus)")
    ap.add_argument("--time", type=float, default=60.0, help="laika limits sekundēs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stall", type=int, default=50_000, help="iterācijas bez uzlabojuma līdz restartam")
    ap.add_argument("--out", help="kur saglabāt sarakstu (eksāmens slots)")
    args = ap.parse_args()

    p = load_toronto(args.exm)
    clique, proven = max_clique(p, time_limit=30)
    lb = len(clique)
    print(f"Instance {p.name}: {p.n} eksāmeni, {p.n_students} studenti, "
          f"{p.m} konfliktu pāri (blīvums {p.density:.3f})")
    print(f"Apakšējā robeža (klika): {lb}{'' if proven else ' (nav pierādīta)'}")

    r = solve(p, time_limit=args.time, seed=args.seed, lower_bound=lb, stall_limit=args.stall)
    print(f"DSatur: {r.initial_k} sloti  ->  tabu meklēšana: {r.best_k} sloti "
          f"(atrasts pēc {r.time_to_best:.2f} s, {r.iterations} iterācijas, {r.restarts} restarti)")
    if r.reached_lower_bound:
        print("Rezultāts sakrīt ar apakšējo robežu -> pierādīti optimāls.")
    print("Derīgs saraksts:", p.is_valid(r.colors))

    sizes = np.bincount(r.colors, minlength=r.best_k)
    print("Eksāmenu skaits pa slotiem:", " ".join(f"{s + 1}:{c}" for s, c in enumerate(sizes)))

    if args.out:
        with open(args.out, "w") as fh:
            for exam, slot in zip(p.exam_ids, r.colors):
                fh.write(f"{exam} {slot + 1}\n")
        print("Saraksts saglabāts:", args.out)


if __name__ == "__main__":
    main()
