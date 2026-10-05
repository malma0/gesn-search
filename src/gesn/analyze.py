"""Разбор по запросам и парная значимость различий между подходами.

    python -m gesn.analyze --methods bm25 e5 ft=models/e5-gesn

Пишет results/per_query.jsonl (ранги и top-5 каждого подхода на каждый запрос)
и results/significance.md (парные сравнения).
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from gesn.evaluate import first_hit_rank, load_qrels, table_of, to_tables
from gesn.retrievers import build_retrievers, load_norms

DEPTH = 50


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Двусторонний точный тест Макнемара для бинарных исходов на одних и тех же запросах."""
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% интервал для среднего различия a − b при ресэмплинге запросов."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def metrics_from_rank(rank: int | None) -> dict[str, float]:
    return {
        "R@1": float(rank == 1),
        "R@10": float(rank is not None and rank <= 10),
        "MRR@10": 1.0 / rank if rank is not None and rank <= 10 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qrels", type=Path, default=Path("data/test/qrels.jsonl"))
    parser.add_argument("--methods", nargs="+", default=["bm25", "e5"])
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    norms = load_norms()
    retrievers = build_retrievers(args.methods, norms)

    rows = []
    values: dict[str, dict[str, list[float]]] = {r.name: {} for r in retrievers}
    for q in qrels:
        relevant = set(q["relevant"])
        row = {"qid": q.get("qid"), "query": q["query"], "n_relevant": len(relevant),
               "relevant_tables": sorted({table_of(c) for c in relevant}), "methods": {}}
        for r in retrievers:
            ranked = r.search(q["query"], k=DEPTH)
            rank_code = first_hit_rank(ranked, relevant)
            rank_table = first_hit_rank(to_tables(ranked), {table_of(c) for c in relevant})
            row["methods"][r.name] = {"rank_code": rank_code, "rank_table": rank_table, "top5": ranked[:5]}
            for level, rank in (("code", rank_code), ("table", rank_table)):
                for metric, v in metrics_from_rank(rank).items():
                    values[r.name].setdefault(f"{metric} {level}", []).append(v)
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        f"Набор: `{args.qrels}`, запросов: {len(qrels)}. Различие = A − B; "
        "интервал — парный бутстреп 95%; p — точный тест Макнемара (только для R@k).\n",
        "| A | B | Метрика | A | B | Различие [95% CI] | p |",
        "|---|---|---|---|---|---|---|",
    ]
    names = [r.name for r in retrievers]
    for a, b in combinations(names[::-1], 2):
        for metric in ("R@1 code", "R@10 code", "R@10 table", "MRR@10 code"):
            va, vb = np.array(values[a][metric]), np.array(values[b][metric])
            lo, hi = paired_bootstrap(va, vb)
            p = mcnemar_exact(va, vb) if metric.startswith("R@") else None
            p_cell = f"{p:.3f}" if p is not None else "—"
            lines.append(
                f"| {a} | {b} | {metric} | {va.mean():.3f} | {vb.mean():.3f} | "
                f"{va.mean() - vb.mean():+.3f} [{lo:+.2f}…{hi:+.2f}] | {p_cell} |"
            )
    report = "\n".join(lines) + "\n"
    (args.out_dir / "significance.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
