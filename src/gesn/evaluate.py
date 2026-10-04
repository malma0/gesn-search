"""Оценка поисковиков на размеченном тесте.

Формат разметки (JSONL), одна строка на запрос:
    {"query": "покрасить стены", "relevant": ["15-04-005-01", "15-04-005-02"]}
Запросы с пустым relevant (работы нет в ГЭСН) в метрики не входят.

Метрики считаются на двух уровнях:
- code  — найден ли правильный код нормы;
- table — найдена ли правильная таблица (код без последнего сегмента).
Recall@k здесь = доля запросов, где хотя бы один релевантный ответ попал в top-k
(при одном правильном ответе совпадает с классическим recall).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gesn.retrievers import build_retrievers, load_norms

KS = (1, 5, 10)
MRR_K = 10


def table_of(code: str) -> str:
    return code.rsplit("-", 1)[0]


def to_tables(codes: list[str]) -> list[str]:
    """Ранжированный список кодов → ранжированный список таблиц без повторов."""
    return list(dict.fromkeys(table_of(c) for c in codes))


def first_hit_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return rank
    return None


def per_query_metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    rank = first_hit_rank(ranked, relevant)
    metrics = {f"R@{k}": float(rank is not None and rank <= k) for k in KS}
    metrics[f"MRR@{MRR_K}"] = 1.0 / rank if rank is not None and rank <= MRR_K else 0.0
    return metrics


def bootstrap_ci(values: np.ndarray, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def load_qrels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r.get("relevant")]


def evaluate(retriever, qrels: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """Возвращает {уровень: {метрика: массив значений по запросам}}."""
    depth = max(max(KS), MRR_K)
    levels: dict[str, dict[str, list[float]]] = {"code": {}, "table": {}}
    for row in qrels:
        ranked = retriever.search(row["query"], k=depth * 5)
        relevant = set(row["relevant"])
        per_level = {
            "code": per_query_metrics(ranked[:depth], relevant),
            "table": per_query_metrics(to_tables(ranked)[:depth], {table_of(c) for c in relevant}),
        }
        for level, metrics in per_level.items():
            for name, value in metrics.items():
                levels[level].setdefault(name, []).append(value)
    return {lvl: {m: np.array(v) for m, v in ms.items()} for lvl, ms in levels.items()}


def format_table(results: dict[str, dict], level: str, with_ci: bool) -> str:
    metric_names = list(next(iter(results.values()))[level])
    lines = [
        "| Подход | " + " | ".join(metric_names) + " |",
        "|---" * (len(metric_names) + 1) + "|",
    ]
    for name, levels in results.items():
        cells = []
        for metric in metric_names:
            values = levels[level][metric]
            cell = f"{values.mean():.3f}"
            if with_ci:
                lo, hi = bootstrap_ci(values)
                cell += f" [{lo:.2f}–{hi:.2f}]"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка поисковиков ГЭСН")
    parser.add_argument("--qrels", type=Path, default=Path("data/test/qrels.jsonl"))
    parser.add_argument("--methods", nargs="+", default=["bm25", "e5", "hybrid"])
    parser.add_argument("--no-ci", action="store_true", help="Не считать доверительные интервалы")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    norms = load_norms()
    results = {r.name: evaluate(r, qrels) for r in build_retrievers(args.methods, norms)}

    print(f"Запросов: {len(qrels)}\n")
    for level, title in (("code", "Точный код нормы"), ("table", "Таблица")):
        print(f"### {title}\n")
        print(format_table(results, level, with_ci=not args.no_ci))
        print()


if __name__ == "__main__":
    main()
