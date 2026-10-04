"""Сборка тестовой разметки: queries.txt + qrels_spec.txt → qrels.jsonl.

    python -m gesn.qrels
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

TEST_DIR = Path("data/test")
ADDON_RE = re.compile(r"добавлять|исключать", re.IGNORECASE)


def read_queries(path: Path) -> list[dict]:
    """Запросы по порядку; строка «# NN» задаёт сборник-подсказку для следующих запросов."""
    queries, hint = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            hint = line.lstrip("#").strip()
            continue
        queries.append({"qid": len(queries) + 1, "collection_hint": hint, "query": line})
    return queries


def expand(item: str, norms: pd.DataFrame) -> list[str]:
    """15-04-005-{02,04} / 10-05-001-* / 06-01-001-21 → список кодов."""
    table, _, suffix = item.rpartition("-")
    if suffix == "*":
        rows = norms[(norms["table_code"] == table) & ~norms["full_name"].str.contains(ADDON_RE)]
        codes = rows["code"].tolist()
    elif suffix.startswith("{"):
        codes = [f"{table}-{s.strip()}" for s in suffix.strip("{}").split(",")]
    else:
        codes = [item]

    known = set(norms["code"])
    missing = [c for c in codes if c not in known]
    if not codes or missing:
        raise ValueError(f"{item}: нет таких норм {missing or ''}")
    return codes


def read_spec(path: Path, norms: pd.DataFrame) -> dict[int, list[str]]:
    spec = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        qid, _, items = line.partition(":")
        codes = [c for item in items.split() for c in expand(item, norms)]
        spec[int(qid)] = list(dict.fromkeys(codes))
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--norms", type=Path, default=Path("data/norms.parquet"))
    args = parser.parse_args()

    norms = pd.read_parquet(args.norms)
    queries = read_queries(TEST_DIR / "queries.txt")
    spec = read_spec(TEST_DIR / "qrels_spec.txt", norms)

    unlabeled = [q["qid"] for q in queries if q["qid"] not in spec]
    if unlabeled:
        raise ValueError(f"Нет разметки для запросов: {unlabeled}")

    out = TEST_DIR / "qrels.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({**q, "relevant": spec[q["qid"]]}, ensure_ascii=False) + "\n")

    labeled = sum(1 for q in queries if spec[q["qid"]])
    print(f"{len(queries)} запросов, из них {labeled} с ответом, {len(queries) - labeled} вне скоупа → {out}")


if __name__ == "__main__":
    main()
