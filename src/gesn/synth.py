"""Очистка синтетики и сборка обучающих троек (запрос, норма, трудный негатив).

    python -m gesn.synth        # data/train/synth_raw.jsonl → train.jsonl, dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from gesn.retrievers import load_norms, tokenize

TRAIN_DIR = Path("data/train")
FORBIDDEN_RE = re.compile(r"гэсн|\bнорм[аы]?\b|\d{2}-\d{2}-\d{3}", re.IGNORECASE)
ADDON_RE = "добавлять|исключать"


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def clean_queries(
    raw: list[dict], norms: pd.DataFrame, max_copy: float = 0.8, keep_ambiguous: bool = False
) -> list[dict]:
    """Фильтры: длина, запрещённые слова, почти дословная копия названия, неоднозначные дубли.

    keep_ambiguous: запрос, сгенерированный для норм из разных таблиц, не выбрасывается, а остаётся
    парой с каждой из них (multi-positive). Сэмплер NO_DUPLICATES при обучении не кладёт
    одинаковые запросы в один батч, поэтому они не становятся друг другу ложными негативами.
    """
    names = dict(zip(norms["code"], norms["full_name"]))
    tables = dict(zip(norms["code"], norms["table_code"]))
    stats: Counter[str] = Counter()
    pairs = []
    for row in raw:
        code = row["code"]
        name_tokens = set(tokenize(names[code]))
        seen = set()
        for q in row.get("queries") or row.get("items", []):
            q = re.sub(r"\s+", " ", q).strip(" .«»\"'")
            key = q.lower()
            if not 2 <= len(q.split()) <= 25:
                stats["длина"] += 1
            elif FORBIDDEN_RE.search(q):
                stats["запрещённые слова"] += 1
            elif jaccard(set(tokenize(q)), name_tokens) >= max_copy:
                stats["копия названия"] += 1
            elif key in seen:
                stats["дубль внутри нормы"] += 1
            else:
                seen.add(key)
                pairs.append({"query": q, "code": code})

    # один и тот же запрос у норм из разных таблиц — неоднозначен
    tables_by_query: dict[str, set[str]] = {}
    for p in pairs:
        tables_by_query.setdefault(p["query"].lower(), set()).add(tables[p["code"]])
    ambiguous = {q for q, ts in tables_by_query.items() if len(ts) > 1}
    n_ambiguous = sum(p["query"].lower() in ambiguous for p in pairs)
    if keep_ambiguous:
        print(f"Неоднозначных пар оставлено: {n_ambiguous}")
    else:
        stats["неоднозначные"] = n_ambiguous
        pairs = [p for p in pairs if p["query"].lower() not in ambiguous]

    print("Отброшено:", dict(stats))
    return pairs


def build_triplets(pairs: list[dict], norms: pd.DataFrame, seed: int = 0) -> list[dict]:
    """Трудный негатив — соседняя норма той же таблицы, иначе случайная из того же сборника."""
    rng = random.Random(seed)
    text = dict(zip(norms["code"], norms["text"]))
    base = norms[~norms["full_name"].str.contains(ADDON_RE, case=False)]
    by_table = base.groupby("table_code")["code"].apply(list).to_dict()
    by_collection = base.groupby("collection_code")["code"].apply(list).to_dict()

    triplets = []
    for p in pairs:
        code = p["code"]
        siblings = [c for c in by_table.get(code.rsplit("-", 1)[0], []) if c != code]
        negative = rng.choice(siblings or [c for c in by_collection[code[:2]] if c != code])
        triplets.append({"query": p["query"], "code": code, "positive": text[code], "negative": text[negative]})
    return triplets


def split_by_norm(pairs: list[dict], dev_share: float, seed: int) -> tuple[list[dict], list[dict]]:
    """dev — отложенные нормы целиком: их запросы модель при обучении не видит."""
    codes = sorted({p["code"] for p in pairs})
    dev_codes = set(random.Random(seed).sample(codes, int(len(codes) * dev_share)))
    return [p for p in pairs if p["code"] not in dev_codes], [p for p in pairs if p["code"] in dev_codes]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=TRAIN_DIR / "synth_raw.jsonl")
    parser.add_argument("--dev-share", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-ambiguous", action="store_true", help="Не выбрасывать запросы к нормам из разных таблиц")
    parser.add_argument("--out-dir", type=Path, default=TRAIN_DIR)
    args = parser.parse_args()

    norms = load_norms()
    with args.raw.open(encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    print(f"Норм: {len(raw)}, сырых запросов: {sum(len(r.get('queries') or r.get('items', [])) for r in raw)}")

    pairs = clean_queries(raw, norms, keep_ambiguous=args.keep_ambiguous)
    train_pairs, dev_pairs = split_by_norm(pairs, args.dev_share, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", build_triplets(train_pairs, norms, args.seed))
    write_jsonl(args.out_dir / "dev.jsonl", [{"query": p["query"], "relevant": [p["code"]]} for p in dev_pairs])
    print(f"train: {len(train_pairs)} троек; dev: {len(dev_pairs)} запросов")


if __name__ == "__main__":
    main()
