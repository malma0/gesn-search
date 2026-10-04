"""Поиск по нормам ГЭСН: выдача нескольких подходов рядом.

    python -m gesn.search "покрасить стены"            # один запрос
    python -m gesn.search                              # интерактивный режим
    python -m gesn.search "стяжка" --methods bm25 -k 5
"""

from __future__ import annotations

import argparse

from gesn.retrievers import build_retrievers, load_norms


def show(query: str, retrievers, names: dict[str, str], units: dict[str, str], k: int) -> None:
    print(f"\n=== {query}")
    for retriever in retrievers:
        print(f"\n--- {retriever.name}")
        for rank, code in enumerate(retriever.search(query, k), start=1):
            print(f"{rank:>3}. {code:<14} {names[code][:110]} [{units[code]}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Поиск по нормам ГЭСН")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--methods", nargs="+", default=["bm25", "e5", "hybrid"])
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    norms = load_norms()
    names = dict(zip(norms["code"], norms["full_name"]))
    units = dict(zip(norms["code"], norms["unit"]))
    retrievers = build_retrievers(args.methods, norms)

    if args.query:
        show(args.query, retrievers, names, units, args.k)
        return
    while True:
        try:
            query = input("\nЗапрос (пусто — выход): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        show(query, retrievers, names, units, args.k)


if __name__ == "__main__":
    main()
