"""Парсер ФСНБ-2022 (ГЭСН.xml) в плоскую таблицу норм.

Структура XML: Section(Сборник) → Section(Отдел/Раздел/Подраздел)* → Section(Таблица)
→ NameGroup(BeginName) → Work(Code, EndName, MeasureUnit) → Content/Item(Text).
Ресурсы (машины, материалы, трудозатраты) для поиска не нужны и пропускаются.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

DEFAULT_COLLECTIONS = ["06", "08", "10", "11", "12", "15"]

COLUMNS = [
    "code",
    "collection_code",
    "collection_name",
    "section_name",
    "subsection_name",
    "table_code",
    "table_name",
    "begin_name",
    "end_name",
    "full_name",
    "unit",
    "content",
]


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def join_name(begin: str, end: str) -> str:
    """Склеивает общую часть названия из NameGroup с уточнением конкретной нормы."""
    if not end:
        return begin.rstrip(":").strip()
    if not begin:
        return end
    return f"{begin} {end}"


def iter_works(section: ET.Element, path: dict[str, str]):
    """Рекурсивно обходит секции, накапливая названия уровней иерархии."""
    kind = section.get("Type")
    path = {**path, kind: clean(section.get("Name"))}

    if kind == "Таблица":
        for group in section.findall("NameGroup"):
            begin = clean(group.get("BeginName"))
            for work in group.findall("Work"):
                end = clean(work.get("EndName"))
                items = [clean(i.get("Text")) for i in work.findall("Content/Item")]
                yield {
                    "code": work.get("Code"),
                    "collection_code": path["_collection_code"],
                    "collection_name": path.get("Сборник", ""),
                    "section_name": path.get("Раздел", ""),
                    "subsection_name": path.get("Подраздел", ""),
                    "table_code": section.get("Code"),
                    "table_name": path["Таблица"],
                    "begin_name": begin,
                    "end_name": end,
                    "full_name": join_name(begin, end),
                    "unit": clean(work.get("MeasureUnit")),
                    "content": " ".join(i for i in items if i),
                }
        return

    for child in section.findall("Section"):
        yield from iter_works(child, path)


def parse(xml_path: Path, collections: list[str] | None = None) -> pd.DataFrame:
    root = ET.parse(xml_path).getroot()
    category = root.find("ResourcesDirectory/ResourceCategory")
    if category is None:
        raise ValueError(f"{xml_path}: не найден ResourcesDirectory/ResourceCategory")

    rows = []
    for collection in category.findall("Section"):
        code = collection.get("Code")
        if collections and code not in collections:
            continue
        rows.extend(iter_works(collection, {"_collection_code": code}))

    df = pd.DataFrame(rows, columns=COLUMNS)
    if df["code"].duplicated().any():
        dupes = df.loc[df["code"].duplicated(), "code"].tolist()[:5]
        raise ValueError(f"Дубли кодов норм: {dupes}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xml", type=Path, default=Path("data/raw/ГЭСН.xml"))
    parser.add_argument("--out", type=Path, default=Path("data/norms.parquet"))
    parser.add_argument(
        "--collections",
        nargs="*",
        default=DEFAULT_COLLECTIONS,
        help="Коды сборников; пустой список = все сборники",
    )
    args = parser.parse_args()

    df = parse(args.xml, args.collections or None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"{len(df)} норм, {df['table_code'].nunique()} таблиц → {args.out}")
    print(df.groupby(["collection_code", "collection_name"]).size().to_string())


if __name__ == "__main__":
    main()
