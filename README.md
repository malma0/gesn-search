# gesn-search

Поиск кода работы ГЭСН (государственные элементные сметные нормы) по описанию в свободной форме: «покрасить стены в квартире» → `15-04-005-01`.

Сравниваются четыре подхода:
1. BM25 с лемматизацией — baseline.
2. Готовая модель эмбеддингов `multilingual-e5-base`.
3. Та же модель, дообученная на доменных парах «запрос ↔ норма».
4. Гибрид BM25 + dense (Reciprocal Rank Fusion).

> 🚧 В работе. План и этапы — в [PLAN.md](PLAN.md).

## Данные

ФСНБ-2022 в формате открытых данных (XML) с [ФГИС ЦС](https://fgiscs.minstroyrf.ru/frsn/fsnb). В скоупе 6 сборников: 06, 08, 10, 11, 12, 15 — 2940 норм, 625 таблиц.

## Запуск

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
# положить ГЭСН.xml в data/raw/
.venv/Scripts/python -m gesn.parse
.venv/Scripts/python -m pytest
```
