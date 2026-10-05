import pandas as pd

from gesn.generate import parse
from gesn.synth import build_triplets, clean_queries, split_by_norm

NORMS = pd.DataFrame(
    {
        "code": ["11-01-011-01", "11-01-011-02", "11-01-011-03", "15-04-005-01"],
        "table_code": ["11-01-011", "11-01-011", "11-01-011", "15-04-005"],
        "collection_code": ["11", "11", "11", "15"],
        "full_name": [
            "Устройство стяжек: цементных толщиной 20 мм",
            "Устройство стяжек: на каждые 5 мм изменения толщины стяжки добавлять или исключать",
            "Устройство стяжек: бетонных толщиной 20 мм",
            "Окраска водоэмульсионными составами стен",
        ],
    }
)
NORMS["text"] = NORMS["full_name"]


def test_clean_queries_filters():
    raw = [
        {"code": "11-01-011-01", "queries": [
            "залить стяжку в квартире",
            "стяжка",                                   # одно слово
            "Устройство стяжек цементных толщиной 20 мм",  # копия названия
            "норма на стяжку",                          # запрещённое слово
            "залить стяжку в квартире",                 # дубль
            "сделать ремонт",                           # неоднозначный (см. ниже)
        ]},
        {"code": "15-04-005-01", "queries": ["покрасить стены водоэмульсионкой", "сделать ремонт"]},
    ]
    pairs = clean_queries(raw, NORMS)
    assert pairs == [
        {"query": "залить стяжку в квартире", "code": "11-01-011-01"},
        {"query": "покрасить стены водоэмульсионкой", "code": "15-04-005-01"},
    ]


def test_triplet_negative_is_sibling_but_not_addon():
    triplets = build_triplets([{"query": "стяжка цементная", "code": "11-01-011-01"}], NORMS)
    assert triplets[0]["negative"] == "Устройство стяжек: бетонных толщиной 20 мм"


def test_split_by_norm_keeps_norm_queries_together():
    pairs = [{"query": f"q{i}", "code": c} for i, c in enumerate(["a", "a", "b", "b", "c", "c"] * 2)]
    train, dev = split_by_norm(pairs, dev_share=0.34, seed=0)
    assert {p["code"] for p in train}.isdisjoint({p["code"] for p in dev})
    assert len(dev) == 4


def test_parse_llm_answer():
    assert parse('Вот: ["залить стяжку", "стяжка 5 см"]') == ["залить стяжку", "стяжка 5 см"]
    assert parse("без json") == []


def test_clean_queries_keep_ambiguous_and_items_key():
    raw = [
        {"code": "11-01-011-01", "items": ["сделать ремонт пола"]},
        {"code": "15-04-005-01", "items": ["сделать ремонт пола"]},
    ]
    assert clean_queries(raw, NORMS) == []
    assert [p["code"] for p in clean_queries(raw, NORMS, keep_ambiguous=True)] == ["11-01-011-01", "15-04-005-01"]


def test_build_prompt_with_glossary():
    from types import SimpleNamespace

    from gesn.generate import build_prompt

    row = SimpleNamespace(full_name="Устройство стяжек", collection_name="Полы", unit="100 м2", content="")
    assert "Как это называют" not in build_prompt(row)
    prompt = build_prompt(row, ["стяжка", "заливка пола"])
    assert "Как это называют в разговоре: стяжка, заливка пола" in prompt
    assert "используй разговорные названия" in prompt
