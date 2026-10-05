"""Генерация синтетических запросов к нормам открытой LLM. Запускается на Kaggle GPU.

    python -m gesn.generate                                  # v1: запросы → data/train/synth_raw.jsonl
    python -m gesn.generate --glossary data/train/glossary.jsonl --out data/train/synth_raw_gloss.jsonl
    python -m gesn.generate --limit 20                       # быстрая проверка

С --glossary сначала для каждой таблицы норм LLM составляет список бытовых и жаргонных
названий работы (если файла ещё нет), затем эти названия подставляются в промпт к нормам.
Словарь строит сама модель по текстам норм — тестовые запросы в нём не участвуют.

Файлы дописываются построчно: при перезапуске готовые записи пропускаются.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gesn.retrievers import load_norms

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_QUERIES = 5

SYSTEM = """Ты помогаешь собрать данные для поиска по строительным сметным нормам (ГЭСН).
Тебе дают официальное описание работы. Придумай, как эту работу назвали бы живые люди, когда ищут её в поиске.
Отвечай только JSON-массивом строк, без пояснений."""

STYLES = [
    "очень коротко, 2–4 слова, как в поисковой строке",
    "как заказчик-неспециалист, бытовыми словами, можно с контекстом (квартира, дача, гараж, баня)",
    "как прораб, профессиональным сленгом",
    "с конкретным параметром из описания (толщина, материал, тип, размер), но своими словами",
    "небрежно: без предлогов, можно с одной опечаткой или сокращением",
]


def build_glossary_prompt(table_name: str, collection_name: str, examples: list[str]) -> str:
    variants = "\n".join(f"- {e}" for e in examples)
    return f"""Таблица сметных норм: {table_name}
Раздел: {collection_name}
Примеры норм из таблицы:
{variants}

Перечисли 5–10 слов и коротких фраз, которыми заказчики, прорабы и рабочие в разговоре называют эту работу,
её материал, конструкцию или результат: бытовые названия, профессиональный сленг, распространённые сокращения
и торговые названия материалов. Только то, что реально употребляется на стройке и в ремонте.

Ответ: JSON-массив строк."""


def build_prompt(row, glossary: list[str] | None = None) -> str:
    content = f"\nСостав работ: {row.content}" if row.content else ""
    slang = f"\nКак это называют в разговоре: {', '.join(glossary)}" if glossary else ""
    slang_rule = "\n- используй разговорные названия из списка хотя бы в двух запросах, если они подходят;" if glossary else ""
    styles = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(STYLES))
    return f"""Работа: {row.full_name}
Раздел: {row.collection_name}
Единица измерения: {row.unit}{content}{slang}

Напиши {N_QUERIES} разных поисковых запроса для этой работы, по одному в каждом стиле:
{styles}

Правила:
- не копируй официальную формулировку, перефразируй;
- не используй слова «устройство», «норма», «ГЭСН» и коды;
- если работа отличается от похожих параметром (толщина, материал, способ), отрази его хотя бы в двух запросах;{slang_rule}
- запросы на русском, без кавычек внутри.

Ответ: JSON-массив из {N_QUERIES} строк."""


def parse(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def read_jsonl(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r[key]: r for r in rows}


class Generator:
    def __init__(self, model_name: str, batch_size: int, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto").eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def run(self, prompts: dict[str, str], out: Path, key: str) -> None:
        """Генерирует ответы на prompts {ключ: текст} и дописывает {key, items, raw} в out."""
        from tqdm.auto import tqdm

        done = read_jsonl(out, key)
        todo = {k: p for k, p in prompts.items() if k not in done}
        print(f"{out.name}: всего {len(prompts)}, готово {len(done)}, осталось {len(todo)}")
        chat = {
            k: self.tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for k, p in todo.items()
        }
        keys = sorted(chat, key=lambda k: len(chat[k]))  # похожие длины в одном батче — меньше паддинга

        out.parent.mkdir(parents=True, exist_ok=True)
        bad = 0
        with out.open("a", encoding="utf-8") as f:
            for i in tqdm(range(0, len(keys), self.batch_size)):
                batch = keys[i : i + self.batch_size]
                inputs = self.tok([chat[k] for k in batch], return_tensors="pt", padding=True).to(self.model.device)
                with self.torch.no_grad():
                    ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.95,
                        pad_token_id=self.tok.pad_token_id,
                    )
                texts = self.tok.batch_decode(ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                for k, text in zip(batch, texts):
                    items = parse(text)
                    bad += not items
                    f.write(json.dumps({key: k, "items": items, "raw": text}, ensure_ascii=False) + "\n")
                f.flush()
        print(f"Готово → {out}; ответов без JSON: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=Path("data/train/synth_raw.jsonl"))
    parser.add_argument("--glossary", type=Path, help="Словарь разговорных названий по таблицам (создаётся, если нет)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--limit", type=int, help="Только первые N норм (для проверки)")
    args = parser.parse_args()

    norms = load_norms()
    norms = norms[~norms["full_name"].str.contains("добавлять|исключать", case=False)]
    if args.limit:
        norms = norms.head(args.limit)

    gen = Generator(args.model, args.batch_size, args.max_new_tokens)

    glossary: dict[str, list[str]] = {}
    if args.glossary:
        tables = norms.groupby("table_code")
        gen.run(
            {
                t: build_glossary_prompt(g["table_name"].iloc[0], g["collection_name"].iloc[0], g["full_name"].head(4).tolist())
                for t, g in tables
            },
            args.glossary,
            key="table_code",
        )
        glossary = {t: r["items"] for t, r in read_jsonl(args.glossary, "table_code").items()}

    gen.run(
        {r.code: build_prompt(r, glossary.get(r.table_code)) for r in norms.itertuples()},
        args.out,
        key="code",
    )


if __name__ == "__main__":
    main()
