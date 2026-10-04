"""Генерация синтетических запросов к нормам открытой LLM. Запускается на Kaggle GPU.

    python -m gesn.generate                    # все нормы → data/train/synth_raw.jsonl
    python -m gesn.generate --limit 20         # быстрая проверка

Файл дописывается построчно: при перезапуске уже готовые нормы пропускаются.
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


def build_prompt(row) -> str:
    content = f"\nСостав работ: {row.content}" if row.content else ""
    styles = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(STYLES))
    return f"""Работа: {row.full_name}
Раздел: {row.collection_name}
Единица измерения: {row.unit}{content}

Напиши {N_QUERIES} разных поисковых запроса для этой работы, по одному в каждом стиле:
{styles}

Правила:
- не копируй официальную формулировку, перефразируй;
- не используй слова «устройство», «норма», «ГЭСН» и коды;
- если работа отличается от похожих параметром (толщина, материал, способ), отрази его хотя бы в двух запросах;
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=Path("data/train/synth_raw.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--limit", type=int, help="Только первые N норм (для проверки)")
    args = parser.parse_args()

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    norms = load_norms()
    norms = norms[~norms["full_name"].str.contains("добавлять|исключать", case=False)]
    if args.limit:
        norms = norms.head(args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            done = {json.loads(line)["code"] for line in f if line.strip()}
    todo = [r for r in norms.itertuples() if r.code not in done]
    print(f"Норм: {len(norms)}, уже готово: {len(done)}, осталось: {len(todo)}")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    prompts = {
        r.code: tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(r)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in todo
    }
    # похожие по длине промпты в одном батче — меньше паддинга
    codes = sorted(prompts, key=lambda c: len(prompts[c]))

    bad = 0
    with args.out.open("a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(codes), args.batch_size)):
            batch = codes[i : i + args.batch_size]
            inputs = tok([prompts[c] for c in batch], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            texts = tok.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            for code, text in zip(batch, texts):
                queries = parse(text)
                bad += not queries
                f.write(json.dumps({"code": code, "queries": queries, "raw": text}, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Готово → {args.out}; ответов без JSON: {bad}")


if __name__ == "__main__":
    main()
