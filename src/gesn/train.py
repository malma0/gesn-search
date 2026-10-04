"""Дообучение e5 на тройках (запрос, норма, трудный негатив). Запускается на Kaggle GPU.

    python -m gesn.train        # data/train/train.jsonl → models/e5-gesn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="intfloat/multilingual-e5-base")
    parser.add_argument("--train", type=Path, default=Path("data/train/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("models/e5-gesn"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    with args.train.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # префиксы e5 те же, что при поиске (DenseRetriever)
    dataset = Dataset.from_list(
        [
            {"anchor": "query: " + r["query"], "positive": "passage: " + r["positive"], "negative": "passage: " + r["negative"]}
            for r in rows
        ]
    )
    print(f"Троек: {len(dataset)}")

    model = SentenceTransformer(args.base)
    # MNRL: остальные нормы батча — негативы; Cached-версия даёт большой батч на маленькой GPU
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.out) + "-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=True,
        # одна и та же норма дважды в батче дала бы ложный негатив
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=dataset, loss=loss)
    trainer.train()

    model.save(str(args.out))
    print(f"Модель → {args.out}")


if __name__ == "__main__":
    main()
