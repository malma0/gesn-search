"""Демо: поиск кода ГЭСН по описанию работы (Hugging Face Space)."""

import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("MODEL_ID") or Path("model_id.txt").read_text().strip()

norms = pd.read_parquet("norms.parquet")
embeddings = np.load("embeddings.npy")
model = SentenceTransformer(MODEL_ID)

EXAMPLES = [
    "покрасить стены в квартире",
    "залить стяжку 5 см",
    "перекрыть крышу металлочерепицей",
    "перегородка из гипсокартона",
    "положить ламинат",
]


def search(query: str, k: int = 10) -> list[list]:
    if not query.strip():
        return []
    q = model.encode(["query: " + query.strip()], normalize_embeddings=True)[0]
    scores = embeddings @ q
    top = np.argsort(-scores, kind="stable")[: int(k)]
    return [
        [norms.at[i, "code"], norms.at[i, "full_name"], norms.at[i, "unit"], round(float(scores[i]), 3)]
        for i in top
    ]


demo = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(label="Опишите работу своими словами", placeholder="например: покрасить потолок в спальне"),
        gr.Slider(1, 20, value=10, step=1, label="Сколько результатов"),
    ],
    outputs=gr.Dataframe(headers=["Код ГЭСН", "Работа", "Ед. изм.", "Сходство"], wrap=True),
    examples=[[e, 10] for e in EXAMPLES],
    title="Поиск кода ГЭСН по описанию",
    description=(
        "Дообученная модель `multilingual-e5-base` ищет по 2940 нормам ГЭСН-2022 "
        "(сборники 06, 08, 10, 11, 12, 15). Код и метрики: "
        "[github.com/malma0/gesn-search](https://github.com/malma0/gesn-search)."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
