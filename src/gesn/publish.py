"""Публикация дообученной модели и демо-Space на Hugging Face. Запускается на Kaggle.

    HF_TOKEN=... python -m gesn.publish --model-dir models/e5-gesn

Создаёт <user>/e5-gesn (модель) и <user>/gesn-search (Gradio Space с готовыми векторами норм).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from gesn.retrievers import DenseRetriever, load_norms

SPACE_SRC = Path("space")


def space_readme(title: str, sdk_version: str) -> str:
    return f"""---
title: {title}
emoji: 🏗️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: {sdk_version}
app_file: app.py
pinned: false
---

Поиск кода ГЭСН по описанию работы в свободной форме. Код, данные и метрики: https://github.com/malma0/gesn-search
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default="models/e5-gesn")
    parser.add_argument("--model-name", default="e5-gesn")
    parser.add_argument("--space-name", default="gesn-search")
    args = parser.parse_args()

    import gradio
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    model_id, space_id = f"{user}/{args.model_name}", f"{user}/{args.space_name}"

    SentenceTransformer(args.model_dir).push_to_hub(
        model_id, token=token, commit_message="e5-base, дообученная на синтетических парах запрос↔норма ГЭСН"
    )
    print(f"Модель → https://huggingface.co/{model_id}")

    norms = load_norms()
    embeddings = DenseRetriever(norms, model_name=args.model_dir, name="ft").embeddings

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in ["app.py", "requirements.txt"]:
            shutil.copy(SPACE_SRC / f, tmp / f)
        norms[["code", "table_code", "full_name", "unit"]].to_parquet(tmp / "norms.parquet", index=False)
        np.save(tmp / "embeddings.npy", embeddings.astype(np.float32))
        (tmp / "model_id.txt").write_text(model_id)
        (tmp / "README.md").write_text(space_readme("Поиск кода ГЭСН", gradio.__version__), encoding="utf-8")

        api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=str(tmp), repo_id=space_id, repo_type="space", commit_message="Демо поиска ГЭСН")
    print(f"Демо → https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
