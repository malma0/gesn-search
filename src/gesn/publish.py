"""Публикация дообученной модели и демо на Hugging Face. Запускается на Kaggle.

    HF_TOKEN=... python -m gesn.publish --model-dir models/e5-gesn

1. <user>/e5-gesn — модель sentence-transformers + ONNX (fp32 и int8) для transformers.js.
2. <user>/gesn-search — статический Space: поиск в браузере, векторы норм посчитаны заранее.
   Статические Space бесплатны; Gradio на бесплатном CPU требует подписки PRO.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from gesn.retrievers import DenseRetriever, load_norms

SPACE_SRC = Path("space")


def ensure_onnx_deps() -> None:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)


def export_onnx(model_dir: str, out_dir: Path) -> None:
    """XLM-R энкодер → onnx/model.onnx и onnx/model_quantized.onnx (раскладка, которую ждёт transformers.js)."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoTokenizer

    class Encoder(torch.nn.Module):
        """Только именованные аргументы: позиционные в transformers v5 съезжают на use_cache."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Encoder(AutoModel.from_pretrained(model_dir, attn_implementation="eager")).eval()
    sample = tok(["query: пример"], return_tensors="pt")

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "model.onnx"
    axes = {0: "batch", 1: "sequence"}
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": axes, "attention_mask": axes, "last_hidden_state": axes},
            opset_version=14,
            dynamo=False,
        )
    quantize_dynamic(str(fp32), str(out_dir / "model_quantized.onnx"), weight_type=QuantType.QInt8)


def space_readme(title: str) -> str:
    return f"""---
title: {title}
emoji: 🏗️
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
---

Поиск кода ГЭСН по описанию работы в свободной форме. Модель работает в браузере (transformers.js).
Код, данные и метрики: https://github.com/malma0/gesn-search
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default="models/e5-gesn")
    parser.add_argument("--model-name", default="e5-gesn")
    parser.add_argument("--space-name", default="gesn-search")
    args = parser.parse_args()

    ensure_onnx_deps()
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    model_id, space_id = f"{user}/{args.model_name}", f"{user}/{args.space_name}"

    SentenceTransformer(args.model_dir).push_to_hub(
        model_id,
        token=token,
        exist_ok=True,
        commit_message="e5-base, дообученная на синтетических парах запрос↔норма ГЭСН",
    )
    with tempfile.TemporaryDirectory() as tmp:
        export_onnx(args.model_dir, Path(tmp) / "onnx")
        api.upload_folder(folder_path=tmp, repo_id=model_id, commit_message="ONNX (fp32 и int8) для transformers.js")
    print(f"Модель → https://huggingface.co/{model_id}")

    norms = load_norms()
    embeddings = DenseRetriever(norms, model_name=args.model_dir, name="ft").embeddings.astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(SPACE_SRC / "index.html", tmp / "index.html")
        records = [{"code": r.code, "name": r.full_name, "unit": r.unit} for r in norms.itertuples()]
        (tmp / "norms.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        (tmp / "embeddings.bin").write_bytes(embeddings.tobytes())
        (tmp / "model_id.txt").write_text(model_id)
        (tmp / "README.md").write_text(space_readme("Поиск кода ГЭСН"), encoding="utf-8")

        api.create_repo(space_id, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(folder_path=str(tmp), repo_id=space_id, repo_type="space", commit_message="Демо поиска ГЭСН")
    print(f"Демо → https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
