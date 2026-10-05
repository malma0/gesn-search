"""Поисковики по нормам ГЭСН с общим интерфейсом search(query, k) -> список кодов."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

NORMS_PATH = Path("data/norms.parquet")
CACHE_DIR = Path("data/cache")

STOPWORDS = {
    "в", "во", "на", "по", "под", "над", "с", "со", "из", "и", "или", "а", "для", "до",
    "от", "при", "к", "у", "о", "об", "за", "не", "без", "то", "что", "как", "мой",
    "свой", "это", "хотеть", "надо", "нужно", "сделать", "делать",
}


def doc_text(row) -> str:
    """Текст нормы, по которому ищут все подходы — одинаковый для честного сравнения."""
    parts = [row.collection_name, row.full_name, f"Ед. изм.: {row.unit}", row.content]
    return ". ".join(p.rstrip(".") for p in parts if p)


def load_norms(path: Path = NORMS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["text"] = [doc_text(r) for r in df.itertuples()]
    return df


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[str]: ...


# --- BM25 ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    return _morph().parse(word)[0].normal_form


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[а-яёa-z]+|\d+(?:[.,]\d+)?", text.lower().replace("ё", "е"))
    lemmas = (lemma(w) if w[0].isalpha() else w for w in words)
    return [w for w in lemmas if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]


class BM25Retriever:
    name = "bm25"

    def __init__(self, norms: pd.DataFrame):
        from rank_bm25 import BM25Okapi

        self.codes = norms["code"].tolist()
        self.bm25 = BM25Okapi([tokenize(t) for t in norms["text"]])

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self.codes))
        return self.bm25.get_scores(tokens)

    def search(self, query: str, k: int = 10) -> list[str]:
        scores = self.scores(query)
        top = np.argsort(-scores, kind="stable")[:k]
        return [self.codes[i] for i in top if scores[i] > 0]


# --- Dense --------------------------------------------------------------------

class DenseRetriever:
    """Би-энкодер из sentence-transformers. Векторы норм кэшируются на диск."""

    def __init__(
        self,
        norms: pd.DataFrame,
        model_name: str = "intfloat/multilingual-e5-base",
        name: str = "e5",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.codes = norms["code"].tolist()
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name)

        texts = [passage_prefix + t for t in norms["text"]]
        key = hashlib.sha1((model_name + "\n" + "\n".join(texts)).encode()).hexdigest()[:12]
        cache = CACHE_DIR / f"{name}-{key}.npy"
        if cache.exists():
            self.embeddings = np.load(cache)
        else:
            self.embeddings = self.model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.embeddings)

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        return self.embeddings @ q

    def search(self, query: str, k: int = 10) -> list[str]:
        top = np.argsort(-self.scores(query), kind="stable")[:k]
        return [self.codes[i] for i in top]


# --- Hybrid -------------------------------------------------------------------

class HybridRetriever:
    """Reciprocal Rank Fusion: score = Σ 1 / (rrf_k + rank) по всем поисковикам."""

    def __init__(self, retrievers: list[Retriever], name: str = "hybrid", rrf_k: int = 60, depth: int = 100):
        self.name = name
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def search(self, query: str, k: int = 10) -> list[str]:
        fused: dict[str, float] = {}
        for retriever in self.retrievers:
            for rank, code in enumerate(retriever.search(query, self.depth), start=1):
                fused[code] = fused.get(code, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused, key=fused.get, reverse=True)[:k]


def build_retrievers(names: list[str], norms: pd.DataFrame) -> list[Retriever]:
    """Собирает поисковики по именам.

    bm25, e5            — встроенные;
    ft=models/e5-gesn   — dense-модель из папки или с HF Hub под именем ft;
    bm25+ft             — гибрид RRF из ранее перечисленных; hybrid = bm25+e5.
    """
    built: dict[str, Retriever] = {}

    def get(name: str) -> Retriever:
        if name == "hybrid":
            name = "bm25+e5"
        if name not in built:
            if name == "bm25":
                built[name] = BM25Retriever(norms)
            elif name == "e5":
                built[name] = DenseRetriever(norms)
            elif "=" in name:
                alias, path = name.split("=", 1)
                built[alias] = DenseRetriever(norms, model_name=path, name=alias)
                return built[alias]
            elif "+" in name:
                built[name] = HybridRetriever([get(part) for part in name.split("+")], name=name)
            else:
                raise ValueError(f"Неизвестный поисковик: {name}")
        return built[name]

    return [get(n) for n in names]
