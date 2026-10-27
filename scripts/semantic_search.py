#!/usr/bin/env python3
"""FAISS cosine-similarity and BM25 search over MLS remarks.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


class SemanticSearcher:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        self.model = model
        self.model_name = model_name
        self.index = None
        self.listings = []
        self.dimension = None

    def _embed(self, texts: list[str], batch_size=64) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError("Model embeddings have unexpected shape")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain nonfinite values")
        return np.ascontiguousarray(vectors)

    def build_index(self, remarks_list, listing_ids=None, batch_size=64):
        import faiss
        remarks = [str(t) if t is not None else "" for t in remarks_list]
        if not remarks:
            raise ValueError("Cannot index an empty corpus")
        ids = [str(x) for x in listing_ids] if listing_ids is not None else [str(i) for i in range(len(remarks))]
        if len(ids) != len(remarks) or len(set(ids)) != len(ids):
            raise ValueError("Listing IDs must be unique and match the number of remarks")
        vectors = self._embed(remarks, batch_size=batch_size)
        faiss.normalize_L2(vectors)
        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.listings = [{"id": ident, "remarks": remark} for ident, remark in zip(ids, remarks)]
        return self

    def search(self, query: str, top_k=10):
        if self.index is None:
            raise RuntimeError("Build or load an index before searching")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        import faiss
        query_vector = self._embed([query])
        if query_vector.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension differs from index")
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, min(top_k, len(self.listings)))
        return [
            {"id": self.listings[int(i)]["id"], "remarks": self.listings[int(i)]["remarks"], "score": float(score)}
            for score, i in zip(scores[0], indices[0]) if i >= 0
        ]

    def save(self, folder):
        import faiss
        if self.index is None:
            raise RuntimeError("Cannot save before building an index")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "listings.faiss"))
        (folder / "listings.json").write_text(json.dumps({
            "model_name": self.model_name, "dimension": self.dimension, "listings": self.listings
        }, ensure_ascii=False), encoding="utf-8")

    def load(self, folder):
        import faiss
        folder = Path(folder)
        metadata = json.loads((folder / "listings.json").read_text(encoding="utf-8"))
        if metadata["model_name"] != self.model_name:
            raise ValueError("Index model differs from query embedding model")
        self.index = faiss.read_index(str(folder / "listings.faiss"))
        self.dimension = metadata["dimension"]
        self.listings = metadata["listings"]
        if self.index.d != self.dimension or self.index.ntotal != len(self.listings):
            raise ValueError("Index/metadata inconsistency")
        return self


class BM25Searcher:
    """BM25 baseline over the same listings and IDs as SemanticSearcher."""
    def __init__(self):
        self.bm25 = None
        self.listings = []

    def build_index(self, remarks_list, listing_ids=None):
        from rank_bm25 import BM25Okapi
        remarks = [str(t) if t is not None else "" for t in remarks_list]
        if not remarks:
            raise ValueError("Cannot index an empty corpus")
        ids = [str(x) for x in listing_ids] if listing_ids is not None else [str(i) for i in range(len(remarks))]
        if len(ids) != len(remarks) or len(set(ids)) != len(ids):
            raise ValueError("Listing IDs must be unique and aligned")
        self.listings = [{"id": i, "remarks": t} for i, t in zip(ids, remarks)]
        self.bm25 = BM25Okapi([tokenize(t) for t in remarks])
        return self

    def search(self, query: str, top_k=10):
        if self.bm25 is None:
            raise RuntimeError("Build the BM25 index before searching")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {"id": self.listings[int(i)]["id"], "remarks": self.listings[int(i)]["remarks"], "score": float(scores[i])}
            for i in order
        ]
