from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from torch import Tensor

from data_used_by_llm.schemas import QuerySpec
# from embedding_ranking import  spec
from data_used_by_llm.schemas import QuerySpec

EMB_MODEL = "BAAI/bge-small-en-v1.5"
EMB_CACHE_DIR = Path(".emb_cache")


class Embedder:
    def __init__(self, model_name: str = EMB_MODEL,
                 cache_dir: Path = EMB_CACHE_DIR):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.global_embeddings = None
        self.company_indices = []

    # compute the embedding for the description of a company
    def embed_companies(self, docs: list[str]) -> np.ndarray:
        embeddings = []
        for doc in docs:
            corpus_hash = hashlib.sha256(
                (self.model_name + doc).encode()).hexdigest()[:16]
            # get the file
            cache_file = self.cache_dir / f"company_{corpus_hash}.npy"
            # if it exists, load it, otherwise compute and save
            if cache_file.exists():
                embeddings.append(np.load(cache_file))
            else:
                emb = self.model.encode(
                    doc, normalize_embeddings=True,
                    batch_size=64, show_progress_bar=True)
                np.save(cache_file, emb)
                embeddings.append(emb)
        return np.vstack(embeddings)

    # compute the embedding for the query description
    def embed_query(self, spec: QuerySpec) -> Tensor | Any:
        corpus_hash = hashlib.sha256(
            (self.model_name + "\n".join(spec.ideal_match_description)).encode()).hexdigest()[:16]
        # get the file
        cache_file = self.cache_dir / f"company_{corpus_hash}.npy"
        # if it exists, load it, otherwise compute and save
        if cache_file.exists():
            return np.load(cache_file)
        emb = self.model.encode(
            spec.ideal_match_description, normalize_embeddings=True,
            batch_size=64, show_progress_bar=True)
        np.save(cache_file, emb)
        return emb

    # compute the embedding for the query description
    def embed_query_text(self, text: str) -> Tensor | Any:
        corpus_hash = hashlib.sha256(
            (self.model_name + text).encode()).hexdigest()[:16]
        # get the file
        cache_file = self.cache_dir / f"company_{corpus_hash}.npy"
        # if it exists, load it, otherwise compute and save
        if cache_file.exists():
            return np.load(cache_file)
        emb = self.model.encode(
            spec.ideal_match_description, normalize_embeddings=True,
            batch_size=64, show_progress_bar=True)
        np.save(cache_file, emb)
        return emb
