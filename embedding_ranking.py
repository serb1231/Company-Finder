
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi
from pathlib import Path

import numpy as np
import pandas as pd

from EmbedderQueryAndCompany import Embedder
from cache_query import CacheQuery
from enhance_data_companies import sync_and_modify
from hard_filter_database import generate_filtered_subset
from query_understanding import QueryParser, MODEL
from read_questions import load_questions

W_SEMANTIC = 0.7
W_NAICS = 0.3


# get lists of objects or objects and transform them in list of strings
def _as_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


# combine the naics labels into a list of strings
def _naics_labels(row: pd.Series) -> list[str]:
    labels = []
    p = row.get("primary_naics")
    if isinstance(p, dict) and p.get("label"):
        labels.append(str(p["label"]))

    p = row.get("secondary_naics")
    if isinstance(p, dict) and p.get("label"):
        labels.append(str(p["label"]))
    return labels


# transform the whole description of a company into a a document
def company_to_document(row: pd.Series) -> str:

    parts: list[str] = []
    name = row.get("operational_name")
    if isinstance(name, str) and name:
        parts.append(name)
    desc = row.get("description")
    if isinstance(desc, str) and desc:
        parts.append(desc)
    offerings = _as_list(row.get("core_offerings"))
    if offerings:
        parts.append("Offers: " + ", ".join(offerings))
    markets = _as_list(row.get("target_markets"))
    if markets:
        parts.append("Serves: " + ", ".join(markets))
    models = _as_list(row.get("business_model"))
    if models:
        parts.append("Business model: " + ", ".join(models))
    naics = _naics_labels(row)
    if naics:
        parts.append("Industry: " + "; ".join(naics))
    return ". ".join(parts) if parts else "(no information available)"


def rank_companies(df, spec, embedder, w_sem=0.7, w_bm25=0.3, k=60):
    out = df.reset_index(drop=True).copy()
    docs = [company_to_document(r) for _, r in out.iterrows()]

    # semantic part. Do cosine similarity
    q_text = ("Represent this sentence for searching relevant passages: "
              + spec.ideal_match_description)
    sim = embedder.embed_companies(docs) @ embedder.embed_query_text(q_text)
    out["semantic_score"] = sim

    # bm25 do word counting
    bm25 = BM25Okapi([_tokenize(d) for d in docs])
    q_tokens = _tokenize(spec.ideal_match_description)
    for t in spec.key_terms:
        q_tokens += _tokenize(t.term) * max(1, round(t.weight * 3))
    out["bm25_score"] = bm25.get_scores(q_tokens)

    out["rank_sem"]  = out["semantic_score"].rank(ascending=False, method="min")
    out["rank_bm25"] = out["bm25_score"].rank(ascending=False, method="min")
    out["rrf"] = w_sem / (k + out["rank_sem"]) + w_bm25 / (k + out["rank_bm25"])
    return out.sort_values("rrf", ascending=False)


JUDGMENT_MARKERS = re.compile(
    r"\bcould\b|\bsimilar\b|compet\w*|fast.growing|\bcritical\b|\bpotential\b")

def select_final(ranked: pd.DataFrame, query: str,
                 score_col: str = "rrf") -> pd.DataFrame:
    out = ranked.sort_values(score_col, ascending=False).reset_index(drop=True)
    s = out[score_col].to_numpy()
    n = len(s)

    # hard filters already did the work, push everything
    if n <= 25:
        return out

    judgment = bool(JUDGMENT_MARKERS.search(query.lower()))
    min_keep = 15 if judgment else 10
    max_keep = min(n, max(40 if judgment else 30, int(0.25 * n)))

    # flat curve
    if s[0] - s[-1] < 0.015:
        return out.head(max_keep)

    # stop at first real cliff
    gaps = s[:-1] - s[1:]
    med = np.median(gaps) + 1e-12
    cut = None
    for i in range(min_keep, min(n - 1, max_keep)):
        if gaps[i] >= 3.0 * med:
            cut = i + 1
            break
    if cut is None:  # smooth decline, no cliff: keep the above-average half
        z = (s - s.mean()) / (s.std() + 1e-9)
        cut = int(np.clip((z >= 0).sum(), min_keep, max_keep))

    # 4. never split a near-tie at the boundary
    while cut < min(n, max_keep) and s[cut - 1] - s[cut] < 0.002:
        cut += 1

    return out.head(cut)

# words that are not needed for the embeding and the bm25
STOP = {"the","a","an","and","or","of","to","for","in","on","as","with",
        "such","its","is","are","that","this","by","from"}

# rules for breaking down sentences, and transforming them in single words
# but in pairs of words as well
def _tokenize(text: str, bigrams: bool = True) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOP and len(t) > 1]
    if bigrams:
        toks += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        toks += [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        toks += [f"{a}-{b}" for a, b in zip(toks, toks[1:])]
    return toks

def select_for_judge(ranked, complexity):
    N = {"structured": 25, "hybrid": 40, "judgment": 60}[complexity.value]
    pool = ranked[(ranked.rank_sem <= N) | (ranked.rank_bm25 <= N)]
    return pool.sort_values("rrf", ascending=False)


if __name__ == "__main__":
    cache = CacheQuery(Path(".query_cache"), MODEL)
    parser = QueryParser(MODEL, cache)

    # Updated to match your path suggestion
    SOURCE_ENHANCE_DATA = 'data/companies.jsonl'
    DEST_ENHANCE_DATA = '.tmp/companies_enhanced.jsonl'

    # run the synchronizer
    sync_and_modify(SOURCE_ENHANCE_DATA, DEST_ENHANCE_DATA)

    embedder = Embedder()
    for i, item in enumerate(load_questions()):
        i = i + 1
        number, query = item["number"], item["question"]
        spec = parser.parse(query)
        print(f"\n=== {number}. {query}")
        # print(spec.model_dump_json(indent=2))

        spec = parser.parse(query)
        print(f"\nQuery: {query}")
        print(f"Ideal match: {spec.ideal_match_description}")
        # print(f"Complexity: {spec.complexity.value}")

        SOURCE_DB = ".tmp/companies_enhanced.jsonl"

        tmp_output_file = f".tmp/tmp_query_{i}_filtered.jsonl"

        result = generate_filtered_subset(source_file=SOURCE_DB,
                dest_file=tmp_output_file,
                query_spec=spec)

        if result is None:
            print(f"Failed to generate filtered subset for question {number}.")
            continue

        survivors, filtered = result
        # transform filtered into pd dataframe
        filtered = pd.DataFrame(filtered)

        ranked = rank_companies(filtered, spec, embedder, w_sem=W_SEMANTIC, w_bm25=W_NAICS, k=60)
        # save each ranked in their own file
        tmp_output_file = f".tmp/tmp_query_{i}_filtered_rank.jsonl"
        with open(tmp_output_file, 'w', encoding='utf-8') as outfile:
            for _, row in ranked.iterrows():
                outfile.write(row.to_json() + '\n')
        print(ranked[["operational_name", "semantic_score", "bm25_score", "rrf"]].head(10))

        select_final(ranked, query, score_col="rrf").to_csv(f".tmp/tmp_query_{i}_final_selection.csv", index=False)