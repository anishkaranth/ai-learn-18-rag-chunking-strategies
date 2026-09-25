"""Chunk every doc, index chunks with BM25, retrieve for each question, and score against gold spans."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from bm25 import BM25
from chunkers import n_words
from data import Doc, Query


def build_index(docs: List[Doc], chunker, title_prefix: bool = False, **kw):
    chunks = []  # dicts: id, doc_id, span, text (what gets indexed)
    for d in docs:
        for i, (s, e) in enumerate(chunker(d.text, **kw)):
            body = d.text[s:e]
            chunks.append({"id": f"{d.id}#{i}", "doc_id": d.id, "span": (s, e),
                           "text": (f"{d.title}: " + body) if title_prefix else body})
    return chunks


def evaluate(docs: List[Doc], queries: List[Query], chunks: List[dict], ks=(1, 3, 5)) -> Dict[str, float]:
    by_id = {c["id"]: c for c in chunks}
    bm = BM25({c["id"]: c["text"] for c in chunks})
    dmap = {d.id: d for d in docs}
    res = {f"{m}@{k}": 0.0 for k in ks for m in ("hit", "contain", "ctx_tokens")}
    res["mrr_contain@10"] = 0.0
    per_query = []
    for q in queries:
        f = dmap[q.doc_id].facts[q.fact]
        (fs, fe), (as_, ae) = f["fact_span"], f["answer_span"]
        ranked = [by_id[i] for i in bm.search(q.text, k=max(max(ks), 10))]
        hit = [c["doc_id"] == q.doc_id and c["span"][0] < fe and c["span"][1] > fs for c in ranked]
        cont = [c["doc_id"] == q.doc_id and c["span"][0] <= as_ and c["span"][1] >= ae for c in ranked]
        for k in ks:
            res[f"hit@{k}"] += any(hit[:k])
            res[f"contain@{k}"] += any(cont[:k])
            res[f"ctx_tokens@{k}"] += sum(n_words(c["text"]) for c in ranked[:k])
        first = next((i for i, c in enumerate(cont[:10]) if c), None)
        res["mrr_contain@10"] += 0.0 if first is None else 1.0 / (first + 1)
        per_query.append({"id": q.id, "contain@3": bool(any(cont[:3])), "hit@3": bool(any(hit[:3]))})
    n = len(queries)
    out = {k: v / n for k, v in res.items()}
    out["per_query"] = per_query
    return out


def index_stats(docs: List[Doc], chunks: List[dict]) -> Dict[str, float]:
    lens = np.array([n_words(c["text"]) for c in chunks])
    corpus = sum(n_words(d.text) for d in docs)
    return {"n_chunks": int(len(chunks)), "indexed_tokens": int(lens.sum()), "corpus_tokens": int(corpus),
            "overhead_x": float(lens.sum() / corpus), "len_mean": float(lens.mean()), "len_std": float(lens.std()),
            "len_min": int(lens.min()), "len_p50": float(np.percentile(lens, 50)),
            "len_p90": float(np.percentile(lens, 90)), "len_max": int(lens.max())}
