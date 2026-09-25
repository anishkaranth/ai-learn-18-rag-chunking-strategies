"""Smoke run: 5 chunking strategies x {plain, title-prefixed} chunks, plus a chunk-size sweep.
Writes results/metrics.json, results/JSON.shot, results/RESULTS.md and SVG plots."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from chunkers import STRATEGIES, fit_lsa_vectorizer, sentences
from data import build_corpus
from evaluate import build_index, evaluate, index_stats
from smoke_plots import make_plots, write_results_md

SEED = 42
CONFIG = {"max_words": 60, "overlap_words": 20, "semantic": {"lsa_dim": 16, "break_percentile": 30, "min_words": 15},
          "retriever": "BM25 (k1=1.5, b=0.75) over chunks", "ks": [1, 3, 5],
          "sweep_sizes": [20, 40, 60, 100, 150], "token_unit": "whitespace word"}


def strategy_kwargs(name: str, size: int, vec) -> dict:
    if name == "fixed":
        return {"size": size}
    if name == "overlap":
        return {"size": size, "overlap": size // 3}
    if name == "semantic":
        s = CONFIG["semantic"]
        return {"max_words": size, "min_words": min(s["min_words"], size // 2), "pct": s["break_percentile"], "vectorize": vec}
    return {"max_words": size}


def main() -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.perf_counter()
    docs, queries = build_corpus(SEED)
    vec = fit_lsa_vectorizer([d.text[s:e] for d in docs for s, e in sentences(d.text)], CONFIG["semantic"]["lsa_dim"])
    results, stats, per_query = {}, {}, {}
    for prefix in (False, True):
        tag = "title_prefix" if prefix else "plain"
        results[tag], stats[tag] = {}, {}
        for name, fn in STRATEGIES.items():
            chunks = build_index(docs, fn, title_prefix=prefix, **strategy_kwargs(name, CONFIG["max_words"], vec))
            ev = evaluate(docs, queries, chunks, CONFIG["ks"])
            per_query[f"{tag}/{name}"] = ev.pop("per_query")
            results[tag][name] = ev
            stats[tag][name] = index_stats(docs, chunks)
    sweep = {}
    for name in ("fixed", "sentence", "recursive", "semantic"):
        sweep[name] = []
        for size in CONFIG["sweep_sizes"]:
            chunks = build_index(docs, STRATEGIES[name], title_prefix=True, **strategy_kwargs(name, size, vec))
            ev = evaluate(docs, queries, chunks, CONFIG["ks"])
            sweep[name].append({"size": size, "n_chunks": len(chunks), "hit@3": ev["hit@3"], "contain@3": ev["contain@3"],
                                "ctx_tokens@3": ev["ctx_tokens@3"]})
    lengths = {n: [len(c["text"].split()) for c in build_index(docs, STRATEGIES[n], **strategy_kwargs(n, CONFIG["max_words"], vec))]
               for n in STRATEGIES}
    m = {"project": "ai-learn-18-rag-chunking-strategies", "seed": SEED, "config": CONFIG,
         "dataset": {"n_docs": len(docs), "n_queries": len(queries), "corpus_tokens": stats["plain"]["fixed"]["corpus_tokens"],
                     "doc_tokens_mean": float(np.mean([len(d.text.split()) for d in docs]))},
         "results": results, "index": stats, "sweep_title_prefix": sweep, "per_query": per_query}
    m["runtime_s"] = time.perf_counter() - t0
    out = Path(__file__).parent / "results"
    plots = make_plots(out, m, lengths)
    write_results_md(out, m, plots)
    (out / "metrics.json").write_text(json.dumps({k: v for k, v in m.items() if k != "per_query"}, indent=2), encoding="utf-8")
    shot = {"project": m["project"], "seed": SEED, "config": {k: CONFIG[k] for k in ("max_words", "overlap_words", "semantic", "retriever")},
            "dataset": m["dataset"],
            "contain@3": {t: {n: round(r["contain@3"], 4) for n, r in results[t].items()} for t in results},
            "hit@3": {t: {n: round(r["hit@3"], 4) for n, r in results[t].items()} for t in results},
            "n_chunks": {n: s["n_chunks"] for n, s in stats["plain"].items()},
            "overhead_x": {n: round(s["overhead_x"], 4) for n, s in stats["plain"].items()},
            "plots": plots, "runtime_s": round(m["runtime_s"], 3)}
    (out / "JSON.shot").write_text(json.dumps(shot, indent=2), encoding="utf-8")
    return m


if __name__ == "__main__":
    m = main()
    for t, rs in m["results"].items():
        for n, r in rs.items():
            print(f"{t:12s} {n:9s} hit@3={r['hit@3']:.3f} contain@3={r['contain@3']:.3f} ctx@3={r['ctx_tokens@3']:.0f} chunks={m['index'][t][n]['n_chunks']}")
    print(f"runtime {m['runtime_s']:.2f}s")
