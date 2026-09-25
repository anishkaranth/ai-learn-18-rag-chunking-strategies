"""Matplotlib SVG plots + RESULTS.md writer for the chunking smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ai-learn-18"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
_META = {"Date": None}
_C = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata=_META)
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any], lengths: Dict[str, List[int]]) -> List[str]:
    out.mkdir(exist_ok=True)
    names = []
    strat = list(m["results"]["plain"])
    x = np.arange(len(strat))
    fig, ax = plt.subplots(figsize=(7, 3.3))
    for i, (tag, col) in enumerate((("plain", _C[0]), ("title_prefix", _C[2]))):
        ax.bar(x + (i - 0.5) * 0.38, [m["results"][tag][s]["contain@3"] for s in strat], 0.38, color=col, label=f"contain@3 ({tag})")
        ax.plot(x + (i - 0.5) * 0.38, [m["results"][tag][s]["hit@3"] for s in strat], "_", ms=16, mew=2, color=_C[4],
                label="hit@3" if i else None)
    ax.set_xticks(x, strat)
    ax.set_ylim(0, 1)
    ax.set_title("Answer containment in top-3 chunks (bars) and hit rate (ticks)")
    ax.legend(fontsize=8, loc="upper right")
    names.append(_save(fig, out / "strategy_containment.svg"))

    fig, ax = plt.subplots(figsize=(7, 3.1))
    ax.boxplot([lengths[s] for s in strat], showfliers=False)
    ax.set_xticks(range(1, len(strat) + 1), strat)
    ax.set_ylabel("chunk length (words)")
    ax.set_title(f"Chunk-length distribution (max_words={m['config']['max_words']})")
    names.append(_save(fig, out / "chunk_lengths.svg"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for i, (s, rows) in enumerate(m["sweep_title_prefix"].items()):
        sz = [r["size"] for r in rows]
        a1.plot(sz, [r["contain@3"] for r in rows], "o-", color=_C[(0, 1, 2, 4)[i]], label=s, ms=4)
        a2.plot(sz, [r["ctx_tokens@3"] for r in rows], "o-", color=_C[(0, 1, 2, 4)[i]], label=s, ms=4)
    a1.set_xlabel("max chunk size (words)"); a1.set_ylabel("contain@3"); a1.set_ylim(0, 1)
    a2.set_xlabel("max chunk size (words)"); a2.set_ylabel("context words sent @3")
    a1.legend(fontsize=8)
    a1.set_title("Containment vs size", fontsize=10); a2.set_title("Context cost vs size", fontsize=10)
    names.append(_save(fig, out / "size_sweep.svg"))
    return names


def write_results_md(out: Path, m: Dict[str, Any], plots: List[str]) -> None:
    c, ds = m["config"], m["dataset"]
    L = ["# Results -- ai-learn-18-rag-chunking-strategies", "",
         f"**Seed:** `{m['seed']}` | docs={ds['n_docs']} manuals (~{ds['doc_tokens_mean']:.0f} words each, {ds['corpus_tokens']} total) | "
         f"questions={ds['n_queries']} (paraphrased) | max_words={c['max_words']} | overlap={c['overlap_words']} | retriever: {c['retriever']}", "",
         "`hit@k`: a top-k chunk from the right manual overlaps the gold fact sentence. `contain@k`: a top-k chunk contains the **whole** answer string. "
         "`ctx@3`: words sent to the generator for k=3.", ""]
    for tag in ("plain", "title_prefix"):
        L += [f"## {tag} chunks (real smoke run)", "",
              "| strategy | hit@1 | hit@3 | contain@1 | contain@3 | contain@5 | MRR(contain)@10 | ctx@3 | chunks | indexed words | overhead |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s, r in m["results"][tag].items():
            st = m["index"][tag][s]
            L.append(f"| `{s}` | {r['hit@1']:.3f} | {r['hit@3']:.3f} | {r['contain@1']:.3f} | {r['contain@3']:.3f} | {r['contain@5']:.3f} | "
                     f"{r['mrr_contain@10']:.3f} | {r['ctx_tokens@3']:.0f} | {st['n_chunks']} | {st['indexed_tokens']} | {st['overhead_x']:.2f}x |")
        L.append("")
    L += ["## Chunk-length distribution (plain, words)", "", "| strategy | mean | std | min | p50 | p90 | max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s, st in m["index"]["plain"].items():
        L.append(f"| `{s}` | {st['len_mean']:.1f} | {st['len_std']:.1f} | {st['len_min']} | {st['len_p50']:.0f} | {st['len_p90']:.0f} | {st['len_max']} |")
    L += ["", "## Size sweep (title-prefixed chunks): contain@3 / ctx@3", "", "| strategy | " + " | ".join(str(z) for z in c["sweep_sizes"]) + " |",
          "|---|" + "---:|" * len(c["sweep_sizes"])]
    for s, rows in m["sweep_title_prefix"].items():
        L.append(f"| `{s}` | " + " | ".join(f"{r['contain@3']:.3f} / {r['ctx_tokens@3']:.0f}" for r in rows) + " |")
    L += ["", "## Plots", ""] + [f"![{p}]({p})" for p in plots] + ["", f"Wall time: {m['runtime_s']:.2f}s on CPU.", ""]
    (out / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
