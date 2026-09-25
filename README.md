# ai-learn-18-rag-chunking-strategies

**Phase D, day 18** of the AI learning track. Before RAG can retrieve anything, long documents must be cut into chunks, and that choice quietly sets both retrieval quality and context cost. This repo implements five chunkers from scratch: **fixed-size**, **overlapping** windows, **sentence packing**, a **recursive** separator splitter and **semantic** (topic-shift) chunking. It runs them on eight procedurally generated ~420-word product manuals with 48 paraphrased questions whose answers sit at known character spans. It then measures **retrieval hit rate**, **answer containment**, **index size** and **chunk-length distributions**, both with and without a contextual title prefix on each chunk.

It follows hybrid search (`ai-learn-16`) and the reranker (`ai-learn-17`), and reuses the from-scratch BM25. NumPy + matplotlib only, no network, seed 42, about 0.3 s on CPU.

## What you'll learn

- How each chunker decides boundaries, and why chunks are tracked as character spans. With spans you can check exactly whether the answer survived the cut.
- The difference between a **hit** (the right passage was retrieved) and **containment** (the whole answer is inside a retrieved chunk). Blind fixed windows produce hits that cut the answer in half.
- Why **context loss** can matter more than clean boundaries. Facts that say "the unit" instead of the product name become unfindable once chunked away from the title. A one-line title prefix fixes most of this.
- The size trade-off: bigger chunks raise containment but send far more words to the generator, and overlap inflates the index.

## Architecture

```mermaid
flowchart LR
  G[data.py: 8 manuals, 7 sections each<br/>facts planted at known char spans] --> C{chunker}
  C --> F[fixed: N-word windows]
  C --> O[overlap: N words, stride 2N/3]
  C --> S[sentence: pack whole sentences to N]
  C --> R["recursive: split on \n\n, \n, '. ', ' ' then merge to N"]
  C --> M[semantic: LSA sentence vectors, break at similarity dips]
  F & O & S & R & M --> P{title prefix?}
  P -->|plain| I[BM25 index over chunks]
  P -->|"'Product: ' + chunk"| I
  Q[48 paraphrased questions] --> I
  I --> K[top-k chunks]
  K --> E[hit@k, contain@k, MRR, context words<br/>chunks, indexed words, length stats]
  E --> OUT[results/]
```

## Layout

| path | purpose |
|---|---|
| `data.py` | seeded manual generator (filler pools + 6 planted facts per product), gold fact/answer spans, paraphrased questions |
| `chunkers.py` | `fixed_size`, `overlapping`, `sentence_pack`, `recursive`, `semantic`, `fit_lsa_vectorizer` (all return char spans) |
| `evaluate.py` | build a chunk index (optionally title-prefixed), BM25 retrieval, span-based hit/containment metrics, index stats |
| `text.py`, `bm25.py` | tokenizer and BM25 (from `ai-learn-16`) |
| `run_smoke.py` / `smoke_plots.py` | full grid + size sweep, SVG plots, `RESULTS.md` |
| `notebooks/chunking_walkthrough.ipynb` | walkthrough: see the boundaries each chunker picks |
| `results/` | committed `RESULTS.md`, `metrics.json`, `JSON.shot`, `*.svg` |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

## Headline results (from `results/metrics.json`, max 60 words per chunk, BM25 top-3)

| strategy | contain@3 plain | contain@3 title-prefixed | hit@3 title-prefixed | context words @3 | chunks | index overhead |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 0.542 | **0.667** | 0.729 | 172 | 59 | 1.00x |
| overlap (20) | 0.479 | 0.604 | 0.688 | 183 | 83 | 1.45x |
| sentence | 0.354 | 0.562 | 0.562 | 162 | 65 | 1.00x |
| recursive | 0.375 | 0.583 | 0.583 | 138 | 77 | 1.00x |
| semantic | 0.271 | 0.583 | 0.583 | **119** | 89 | 1.00x |

- **Context loss dominated.** Without a prefix, structure-aware chunkers scored *worse* than blind fixed windows. Their smaller, topic-pure chunks rarely include the product name, so BM25 cannot tell eight near-identical "battery" chunks apart. Prefixing the title raised contain@3 for every strategy, by +0.12 to +0.31. Semantic gained the most, going from 0.271 to 0.583.
- **Boundary cuts are real but smaller.** Only fixed and overlap show hit > containment (for example fixed 0.729 vs 0.667): the right passage was retrieved but the answer was cut. Sentence, recursive and semantic chunks never split an answer, so hit equals containment.
- **Overlap was not free.** It grew the index by 1.45x and only beat fixed at k=5 (0.750 vs 0.729 prefixed), not at k=3.
- **Size trade-off.** In the prefixed sweep, containment climbs with chunk size. Fixed reaches 1.000 at 150 words, but only by sending 428 words per query, roughly a whole manual. Semantic uses the fewest context words, because its topic breaks cap chunk size.

## Honest caveats

The corpus is synthetic and repetitive by design: the same filler pools and fact templates appear across products, which exaggerates the product-name problem. The retriever is BM25 only. A neural encoder would change absolute numbers, though not the span-based metric definitions. The semantic chunker settings (LSA dim 16, 30th-percentile break) are the middle point of a small 3×3 grid I tried on this same eval set, not the best point. The best plain contain@3 seen in that grid was about 0.29. "Tokens" are whitespace words, not BPE tokens.

## Next

`ai-learn-19` moves from document memory to conversational memory: window buffers, vector memory and rolling summaries for an agent.
