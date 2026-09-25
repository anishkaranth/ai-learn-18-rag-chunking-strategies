"""Chunking strategies. Each returns a list of (start, end) character spans into the document,
so we can check exactly which gold fact/answer spans a chunk covers. "Tokens" = whitespace words."""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from text import tokenize

Span = Tuple[int, int]
_WORD = re.compile(r"\S+")
_SENT = re.compile(r"[^\n]+?(?:[.!?](?=\s)|$)", re.M)  # sentences and heading lines, never crossing a newline


def n_words(text: str) -> int:
    return len(text.split())


def _strip(text: str, s: int, e: int) -> Span:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def fixed_size(text: str, size: int = 60) -> List[Span]:
    """Consecutive windows of `size` words, blind to sentences and sections."""
    w = [m.span() for m in _WORD.finditer(text)]
    return [(w[i][0], w[min(i + size, len(w)) - 1][1]) for i in range(0, len(w), size)]


def overlapping(text: str, size: int = 60, overlap: int = 20) -> List[Span]:
    """Sliding windows of `size` words with `overlap` words shared between neighbours."""
    w = [m.span() for m in _WORD.finditer(text)]
    step = size - overlap
    out = []
    for i in range(0, len(w), step):
        out.append((w[i][0], w[min(i + size, len(w)) - 1][1]))
        if i + size >= len(w):
            break
    return out


def sentences(text: str) -> List[Span]:
    return [_strip(text, *m.span()) for m in _SENT.finditer(text) if m.group().strip()]


def _pack(text: str, units: Sequence[Span], max_words: int) -> List[Span]:
    """Greedily merge adjacent units while the merged chunk stays within max_words."""
    out: List[Span] = []
    cur, cur_n = None, 0
    for s, e in units:
        n = n_words(text[s:e])
        if cur is not None and cur_n + n <= max_words:
            cur, cur_n = (cur[0], e), cur_n + n
        else:
            if cur is not None:
                out.append(cur)
            cur, cur_n = (s, e), n
    if cur is not None:
        out.append(cur)
    return out


def sentence_pack(text: str, max_words: int = 60) -> List[Span]:
    """Whole sentences packed up to max_words (a sentence is never cut)."""
    return _pack(text, sentences(text), max_words)


def recursive(text: str, max_words: int = 60, seps: Sequence[str] = ("\n\n", "\n", ". ", " ")) -> List[Span]:
    """LangChain-style recursive splitter: split on the coarsest separator present, recurse into
    pieces that are still too long, then merge small neighbours back up to max_words."""
    def split(s: int, e: int, level: int) -> List[Span]:
        if n_words(text[s:e]) <= max_words or level >= len(seps):
            return [_strip(text, s, e)]
        sep, pieces, i = seps[level], [], s
        while True:
            j = text.find(sep, i, e)
            if j < 0:
                pieces.append((i, e))
                break
            pieces.append((i, j + len(sep)))
            i = j + len(sep)
        out: List[Span] = []
        for ps, pe in pieces:
            if text[ps:pe].strip():
                out.extend(split(ps, pe, level + 1))
        return _pack(text, out, max_words)

    return split(0, len(text), 0)


def semantic(text: str, max_words: int = 60, min_words: int = 15, pct: float = 30.0,
             vectorize: Callable[[List[str]], np.ndarray] | None = None) -> List[Span]:
    """Break between sentences where topical similarity dips.

    Each sentence gets a TF-IDF vector. We compare the mean of the previous 2 sentences against the
    next 2 and break where that cosine is below the doc's `pct` percentile, subject to min/max size."""
    units = sentences(text)
    vecs = (vectorize or _tfidf)([text[s:e] for s, e in units])
    sims = []
    for i in range(1, len(units)):
        a, b = vecs[max(0, i - 2):i].mean(0), vecs[i:i + 2].mean(0)
        sims.append(float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)))
    thr = np.percentile(sims, pct) if sims else 0.0
    out, cur, cur_n = [], units[0], n_words(text[slice(*units[0])])
    for i in range(1, len(units)):
        s, e = units[i]
        n = n_words(text[s:e])
        if (sims[i - 1] <= thr and cur_n >= min_words) or cur_n + n > max_words:
            out.append(cur)
            cur, cur_n = (s, e), n
        else:
            cur, cur_n = (cur[0], e), cur_n + n
    out.append(cur)
    return out


def _tfidf(sents: List[str]) -> np.ndarray:
    toks = [tokenize(s) for s in sents]
    vocab = {t: i for i, t in enumerate(sorted({t for ts in toks for t in ts}))}
    X = np.zeros((len(sents), max(len(vocab), 1)))
    for r, ts in enumerate(toks):
        for t in ts:
            X[r, vocab[t]] += 1
    df = (X > 0).sum(0)
    return X * np.log((1 + len(sents)) / (1 + df)) + X * 1.0


def fit_lsa_vectorizer(corpus_sentences: List[str], dim: int = 16) -> Callable[[List[str]], np.ndarray]:
    """TF-IDF -> truncated SVD fitted on every sentence in the corpus (a stand-in for a sentence encoder)."""
    toks = [tokenize(s) for s in corpus_sentences]
    vocab = {t: i for i, t in enumerate(sorted({t for ts in toks for t in ts}))}

    def counts(ss: List[str]) -> np.ndarray:
        M = np.zeros((len(ss), len(vocab)))
        for r, s in enumerate(ss):
            for t in tokenize(s):
                if t in vocab:
                    M[r, vocab[t]] += 1
        return M

    X = counts(corpus_sentences)
    idf = np.log((1 + len(X)) / (1 + (X > 0).sum(0))) + 1
    _, _, vt = np.linalg.svd(X * idf, full_matrices=False)
    proj = vt[:dim].T
    return lambda ss: (counts(ss) * idf) @ proj


STRATEGIES: Dict[str, Callable[..., List[Span]]] = {
    "fixed": fixed_size, "overlap": overlapping, "sentence": sentence_pack,
    "recursive": recursive, "semantic": semantic,
}
