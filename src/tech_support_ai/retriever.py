from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .preprocess import tokenize
from .schema import SupportDocument


@dataclass
class RetrievedChunk:
    doc: SupportDocument
    score: float


class BM25Retriever:
    def __init__(self, documents: Sequence[SupportDocument], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc.content) for doc in self.documents]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_dl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.idf = self._compute_idf()

    def _compute_idf(self) -> Dict[str, float]:
        N = len(self.documents)
        df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        return {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def _score(self, query_tokens: Iterable[str], idx: int) -> float:
        score = 0.0
        doc_freq = self.doc_freqs[idx]
        dl = self.doc_lengths[idx]
        for token in query_tokens:
            tf = doc_freq.get(token, 0)
            if tf == 0:
                continue
            idf = self.idf.get(token, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1e-9))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        query_tokens = tokenize(query)
        scored = [RetrievedChunk(doc=self.documents[i], score=self._score(query_tokens, i)) for i in range(len(self.documents))]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
