from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import random
import torch
from torch import nn

from .preprocess import tokenize


class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class Vectorizer:
    token_to_idx: Dict[str, int]

    STOPWORDS = {
        "hello", "hi", "dear", "team", "please", "thanks", "thank",
        "regards", "best", "kind", "good", "day", "issue", "problem"
    }

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2) -> "Vectorizer":
        counts: Dict[str, int] = {}

        for text in texts:
            tokens = [t for t in tokenize(text) if t not in cls.STOPWORDS]

            bigrams = [
                tokens[i] + "_" + tokens[i + 1]
                for i in range(len(tokens) - 1)
            ]

            for tok in tokens + bigrams:
                counts[tok] = counts.get(tok, 0) + 1

        vocab = sorted([t for t, c in counts.items() if c >= min_freq])
        return cls(token_to_idx={tok: idx for idx, tok in enumerate(vocab)})

    def encode(self, text: str) -> torch.Tensor:
        vec = torch.zeros(len(self.token_to_idx), dtype=torch.float32)

        tokens = [t for t in tokenize(text) if t not in self.STOPWORDS]

        bigrams = [
            tokens[i] + "_" + tokens[i + 1]
            for i in range(len(tokens) - 1)
        ]

        for tok in tokens + bigrams:
            idx = self.token_to_idx.get(tok)
            if idx is not None:
                vec[idx] += 1.0

        return vec


@dataclass
class TrainedClassifier:
    model: BagOfWordsClassifier
    vectorizer: Vectorizer
    labels: Sequence[str]

    def predict(self, text: str) -> Tuple[str, float]:
        self.model.eval()
        x = self.vectorizer.encode(text).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        idx = int(torch.argmax(probs).item())
        return self.labels[idx], float(probs[idx].item())


def train_classifier(
    examples: Sequence[Tuple[str, str]],
    labels: Sequence[str],
    epochs: int = 5,
    lr: float = 1e-2,
) -> TrainedClassifier:

    from collections import defaultdict

    grouped = defaultdict(list)
    for text, label in examples:
        grouped[label].append((text, label))

    min_size = min(len(grouped[l]) for l in labels)

    balanced: list[Tuple[str, str]] = []
    for l in labels:
        balanced.extend(random.sample(grouped[l], min_size))

    random.shuffle(balanced)

    texts = [text for text, _ in balanced]
    vectorizer = Vectorizer.build(texts)

    model = BagOfWordsClassifier(
        vocab_size=len(vectorizer.token_to_idx),
        num_classes=len(labels),
    )

    label_to_idx = {label: i for i, label in enumerate(labels)}

    X = torch.stack([vectorizer.encode(text) for text, _ in balanced])
    y = torch.tensor([label_to_idx[label] for _, label in balanced], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    return TrainedClassifier(model=model, vectorizer=vectorizer, labels=labels)