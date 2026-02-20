from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from .preprocess import tokenize


class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vocab_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class Vectorizer:
    token_to_idx: Dict[str, int]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "Vectorizer":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        vocab = sorted([t for t, c in counts.items() if c >= min_freq])
        return cls(token_to_idx={tok: idx for idx, tok in enumerate(vocab)})

    def encode(self, text: str) -> torch.Tensor:
        vec = torch.zeros(len(self.token_to_idx), dtype=torch.float32)
        for token in tokenize(text):
            idx = self.token_to_idx.get(token)
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
    epochs: int = 80,
    lr: float = 1e-2,
) -> TrainedClassifier:
    texts = [text for text, _ in examples]
    vectorizer = Vectorizer.build(texts)
    model = BagOfWordsClassifier(vocab_size=len(vectorizer.token_to_idx), num_classes=len(labels))

    label_to_idx = {label: i for i, label in enumerate(labels)}
    X = torch.stack([vectorizer.encode(text) for text, _ in examples])
    y = torch.tensor([label_to_idx[label] for _, label in examples], dtype=torch.long)

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
