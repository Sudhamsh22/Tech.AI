from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .classifier import TrainedClassifier, train_classifier
from .datasets import ISSUE_LABELS, SEED_TRAINING_EXAMPLES, sample_knowledge_base
from .retriever import BM25Retriever
from .schema import SupportResponse


@dataclass
class TechSupportAssistantModel:
    classifier: TrainedClassifier
    retriever: BM25Retriever

    @classmethod
    def build_default(cls) -> "TechSupportAssistantModel":
        classifier = train_classifier(SEED_TRAINING_EXAMPLES, ISSUE_LABELS)
        retriever = BM25Retriever(sample_knowledge_base())
        return cls(classifier=classifier, retriever=retriever)

    def answer(self, query: str, top_k: int = 2) -> SupportResponse:
        label, confidence = self.classifier.predict(query)
        chunks = self.retriever.retrieve(query, top_k=top_k)

        steps = [
            "Confirm affected laptop model, OS version, and recent changes.",
            "Run guided checks from the retrieved runbook sections.",
            "If unresolved, escalate with logs/diagnostics attached to ticket.",
        ]

        if chunks:
            steps.insert(1, f"Primary reference: {chunks[0].doc.title} ({chunks[0].doc.doc_id}).")

        evidence_ids = [chunk.doc.doc_id for chunk in chunks]
        evidence_text = " ".join(chunk.doc.content for chunk in chunks)

        answer = (
            f"Detected category: {label} (confidence={confidence:.2f}). "
            f"Based on retrieved documentation: {evidence_text}"
        )

        return SupportResponse(
            issue_category=label,
            confidence=confidence,
            answer=answer,
            steps=steps,
            evidence_doc_ids=evidence_ids,
        )
