from dataclasses import dataclass
from typing import List


@dataclass
class SupportDocument:
    doc_id: str
    title: str
    product_family: str
    content: str
    source: str


@dataclass
class SupportResponse:
    issue_category: str
    confidence: float
    answer: str
    steps: List[str]
    evidence_doc_ids: List[str]
