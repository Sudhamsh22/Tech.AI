import re
from typing import List

TOKEN_RE = re.compile(r"[a-z0-9_+-]+")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(clean_text(text))
