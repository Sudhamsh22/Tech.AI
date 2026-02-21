import re
from typing import List

TOKEN_RE = re.compile(r"[a-z0-9_+-]+")

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", 
    "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", 
    "its", "itself", "me", "my", "myself", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", 
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", 
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", 
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    "hello", "hi", "dear", "team", "please", "thanks", "thank", "regards", "best", "kind", "good", "day"
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(clean_text(text))
    return [t for t in tokens if t not in STOP_WORDS]
