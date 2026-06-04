from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", str(text).lower())


def jaccard_similarity(a: str, b: str) -> float:
    left = set(tokenize(a))
    right = set(tokenize(b))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cosine_bow_similarity(a: str, b: str) -> float:
    left = Counter(tokenize(a))
    right = Counter(tokenize(b))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / max(left_norm * right_norm, 1e-12)


def stable_hash_embedding(text: str, dim: int = 384) -> list[float]:
    values = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        values[idx] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return values
    return [value / norm for value in values]

