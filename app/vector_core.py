from dataclasses import dataclass
from typing import Callable


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: list[float]


def euclidean(a: list[float], b: list[float]) -> float:
    total = 0.0

    for i in range(len(a)):
        diff = a[i] - b[i]
        total += diff * diff

    return total ** 0.5


def cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 1.0

    return 1.0 - (
        dot / ((norm_a ** 0.5) * (norm_b ** 0.5))
    )


def manhattan(a: list[float], b: list[float]) -> float:
    total = 0.0

    for i in range(len(a)):
        total += abs(a[i] - b[i])

    return total


def get_distance_function(metric: str) -> Callable:
    if metric == "cosine":
        return cosine

    if metric == "manhattan":
        return manhattan

    return euclidean