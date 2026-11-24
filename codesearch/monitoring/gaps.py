"""Coverage gaps: cluster the questions the app abstained on, biggest topic first."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from codesearch.interfaces import Embedder


@dataclass(frozen=True)
class GapCluster:
    size: int  # how many abstained questions fell in this topic
    examples: list[str]  # a few distinct questions, to show a human what the topic is


def content_gaps(
    questions: list[str], embedder: Embedder, k: int = 5, examples: int = 5
) -> list[GapCluster]:
    if k <= 0:
        raise ValueError(f"Expected k > 0 got {k}")

    if examples <= 0:
        raise ValueError(f"expected examples to be greater than 0 got {examples}")

    if not questions:
        return []

    n_clusters = min(k, len(set(questions)))
    vectors = np.asarray(embedder.embed(questions), dtype=np.float32)
    # unit length, so k-means groups by direction (cosine)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    if n_clusters == 1:
        labels = np.zeros(len(questions), dtype=int)  # [0, 0, 0, ...]
    else:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(vectors)

    # group the questions by label, keeping their original order
    members: dict[int, list[str]] = {}
    for question, label in zip(questions, labels, strict=True):
        members.setdefault(int(label), []).append(question)

    clusters = [
        GapCluster(
            size=len(group),  # repeats count: 3x the same question = size 3
            examples=list(dict.fromkeys(group))[:examples],  # unique, first-seen order, capped
        )
        for group in members.values()
    ]
    return sorted(clusters, key=lambda c: c.size, reverse=True)  # biggest gap first
