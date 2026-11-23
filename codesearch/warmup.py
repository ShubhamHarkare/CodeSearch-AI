import time
from typing import Any

COMPONENTS = (("embedder", "embedder"), ("reranker", "reranker"), ("nli", "grounder"))
# (name for the log, attribute on the pipeline)


def warm_up(pipeline: Any) -> dict[str, float]:
    seconds: dict[str, float] = {}
    for name, attribute in COMPONENTS:
        component = getattr(pipeline, attribute, None)
        warm = getattr(component, "warm_up", None)
        if warm is None:
            continue  # fake or missing: nothing to time

        start = time.perf_counter()  # 1. clock before
        warm()  # 2. load the weights + one inference
        seconds[name] = round(time.perf_counter() - start, 2)  # 3. clock after, minus before
    return seconds
