"""Writes dashboards/codesearch.json: the CodeSearch Grafana dashboard.

The dashboard is code so it can be reviewed in a PR and regenerated, instead of a
2,000-line JSON export nobody reads. Run: uv run python deploy/grafana/make_dashboard.py
"""

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).parent / "dashboards" / "codesearch.json"
DS = {"type": "prometheus", "uid": "prometheus"}  # matches provisioning/datasources
RATE = "$__rate_interval"  # Grafana picks a window that fits the scrape interval

# Everything the users hit; leave out Prometheus scraping /metrics and health probes.
USER = 'path!~"/metrics|/healthz|/readyz"'


def known(metric: str) -> str:
    """Drop NaN ("not enough answers yet"): NaN >= 0 is false, so the panel shows no data."""
    return f"{metric} >= 0"


def target(expr: str, legend: str = "", ref: str = "A", **extra: Any) -> dict[str, Any]:
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, **extra}


def quantile(q: float, metric: str, by: str = "") -> str:
    labels = f"le, {by}" if by else "le"
    return f"histogram_quantile({q}, sum by ({labels}) (rate({metric}_bucket[{RATE}])))"


def thresholds(*steps: tuple[float | None, str]) -> dict[str, Any]:
    return {"mode": "absolute", "steps": [{"value": v, "color": c} for v, c in steps]}


class Layout:
    """Places panels left to right on Grafana's 24-column grid, wrapping rows."""

    def __init__(self) -> None:
        self.panels: list[dict[str, Any]] = []
        self.x = self.y = self.row_h = 0

    def row(self, title: str) -> None:
        self.newline()
        self.panels.append(
            {
                "type": "row",
                "title": title,
                "collapsed": False,
                "id": len(self.panels) + 1,
                "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1},
                "panels": [],
            }
        )
        self.y += 1

    def newline(self) -> None:
        self.y += self.row_h
        self.x = self.row_h = 0

    def add(self, panel: dict[str, Any], w: int, h: int) -> None:
        if self.x + w > 24:
            self.newline()
        panel["id"] = len(self.panels) + 1
        panel["gridPos"] = {"x": self.x, "y": self.y, "w": w, "h": h}
        panel.setdefault("datasource", DS)
        self.panels.append(panel)
        self.x += w
        self.row_h = max(self.row_h, h)


def stat(
    title: str,
    expr: str,
    unit: str,
    desc: str,
    steps: dict[str, Any] | None = None,
    decimals: int | None = None,
    mappings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "unit": unit,
        "thresholds": steps or thresholds((None, "text")),
        "color": {"mode": "thresholds"},
        "noValue": "no data yet",
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    if mappings:
        defaults["mappings"] = mappings
    return {
        "type": "stat",
        "title": title,
        "description": desc,
        "targets": [target(expr)],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "area",
            "textMode": "value",
            "justifyMode": "center",
        },
    }


def series(
    title: str,
    targets: list[dict[str, Any]],
    unit: str,
    desc: str,
    stack: bool = False,
    steps: dict[str, Any] | None = None,
    bars: bool = False,
) -> dict[str, Any]:
    custom: dict[str, Any] = {
        "drawStyle": "bars" if bars else "line",
        "lineWidth": 2,
        "fillOpacity": 60 if bars else (25 if stack else 10),
        "showPoints": "never",
        "spanNulls": True,
        "stacking": {"mode": "normal" if stack else "none"},
    }
    if steps is not None:
        custom["thresholdsStyle"] = {"mode": "dashed"}
    return {
        "type": "timeseries",
        "title": title,
        "description": desc,
        "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": custom,
                "thresholds": steps or thresholds((None, "green")),
                "color": {"mode": "palette-classic"},
            },
            "overrides": [],
        },
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def build() -> dict[str, Any]:
    g = Layout()
    good_high = thresholds((None, "red"), (0.7, "orange"), (0.85, "green"))

    # ------------------------------------------------------------------ overview
    g.row("Overview")
    g.add(
        stat(
            "Requests / s",
            f"sum(rate(codesearch_http_requests_total{{{USER}}}[{RATE}]))",
            "reqps",
            "API requests per second, not counting /metrics and health checks.",
            decimals=2,
        ),
        4,
        4,
    )
    g.add(
        stat(
            "Answer p95",
            quantile(0.95, "codesearch_answer_seconds"),
            "s",
            "95% of answers (cached or generated) finished within this time.",
            thresholds((None, "green"), (5, "orange"), (15, "red")),
            decimals=2,
        ),
        4,
        4,
    )
    g.add(
        stat(
            "First token p50",
            quantile(0.5, "codesearch_ttft_seconds"),
            "s",
            "Median time until the first word of a streamed answer appears.",
            thresholds((None, "green"), (2, "orange"), (5, "red")),
            decimals=2,
        ),
        4,
        4,
    )
    g.add(
        stat(
            "Server errors",
            # "or vector(0)": no 5xx at all is 0%, not "no data"
            f'(sum(rate(codesearch_http_requests_total{{{USER},status=~"5.."}}[{RATE}]))'
            " or vector(0))"
            f" / sum(rate(codesearch_http_requests_total{{{USER}}}[{RATE}]))",
            "percentunit",
            "Share of requests answered with a 5xx (LLM down, index not ready, bugs).",
            thresholds((None, "green"), (0.01, "orange"), (0.05, "red")),
            decimals=1,
        ),
        4,
        4,
    )
    g.add(
        stat(
            "LLM busy (429), last hour",
            'sum(increase(codesearch_http_requests_total{status="429"}[1h])) or vector(0)',
            "short",
            "Answers refused because the Groq quota was used up. Users saw 'busy, try again'.",
            thresholds((None, "green"), (1, "orange"), (20, "red")),
            decimals=0,
        ),
        4,
        4,
    )
    g.add(
        stat(
            "Answers in window",
            "codesearch_window_answers",
            "short",
            "How many recent answers the rolling gauges (abstain, cache, groundedness, "
            "drift) are computed over.",
            decimals=0,
        ),
        4,
        4,
    )

    # ------------------------------------------------------------------ traffic
    g.row("Traffic and latency")
    g.add(
        series(
            "Requests / s by status",
            [
                target(
                    f"sum by (status) (rate(codesearch_http_requests_total{{{USER}}}[{RATE}]))",
                    "{{status}}",
                )
            ],
            "reqps",
            "Every API response, split by HTTP status.",
            stack=True,
        ),
        12,
        8,
    )
    g.add(
        series(
            "Answers / s by cache tier",
            [
                target(
                    f"sum by (cache_tier) (rate(codesearch_answers_total[{RATE}]))",
                    "{{cache_tier}}",
                )
            ],
            "reqps",
            "generated = the LLM was called; exact / semantic = served from a cache; "
            "shared = waited for an identical in-flight question (single-flight).",
            stack=True,
        ),
        12,
        8,
    )
    g.add(
        series(
            "Answer latency by cache tier",
            [
                target(
                    quantile(0.5, "codesearch_answer_seconds", "cache_tier"),
                    "p50 {{cache_tier}}",
                    "A",
                ),
                target(
                    quantile(0.95, "codesearch_answer_seconds", "cache_tier"),
                    "p95 {{cache_tier}}",
                    "B",
                ),
            ],
            "s",
            "Cache hits should sit near zero; generated answers are bounded by the LLM.",
        ),
        12,
        8,
    )
    g.add(
        series(
            "First token and retrieval",
            [
                target(quantile(0.5, "codesearch_ttft_seconds"), "first token p50", "A"),
                target(quantile(0.95, "codesearch_ttft_seconds"), "first token p95", "B"),
                target(quantile(0.95, "codesearch_retrieval_seconds"), "retrieval p95", "C"),
            ],
            "s",
            "First token = retrieval + rerank + the LLM's own delay. If retrieval is a "
            "big share, look at the embedder and reranker, not the LLM.",
        ),
        12,
        8,
    )

    # ------------------------------------------------------------------ quality
    g.row("Answer quality (rolling window)")
    g.add(
        stat(
            "Groundedness (median)",
            known("codesearch_groundedness_median"),
            "percentunit",
            "Share of answer sentences the NLI check found supported by the sources. "
            "Alert below 70% for 15 minutes.",
            good_high,
            decimals=0,
        ),
        6,
        5,
    )
    g.add(
        stat(
            "Abstain rate",
            known("codesearch_abstain_rate"),
            "percentunit",
            "Share of questions answered with 'not covered in the React docs'.",
            thresholds((None, "green"), (0.2, "orange"), (0.4, "red")),
            decimals=0,
        ),
        6,
        5,
    )
    g.add(
        stat(
            "Cache hit rate",
            known("codesearch_cache_hit_rate"),
            "percentunit",
            "Share of answers that did not call the LLM. Every hit saves Groq quota.",
            thresholds((None, "text")),
            decimals=0,
        ),
        6,
        5,
    )
    g.add(
        stat(
            "Query drift",
            known("codesearch_query_drift"),
            "none",
            "Cosine distance between recent questions and the golden set (0 = same topics). "
            "Alert above 0.20 for 30 minutes.",
            thresholds((None, "green"), (0.15, "orange"), (0.2, "red")),
            decimals=3,
        ),
        6,
        5,
    )
    g.add(
        series(
            "Quality over time",
            [
                target(known("codesearch_groundedness_median"), "groundedness (median)", "A"),
                target(known("codesearch_abstain_rate"), "abstain rate", "B"),
                target(known("codesearch_cache_hit_rate"), "cache hit rate", "C"),
            ],
            "percentunit",
            "The rolling gauges over time. A step change usually lines up with a "
            "re-index or a prompt change.",
            steps=thresholds((None, "red"), (0.7, "transparent")),
        ),
        12,
        8,
    )
    g.add(
        series(
            "Query drift over time",
            [target(known("codesearch_query_drift"), "drift")],
            "none",
            "Rising drift: users ask about topics the eval set doesn't cover. "
            "Check GET /v1/admin/gaps.",
            steps=thresholds((None, "transparent"), (0.2, "red")),
        ),
        12,
        8,
    )

    # ------------------------------------------------------------------ feedback
    g.row("User feedback")
    g.add(
        stat(
            "Thumbs down share (all time)",
            'sum(codesearch_feedback_total{vote="down"}) / sum(codesearch_feedback_total)',
            "percentunit",
            "Of all votes since the API started, the share that were thumbs down.",
            thresholds((None, "green"), (0.2, "orange"), (0.4, "red")),
            decimals=0,
        ),
        6,
        8,
    )
    g.add(
        series(
            "Votes per hour",
            [target("sum by (vote) (increase(codesearch_feedback_total[1h]))", "{{vote}}")],
            "short",
            "Thumbs up / down from the UI. Read the comments: FeedbackStore.thumbs_down().",
            stack=True,
            bars=True,
        ),
        18,
        8,
    )

    # ------------------------------------------------------------------ index
    g.row("Index and re-indexing")
    g.add(
        stat(
            "Chunks in the live index",
            "codesearch_index_chunks",
            "short",
            "Chunks the retriever searches. A sudden drop means a bad build.",
            decimals=0,
        ),
        5,
        5,
    )
    g.add(
        stat(
            "Re-index runs",
            "codesearch_reindex_runs",
            "short",
            "Rebuilds attempted since the API started (webhook + timer).",
            decimals=0,
        ),
        5,
        5,
    )
    g.add(
        stat(
            "Last re-index",
            # only once a rebuild has run: before that the gauge is a meaningless 0
            "codesearch_reindex_last_success and on() codesearch_reindex_runs > 0",
            "none",
            "Whether the most recent rebuild worked. A failed build leaves the old "
            "index serving: nothing breaks, but the docs go stale. Empty until the "
            "first rebuild.",
            thresholds((None, "red"), (1, "green")),
            mappings=[
                {
                    "type": "value",
                    "options": {
                        "0": {"text": "FAILED", "color": "red"},
                        "1": {"text": "OK", "color": "green"},
                    },
                }
            ],
        ),
        5,
        5,
    )
    g.add(
        {
            "type": "table",
            "title": "Serving",
            "description": "The index version and react.dev commit the API answers from.",
            "targets": [target("codesearch_index_info", format="table", instant=True)],
            "transformations": [
                {
                    "id": "organize",
                    "options": {
                        "excludeByName": {
                            "Time": True,
                            "Value": True,
                            "__name__": True,
                            "instance": True,
                            "job": True,
                        },
                        "renameByName": {
                            "index_version": "Index version",
                            "corpus_sha": "react.dev commit",
                        },
                    },
                }
            ],
            "fieldConfig": {"defaults": {}, "overrides": []},
            "options": {"showHeader": True},
        },
        9,
        5,
    )
    g.add(
        series(
            "Live index size and re-index events",
            [
                target("codesearch_index_chunks", "chunks", "A"),
                target("changes(codesearch_index_chunks[5m]) > 0", "index swapped", "B"),
            ],
            "short",
            "The chunk count over time; a dot marks each swap to a new build.",
        ),
        24,
        7,
    )

    return {
        "uid": "codesearch",
        "title": "CodeSearch AI",
        "description": "Traffic, latency, answer quality, feedback and the index.",
        "tags": ["codesearch", "rag"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "editable": True,
        "graphTooltip": 1,  # shared crosshair across panels
        "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"},
        "templating": {"list": []},
        "annotations": {"list": []},
        "links": [],
        "panels": g.panels,
    }


def all_exprs(dashboard: dict[str, Any]) -> list[str]:
    """Every PromQL query in the dashboard (used to check them against Prometheus)."""
    return [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(build()['panels'])} panels)")
