from __future__ import annotations

import statistics
import time
from typing import Iterable

from .engine import RabbitAI

DEFAULT_BENCHMARKS = [
    {
        "query": "What is Python programming language?",
        "expected_keywords": ["python", "language", "readability"],
    },
    {
        "query": "Compare HTTP and HTTPS",
        "expected_keywords": ["http", "https", "security"],
    },
    {
        "query": "How do I boil an egg?",
        "expected_keywords": ["boil", "water", "egg"],
    },
]


def keyword_hit_score(answer_text: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    answer_lower = answer_text.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    return round(hits / len(expected_keywords), 3)


def evaluate_agent(
    agent: RabbitAI,
    benchmarks: list[dict[str, object]] | None = None,
    *,
    use_web: bool = True,
) -> list[dict[str, object]]:
    benchmarks = benchmarks or DEFAULT_BENCHMARKS
    report: list[dict[str, object]] = []

    for case in benchmarks:
        query = str(case["query"])
        expected_keywords = [str(item) for item in case.get("expected_keywords", [])]
        start = time.perf_counter()
        answer = agent.ask(query, use_web=use_web)
        latency = time.perf_counter() - start
        report.append(
            {
                "query": query,
                "answer": answer.text,
                "confidence": answer.confidence,
                "latency_seconds": round(latency, 3),
                "keyword_hit_score": keyword_hit_score(answer.text, expected_keywords),
                "sources": answer.sources,
                "used_memory": answer.used_memory,
                "used_web": answer.used_web,
            }
        )
    return report


def summarize_report(report: list[dict[str, object]]) -> dict[str, float]:
    if not report:
        return {"avg_latency_seconds": 0.0, "avg_keyword_hit_score": 0.0, "avg_confidence": 0.0}
    return {
        "avg_latency_seconds": round(statistics.mean(float(item["latency_seconds"]) for item in report), 3),
        "avg_keyword_hit_score": round(statistics.mean(float(item["keyword_hit_score"]) for item in report), 3),
        "avg_confidence": round(statistics.mean(float(item["confidence"]) for item in report), 3),
    }

