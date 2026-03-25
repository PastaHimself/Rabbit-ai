from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True, slots=True)
class SearchConfig:
    max_results: int = 5
    max_pages_to_fetch: int = 3
    timeout_seconds: int = 10
    cache_ttl_hours: int = 24
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )


@dataclass(frozen=True, slots=True)
class RankingWeights:
    cosine: float = 0.55
    title_overlap: float = 0.20
    rank_prior: float = 0.15
    source_quality: float = 0.10


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    memory_reuse_threshold: float = 0.82
    max_answer_sentences: int = 4
    min_passage_length: int = 80
    max_passages_per_page: int = 3
    max_cached_passage_chars: int = 10000


@dataclass(frozen=True, slots=True)
class RabbitConfig:
    app_name: str = "Rabbit AI"
    db_path: str = "rabbit_ai.db"
    search: SearchConfig = field(default_factory=SearchConfig)
    ranking: RankingWeights = field(default_factory=RankingWeights)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


DEFAULT_CONFIG = RabbitConfig()

