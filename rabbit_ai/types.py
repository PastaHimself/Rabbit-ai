from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    rank: int = 0
    source: str = "duckduckgo"


@dataclass(slots=True)
class Passage:
    title: str
    url: str
    text: str
    source: str
    rank: int = 0
    score: float = 0.0


@dataclass(slots=True)
class MemoryRecord:
    id: int
    query: str
    answer: str
    sources: list[str] = field(default_factory=list)
    query_type: str = "factoid"
    confidence: float = 0.0
    created_at: str = ""
    similarity: float = 0.0


@dataclass(slots=True)
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    used_memory: bool = False
    used_web: bool = False
    query_type: str = "factoid"

