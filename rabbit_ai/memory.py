from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .retrieval import SimpleTfidfVectorizer, cosine_similarity, normalize_text
from .types import Answer, MemoryRecord, SearchResult


class MemoryStore:
    def __init__(self, db_path: str = "rabbit_ai.db", cache_ttl_hours: int = 24) -> None:
        self.db_path = Path(db_path)
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS search_cache (
                    provider TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (provider, normalized_query)
                );

                CREATE TABLE IF NOT EXISTS page_cache (
                    url TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def save_interaction(self, query: str, answer: Answer) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interactions (
                    query,
                    normalized_query,
                    answer,
                    sources_json,
                    query_type,
                    confidence,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query,
                    normalize_text(query),
                    answer.text,
                    json.dumps(answer.sources),
                    answer.query_type,
                    answer.confidence,
                    time.time(),
                ),
            )

    def recall(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, query, normalized_query, answer, sources_json, query_type, confidence, created_at
                FROM interactions
                ORDER BY created_at DESC
                LIMIT 200
                """
            ).fetchall()

        if not rows:
            return []

        normalized_query = normalize_text(query)
        candidate_queries = [row["normalized_query"] for row in rows]
        vectorizer = SimpleTfidfVectorizer()
        matrix = vectorizer.fit_transform([normalized_query, *candidate_queries])
        scores = cosine_similarity(matrix[0], matrix[1:])

        memories: list[MemoryRecord] = []
        for row, score in zip(rows, scores, strict=True):
            memories.append(
                MemoryRecord(
                    id=int(row["id"]),
                    query=row["query"],
                    answer=row["answer"],
                    sources=json.loads(row["sources_json"]),
                    query_type=row["query_type"],
                    confidence=float(row["confidence"]),
                    created_at=str(row["created_at"]),
                    similarity=float(score),
                )
            )
        memories.sort(key=lambda memory: (-memory.similarity, -memory.id))
        return memories[:limit]

    def cache_search(self, query: str, provider: str, results: list[SearchResult]) -> None:
        payload = [
            {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "rank": result.rank,
                "source": result.source,
            }
            for result in results
        ]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO search_cache (provider, normalized_query, results_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(provider, normalized_query)
                DO UPDATE SET results_json = excluded.results_json, created_at = excluded.created_at
                """,
                (provider, normalize_text(query), json.dumps(payload), time.time()),
            )

    def get_cached_search(self, query: str, provider: str) -> list[SearchResult] | None:
        normalized_query = normalize_text(query)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT results_json, created_at
                FROM search_cache
                WHERE provider = ? AND normalized_query = ?
                """,
                (provider, normalized_query),
            ).fetchone()

        if row is None:
            return None
        if time.time() - float(row["created_at"]) > self.cache_ttl_seconds:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM search_cache WHERE provider = ? AND normalized_query = ?",
                    (provider, normalized_query),
                )
            return None

        return [SearchResult(**item) for item in json.loads(row["results_json"])]

    def cache_page(self, url: str, title: str, text: str, source: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO page_cache (url, title, text, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url)
                DO UPDATE SET title = excluded.title, text = excluded.text, source = excluded.source, created_at = excluded.created_at
                """,
                (url, title, text, source, time.time()),
            )

    def get_cached_page(self, url: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT title, text, source, created_at FROM page_cache WHERE url = ?",
                (url,),
            ).fetchone()

        if row is None:
            return None
        if time.time() - float(row["created_at"]) > self.cache_ttl_seconds:
            with self._connect() as connection:
                connection.execute("DELETE FROM page_cache WHERE url = ?", (url,))
            return None
        return {
            "title": row["title"],
            "text": row["text"],
            "source": row["source"],
            "created_at": float(row["created_at"]),
        }

    def clear_cache(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM search_cache")
            connection.execute("DELETE FROM page_cache")

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            interactions = int(connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0])
            search_cache = int(connection.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0])
            page_cache = int(connection.execute("SELECT COUNT(*) FROM page_cache").fetchone()[0])
        return {
            "interactions": interactions,
            "search_cache_entries": search_cache,
            "page_cache_entries": page_cache,
        }

