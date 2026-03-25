from __future__ import annotations

import abc
import json
import re
import sqlite3
import statistics
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup


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
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
TIME_SENSITIVE_PATTERN = re.compile(r"\b(latest|today|current|now|recent|news|this year|yesterday|tomorrow|202\d|203\d)\b")


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


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text, keep_stopwords=False))


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if keep_stopwords:
        return tokens
    filtered = []
    for token in tokens:
        if token in DEFAULT_STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        filtered.append(token)
    return filtered


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    counts = Counter(tokenize(text))
    return [token for token, _ in counts.most_common(limit)]


def dense_keyword_query(text: str, limit: int = 8) -> str:
    keywords = extract_keywords(text, limit=limit)
    return " ".join(keywords) if keywords else text


class SimpleTfidfVectorizer:
    def __init__(self) -> None:
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, documents: list[str]) -> "SimpleTfidfVectorizer":
        token_lists = [tokenize(doc) for doc in documents]
        vocabulary = sorted({token for tokens in token_lists for token in tokens})
        self.vocabulary_ = {token: index for index, token in enumerate(vocabulary)}
        document_count = len(documents)
        if not vocabulary:
            self.idf_ = np.zeros(0, dtype=float)
            return self

        document_frequencies = np.zeros(len(vocabulary), dtype=float)
        for tokens in token_lists:
            for token in set(tokens):
                document_frequencies[self.vocabulary_[token]] += 1.0

        self.idf_ = np.log((1.0 + document_count) / (1.0 + document_frequencies)) + 1.0
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise ValueError("SimpleTfidfVectorizer must be fitted before transform().")

        matrix = np.zeros((len(documents), len(self.vocabulary_)), dtype=float)
        if not self.vocabulary_:
            return matrix

        for row_index, document in enumerate(documents):
            counts = Counter(tokenize(document))
            if not counts:
                continue
            total_tokens = float(sum(counts.values()))
            for token, count in counts.items():
                column = self.vocabulary_.get(token)
                if column is None:
                    continue
                tf = count / total_tokens
                matrix[row_index, column] = tf * self.idf_[column]
        return matrix

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)


def cosine_similarity(query_vector: np.ndarray, document_matrix: np.ndarray) -> np.ndarray:
    if document_matrix.size == 0:
        return np.zeros(0, dtype=float)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return np.zeros(document_matrix.shape[0], dtype=float)
    document_norms = np.linalg.norm(document_matrix, axis=1)
    safe_norms = np.where(document_norms == 0, 1.0, document_norms)
    scores = document_matrix @ query_vector / (safe_norms * query_norm)
    scores = np.where(document_norms == 0, 0.0, scores)
    return scores


def keyword_overlap_score(query_text: str, candidate_text: str) -> float:
    query_tokens = set(tokenize(query_text))
    if not query_tokens:
        return 0.0
    candidate_tokens = set(tokenize(candidate_text))
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def search_rank_prior(rank: int) -> float:
    return 1.0 / (1.0 + max(rank, 0))


def source_quality_bonus(url: str, source: str, title: str = "") -> float:
    if source == "memory":
        return 0.75

    domain = urlparse(url).netloc.lower()
    title_lower = title.lower()
    if "wikipedia.org" in domain:
        return 0.95
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.95
    if domain.startswith("docs.") or "/docs/" in url.lower():
        return 0.9
    if "official" in title_lower or "documentation" in title_lower:
        return 0.85
    return 0.6


class Ranker:
    def __init__(self, weights: RankingWeights | None = None) -> None:
        self.weights = weights or RankingWeights()

    def rank(self, query: str, passages: list[Passage]) -> list[Passage]:
        if not passages:
            return []

        documents = [passage.text for passage in passages]
        vectorizer = SimpleTfidfVectorizer()
        matrix = vectorizer.fit_transform([query, *documents])
        cosine_scores = cosine_similarity(matrix[0], matrix[1:])

        ranked: list[Passage] = []
        for passage, cosine_score in zip(passages, cosine_scores, strict=True):
            title_overlap = keyword_overlap_score(query, passage.title)
            prior = search_rank_prior(passage.rank)
            quality = source_quality_bonus(passage.url, passage.source, passage.title)
            total_score = (
                self.weights.cosine * float(cosine_score)
                + self.weights.title_overlap * title_overlap
                + self.weights.rank_prior * prior
                + self.weights.source_quality * quality
            )
            ranked.append(replace(passage, score=round(total_score, 6)))

        ranked.sort(key=lambda passage: (-passage.score, passage.rank, passage.title.lower()))
        return ranked


def chunk_text(text: str, max_chars: int = 420) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and current_length + len(sentence) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_length = len(sentence)
        else:
            current.append(sentence)
            current_length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def top_relevant_chunks(query: str, text: str, *, limit: int = 3, max_chars: int = 420) -> list[str]:
    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        return []
    scored = []
    for index, chunk in enumerate(chunks):
        score = keyword_overlap_score(query, chunk) + 0.05 * search_rank_prior(index)
        scored.append((score, index, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored[:limit] if chunk]


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

    def get_cached_page(self, url: str) -> dict[str, object] | None:
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


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class SearchProvider(abc.ABC):
    name = "search"

    @abc.abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class DuckDuckGoSearchProvider(SearchProvider):
    name = "duckduckgo"
    HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
    LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"

    def __init__(self, config: SearchConfig | None = None, session: requests.Session | None = None) -> None:
        self.config = config or SearchConfig()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        for endpoint, parser in (
            (self.HTML_ENDPOINT, self._parse_html_results),
            (self.LITE_ENDPOINT, self._parse_lite_results),
        ):
            html = self._download(endpoint, {"q": query})
            if not html:
                continue
            results = parser(html)
            if results:
                return results[:max_results]
        return []

    def _download(self, endpoint: str, params: dict[str, str]) -> str:
        try:
            response = self.session.get(endpoint, params=params, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return ""

    @staticmethod
    def _looks_like_challenge(html: str) -> bool:
        lowered = html.lower()
        return "anomaly.js" in lowered or "challenge-form" in lowered or "botnet" in lowered

    @classmethod
    def _parse_html_results(cls, html: str) -> list[SearchResult]:
        if cls._looks_like_challenge(html):
            return []

        soup = BeautifulSoup(html, "html.parser")
        wrappers = soup.select("div.result")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        if wrappers:
            for rank, wrapper in enumerate(wrappers):
                link = wrapper.select_one("a.result__a") or wrapper.select_one("h2 a") or wrapper.select_one("a[href]")
                if link is None:
                    continue
                url = cls._unwrap_url(link.get("href", ""))
                if not cls._is_external_url(url) or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = _clean_spaces(link.get_text(" ", strip=True))
                snippet_node = wrapper.select_one(".result__snippet") or wrapper.select_one(".result__extras__url")
                snippet = _clean_spaces(snippet_node.get_text(" ", strip=True)) if snippet_node else ""
                results.append(SearchResult(title=title, url=url, snippet=snippet, rank=rank, source=cls.name))
        if results:
            return results

        for rank, link in enumerate(soup.select("a[href]")):
            title = _clean_spaces(link.get_text(" ", strip=True))
            url = cls._unwrap_url(link.get("href", ""))
            if not title or not cls._is_external_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(SearchResult(title=title, url=url, rank=rank, source=cls.name))
            if len(results) >= 10:
                break
        return results

    @classmethod
    def _parse_lite_results(cls, html: str) -> list[SearchResult]:
        if cls._looks_like_challenge(html):
            return []

        soup = BeautifulSoup(html, "html.parser")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for rank, link in enumerate(soup.select("a[href]")):
            title = _clean_spaces(link.get_text(" ", strip=True))
            if not title:
                continue
            url = cls._unwrap_url(link.get("href", ""))
            if not cls._is_external_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(SearchResult(title=title, url=url, rank=rank, source=cls.name))
            if len(results) >= 10:
                break
        return results

    @staticmethod
    def _unwrap_url(url: str) -> str:
        if not url:
            return ""
        if url.startswith("//"):
            return f"https:{url}"
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if "uddg" in query:
            return unquote(query["uddg"][0])
        return url

    @staticmethod
    def _is_external_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and "duckduckgo.com" not in parsed.netloc.lower()


class WikipediaSearchProvider(SearchProvider):
    name = "wikipedia"
    SEARCH_ENDPOINT = "https://en.wikipedia.org/w/index.php"

    def __init__(self, config: SearchConfig | None = None, session: requests.Session | None = None) -> None:
        self.config = config or SearchConfig()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            response = self.session.get(
                self.SEARCH_ENDPOINT,
                params={"search": query, "title": "Special:Search", "ns0": "1"},
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        direct_title = soup.select_one("#firstHeading")
        if direct_title and "wikipedia.org/wiki/" in response.url:
            snippet_node = soup.select_one("p")
            snippet = _clean_spaces(snippet_node.get_text(" ", strip=True)) if snippet_node else ""
            return [
                SearchResult(
                    title=_clean_spaces(direct_title.get_text(" ", strip=True)),
                    url=response.url,
                    snippet=snippet,
                    rank=0,
                    source=self.name,
                )
            ]

        results: list[SearchResult] = []
        for rank, item in enumerate(soup.select("li.mw-search-result")):
            link = item.select_one("a[href]")
            if link is None:
                continue
            snippet_node = item.select_one(".searchresult")
            results.append(
                SearchResult(
                    title=_clean_spaces(link.get_text(" ", strip=True)),
                    url=urljoin("https://en.wikipedia.org/", link.get("href", "")),
                    snippet=_clean_spaces(snippet_node.get_text(" ", strip=True)) if snippet_node else "",
                    rank=rank,
                    source=self.name,
                )
            )
            if len(results) >= max_results:
                break
        return results


class PageFetcher:
    def __init__(self, config: SearchConfig | None = None, session: requests.Session | None = None) -> None:
        self.config = config or SearchConfig()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def fetch(self, url: str) -> tuple[str, str]:
        try:
            response = self.session.get(url, timeout=self.config.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException:
            return "", ""

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text/plain" not in content_type:
            return "", ""
        return self._extract_text(response.text, response.url)

    @staticmethod
    def _extract_text(html: str, url: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        title_node = soup.find("title")
        title = _clean_spaces(title_node.get_text(" ", strip=True)) if title_node else url

        for node in soup(["script", "style", "noscript", "svg", "img", "header", "footer", "nav", "aside", "form"]):
            node.decompose()

        container = soup.find("main") or soup.find("article") or soup.body or soup
        blocks = []
        for node in container.find_all(["p", "li", "h1", "h2", "h3"]):
            text = _clean_spaces(node.get_text(" ", strip=True))
            if len(text) >= 40:
                blocks.append(text)
        return title, _clean_spaces("\n".join(blocks))

    def fetch_passages(
        self,
        query: str,
        result: SearchResult,
        *,
        max_passages: int = 3,
        min_length: int = 80,
    ) -> list[Passage]:
        title, text = self.fetch(result.url)
        if not text:
            if result.snippet:
                return [
                    Passage(
                        title=result.title,
                        url=result.url,
                        text=result.snippet,
                        source=result.source,
                        rank=result.rank,
                    )
                ]
            return []

        return self.build_passages_from_text(
            query=query,
            title=title or result.title,
            url=result.url,
            text=text,
            source=result.source,
            rank=result.rank,
            max_passages=max_passages,
            min_length=min_length,
        )

    @staticmethod
    def build_passages_from_text(
        query: str,
        title: str,
        url: str,
        text: str,
        source: str,
        rank: int,
        *,
        max_passages: int = 3,
        min_length: int = 80,
    ) -> list[Passage]:
        if not text:
            return []

        chunks = top_relevant_chunks(query, text, limit=max_passages)
        passages = [
            Passage(
                title=title,
                url=url,
                text=chunk,
                source=source,
                rank=rank,
            )
            for chunk in chunks
            if len(chunk) >= min_length
        ]
        if passages:
            return passages
        if len(text) >= min_length:
            return [
                Passage(
                    title=title,
                    url=url,
                    text=text[: min(len(text), 600)],
                    source=source,
                    rank=rank,
                )
            ]
        return []


class Reasoner:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()

    def classify(self, query: str) -> str:
        lowered = query.lower()
        if TIME_SENSITIVE_PATTERN.search(lowered):
            return "time_sensitive"
        if " vs " in lowered or "difference between" in lowered or lowered.startswith("compare "):
            return "comparison"
        if lowered.startswith("how ") or "how do " in lowered or "how can " in lowered:
            return "how_to"
        if lowered.count("?") > 1 or (" and " in lowered and any(word in lowered for word in ("what", "how", "why", "when", "where"))):
            return "multi_part"
        if lowered.startswith(("what is", "who is", "define", "explain")):
            return "definition"
        return "factoid"

    def compose(self, query: str, passages: list[Passage], memories: list[MemoryRecord]) -> Answer:
        query_type = self.classify(query)
        if not passages and memories:
            top_memory = memories[0]
            return Answer(
                text=top_memory.answer,
                sources=top_memory.sources,
                confidence=max(0.35, min(0.8, top_memory.similarity)),
                used_memory=True,
                used_web=False,
                query_type=query_type,
            )

        if not passages:
            return Answer(
                text="I could not find enough reliable information to answer that confidently.",
                confidence=0.1,
                used_memory=bool(memories),
                used_web=False,
                query_type=query_type,
            )

        selected_sentences = self._select_sentences(query, passages)
        text = self._render_answer(query_type, selected_sentences)
        confidence = self._estimate_confidence(passages, memories)
        if confidence < 0.35:
            text = f"I found limited evidence. Best available summary: {text}"

        unique_sources = list(OrderedDict.fromkeys(passage.url for passage in passages if passage.url))
        return Answer(
            text=text,
            sources=unique_sources[:3],
            confidence=confidence,
            used_memory=any(passage.source == "memory" for passage in passages),
            used_web=any(passage.source != "memory" for passage in passages),
            query_type=query_type,
        )

    def _select_sentences(self, query: str, passages: list[Passage]) -> list[str]:
        scored: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        for passage_index, passage in enumerate(passages[:5]):
            for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", passage.text)):
                sentence = sentence.strip()
                normalized = re.sub(r"\s+", " ", sentence.lower())
                if len(sentence) < 30 or normalized in seen:
                    continue
                seen.add(normalized)
                overlap = keyword_overlap_score(query, sentence)
                position_bonus = 0.05 / (1 + sentence_index)
                score = passage.score + overlap + position_bonus - 0.01 * passage_index
                scored.append((score, len(scored), sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [sentence for _, _, sentence in scored[: self.config.max_answer_sentences]]

    def _render_answer(self, query_type: str, sentences: list[str]) -> str:
        if not sentences:
            return "I found evidence, but it was too fragmented to summarize clearly."
        if query_type == "how_to":
            steps = [self._trim(sentence) for sentence in sentences[:3]]
            return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
        if query_type == "comparison":
            lead = self._trim(sentences[0])
            support = self._trim(sentences[1]) if len(sentences) > 1 else ""
            return f"{lead} {support}".strip()
        if query_type == "multi_part":
            return "Key points: " + " ".join(self._trim(sentence) for sentence in sentences[:3])
        return " ".join(self._trim(sentence) for sentence in sentences[:3])

    @staticmethod
    def _trim(sentence: str) -> str:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        return sentence[:1].upper() + sentence[1:] if sentence else sentence

    @staticmethod
    def _estimate_confidence(passages: list[Passage], memories: list[MemoryRecord]) -> float:
        if not passages and not memories:
            return 0.1
        top_score = passages[0].score if passages else 0.0
        source_diversity = len({passage.url for passage in passages[:3] if passage.url})
        memory_bonus = 0.08 if memories and memories[0].similarity > 0.65 else 0.0
        confidence = 0.2 + 0.7 * min(top_score, 1.0) + 0.05 * min(source_diversity, 3) + memory_bonus
        return round(max(0.1, min(confidence, 0.95)), 3)


class RabbitAI:
    def __init__(
        self,
        config: RabbitConfig | None = None,
        memory_store: MemoryStore | None = None,
        search_provider: SearchProvider | None = None,
        fallback_provider: SearchProvider | None = None,
        page_fetcher: PageFetcher | None = None,
        ranker: Ranker | None = None,
        reasoner: Reasoner | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.memory = memory_store or MemoryStore(
            db_path=self.config.db_path,
            cache_ttl_hours=self.config.search.cache_ttl_hours,
        )
        self.search_provider = search_provider or DuckDuckGoSearchProvider(self.config.search)
        self.fallback_provider = fallback_provider or WikipediaSearchProvider(self.config.search)
        self.page_fetcher = page_fetcher or PageFetcher(self.config.search)
        self.ranker = ranker or Ranker(self.config.ranking)
        self.reasoner = reasoner or Reasoner(self.config.runtime)

    def ask(self, query: str, use_web: bool = True) -> Answer:
        query = query.strip()
        if not query:
            return Answer(text="Please enter a question.", confidence=0.0)

        query_type = self.reasoner.classify(query)
        memories = self.memory.recall(query, limit=5)

        if query_type != "time_sensitive" and memories and memories[0].similarity >= self.config.runtime.memory_reuse_threshold:
            top_memory = memories[0]
            answer = Answer(
                text=top_memory.answer,
                sources=top_memory.sources,
                confidence=max(top_memory.confidence, min(0.92, top_memory.similarity)),
                used_memory=True,
                used_web=False,
                query_type=query_type,
            )
            self.memory.save_interaction(query, answer)
            return answer

        candidate_passages = self._memory_support_passages(memories)
        used_web = False
        if use_web:
            web_passages = self._collect_web_passages(query)
            used_web = bool(web_passages)
            candidate_passages.extend(web_passages)

        ranked = self.ranker.rank(query, candidate_passages)
        answer = self.reasoner.compose(query, ranked, memories)
        answer.query_type = query_type
        answer.used_web = used_web
        answer.used_memory = answer.used_memory or any(passage.source == "memory" for passage in ranked)
        self.memory.save_interaction(query, answer)
        return answer

    def stats(self) -> dict[str, int]:
        return self.memory.stats()

    def clear_cache(self) -> None:
        self.memory.clear_cache()

    def _memory_support_passages(self, memories: list[MemoryRecord]) -> list[Passage]:
        passages: list[Passage] = []
        for index, memory in enumerate(memories[:3]):
            passages.append(
                Passage(
                    title=f"Memory: {memory.query}",
                    url=f"memory://{memory.id}",
                    text=memory.answer,
                    source="memory",
                    rank=index,
                )
            )
        return passages

    def _collect_web_passages(self, query: str) -> list[Passage]:
        results = self._search_with_fallbacks(query)
        passages: list[Passage] = []
        for result in results[: self.config.search.max_pages_to_fetch]:
            passages.extend(self._passages_for_result(query, result))
        return passages

    def _search_with_fallbacks(self, query: str) -> list[SearchResult]:
        cached = self.memory.get_cached_search(query, self.search_provider.name)
        if cached is not None:
            results = cached
        else:
            results = self.search_provider.search(query, max_results=self.config.search.max_results)
            if results:
                self.memory.cache_search(query, self.search_provider.name, results)
        if self._search_results_look_reliable(results):
            return results

        rewritten_query = dense_keyword_query(query)
        if rewritten_query and normalize_text(rewritten_query) != normalize_text(query):
            cached = self.memory.get_cached_search(rewritten_query, self.search_provider.name)
            if cached is not None:
                results = cached
            else:
                results = self.search_provider.search(rewritten_query, max_results=self.config.search.max_results)
                if results:
                    self.memory.cache_search(rewritten_query, self.search_provider.name, results)
            if self._search_results_look_reliable(results):
                return results

        fallback_results = self.fallback_provider.search(query, max_results=3)
        if fallback_results:
            self.memory.cache_search(query, self.fallback_provider.name, fallback_results)
            return fallback_results
        return results

    def _passages_for_result(self, query: str, result: SearchResult) -> list[Passage]:
        cached_page = self.memory.get_cached_page(result.url)
        if cached_page:
            passages = self.page_fetcher.build_passages_from_text(
                query=query,
                title=str(cached_page["title"]),
                url=result.url,
                text=str(cached_page["text"]),
                source=str(cached_page["source"]),
                rank=result.rank,
                max_passages=self.config.runtime.max_passages_per_page,
                min_length=self.config.runtime.min_passage_length,
            )
            if passages:
                return passages

        passages = self.page_fetcher.fetch_passages(
            query,
            result,
            max_passages=self.config.runtime.max_passages_per_page,
            min_length=self.config.runtime.min_passage_length,
        )
        if passages:
            cached_text = "\n".join(passage.text for passage in passages)[: self.config.runtime.max_cached_passage_chars]
            self.memory.cache_page(result.url, passages[0].title, cached_text, result.source)
        return passages

    @staticmethod
    def _search_results_look_reliable(results: list[SearchResult]) -> bool:
        if not results:
            return False
        useful = 0
        for result in results[:3]:
            combined = f"{result.title} {result.snippet}".strip()
            if len(combined) >= 20:
                useful += 1
        return useful > 0


DEFAULT_BENCHMARKS = [
    {"query": "What is Python programming language?", "expected_keywords": ["python", "language", "readability"]},
    {"query": "Compare HTTP and HTTPS", "expected_keywords": ["http", "https", "security"]},
    {"query": "How do I boil an egg?", "expected_keywords": ["boil", "water", "egg"]},
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


def run_cli() -> None:
    rabbit = RabbitAI()
    show_sources = False

    print("Rabbit AI")
    print("Type a question or use: help, sources on, sources off, clear-cache, stats, exit")

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Rabbit AI.")
            break

        if not query:
            continue
        if query in {"exit", "quit"}:
            print("Exiting Rabbit AI.")
            break
        if query == "help":
            print("Commands: help, sources on, sources off, clear-cache, stats, exit")
            continue
        if query == "sources on":
            show_sources = True
            print("Source display enabled.")
            continue
        if query == "sources off":
            show_sources = False
            print("Source display disabled.")
            continue
        if query == "clear-cache":
            rabbit.clear_cache()
            print("Search and page cache cleared.")
            continue
        if query == "stats":
            print(json.dumps(rabbit.stats(), indent=2))
            continue

        answer = rabbit.ask(query, use_web=True)
        print(f"\n{answer.text}")
        print(f"\nconfidence={answer.confidence:.2f} memory={answer.used_memory} web={answer.used_web}")
        if show_sources and answer.sources:
            print("sources:")
            for source in answer.sources:
                print(f"- {source}")


if __name__ == "__main__":
    run_cli()
