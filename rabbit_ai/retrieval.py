from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from urllib.parse import urlparse

try:
    import torch
except ModuleNotFoundError:
    torch = None

from .config import DEFAULT_STOPWORDS, RankingWeights
from .types import Passage

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


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
        self.idf_: object | None = None

    def fit(self, documents: list[str]) -> "SimpleTfidfVectorizer":
        token_lists = [tokenize(doc) for doc in documents]
        vocabulary = sorted({token for tokens in token_lists for token in tokens})
        self.vocabulary_ = {token: index for index, token in enumerate(vocabulary)}
        document_count = len(documents)
        if not vocabulary:
            self.idf_ = torch.zeros(0, dtype=torch.float32) if torch is not None else []
            return self

        if torch is not None:
            document_frequencies = torch.zeros(len(vocabulary), dtype=torch.float32)
            for tokens in token_lists:
                for token in set(tokens):
                    document_frequencies[self.vocabulary_[token]] += 1.0
            self.idf_ = torch.log((1.0 + document_count) / (1.0 + document_frequencies)) + 1.0
            return self

        document_frequencies = [0.0] * len(vocabulary)
        for tokens in token_lists:
            for token in set(tokens):
                document_frequencies[self.vocabulary_[token]] += 1.0
        self.idf_ = [math.log((1.0 + document_count) / (1.0 + frequency)) + 1.0 for frequency in document_frequencies]
        return self

    def transform(self, documents: list[str]):
        if self.idf_ is None:
            raise ValueError("SimpleTfidfVectorizer must be fitted before transform().")

        if torch is not None:
            matrix = torch.zeros((len(documents), len(self.vocabulary_)), dtype=torch.float32)
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

        matrix = [[0.0] * len(self.vocabulary_) for _ in documents]
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
                matrix[row_index][column] = tf * self.idf_[column]
        return matrix

    def fit_transform(self, documents: list[str]):
        self.fit(documents)
        return self.transform(documents)


def cosine_similarity(query_vector, document_matrix) -> list[float]:
    if torch is not None and hasattr(document_matrix, "numel"):
        if document_matrix.numel() == 0:
            return []
        query_norm = torch.linalg.vector_norm(query_vector)
        if torch.isclose(query_norm, query_vector.new_tensor(0.0)):
            return [0.0] * len(document_matrix)
        document_norms = torch.linalg.vector_norm(document_matrix, dim=1)
        safe_norms = torch.where(document_norms == 0, torch.ones_like(document_norms), document_norms)
        scores = torch.matmul(document_matrix, query_vector) / (safe_norms * query_norm)
        scores = torch.where(document_norms == 0, torch.zeros_like(scores), scores)
        return [float(value) for value in scores.tolist()]

    if not document_matrix:
        return []
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    if query_norm == 0:
        return [0.0] * len(document_matrix)
    scores: list[float] = []
    for document_vector in document_matrix:
        document_norm = math.sqrt(sum(value * value for value in document_vector))
        if document_norm == 0:
            scores.append(0.0)
            continue
        dot_product = sum(document_value * query_value for document_value, query_value in zip(document_vector, query_vector, strict=True))
        scores.append(dot_product / (document_norm * query_norm))
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
