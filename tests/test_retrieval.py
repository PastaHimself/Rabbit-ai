from __future__ import annotations

import unittest

from rabbit_ai.retrieval import Ranker, SimpleTfidfVectorizer, cosine_similarity, normalize_text, tokenize
from rabbit_ai.types import Passage


class RetrievalTests(unittest.TestCase):
    def test_tokenize_and_normalize_remove_stopwords(self) -> None:
        self.assertEqual(tokenize("What is the best use of Rabbit AI today?"), ["best", "use", "rabbit", "ai", "today"])
        self.assertEqual(normalize_text("Rabbit AI, Rabbit AI!"), "rabbit ai rabbit ai")

    def test_tfidf_and_cosine_similarity(self) -> None:
        vectorizer = SimpleTfidfVectorizer()
        matrix = vectorizer.fit_transform(
            [
                "rabbit ai web search",
                "rabbit ai uses web search",
                "gardening tips and soil",
            ]
        )
        scores = cosine_similarity(matrix[0], matrix[1:])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], 0.4)

    def test_ranker_is_deterministic(self) -> None:
        passages = [
            Passage(
                title="Rabbit AI web search",
                url="https://example.com/search",
                text="Rabbit AI uses web search, ranking, and caching for fast answers.",
                source="duckduckgo",
                rank=0,
            ),
            Passage(
                title="General assistant note",
                url="https://example.com/other",
                text="This page is about unrelated household maintenance tasks.",
                source="duckduckgo",
                rank=1,
            ),
        ]
        ranker = Ranker()
        ranked_once = ranker.rank("rabbit ai web search", passages)
        ranked_twice = ranker.rank("rabbit ai web search", passages)
        self.assertEqual([item.url for item in ranked_once], [item.url for item in ranked_twice])
        self.assertEqual(ranked_once[0].url, "https://example.com/search")


if __name__ == "__main__":
    unittest.main()
