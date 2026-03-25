from __future__ import annotations

import tempfile
import unittest

from rabbit_ai.memory import MemoryStore
from rabbit_ai.types import Answer, SearchResult


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(db_path=f"{self.temp_dir.name}/rabbit_ai.db", cache_ttl_hours=24)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_and_recall_interaction(self) -> None:
        answer = Answer(
            text="Rabbit AI is a lightweight assistant that mixes search with reasoning.",
            sources=["https://example.com/rabbit"],
            confidence=0.81,
            query_type="definition",
        )
        self.store.save_interaction("What is Rabbit AI?", answer)

        memories = self.store.recall("What is Rabbit AI?", limit=3)
        self.assertEqual(len(memories), 1)
        self.assertGreaterEqual(memories[0].similarity, 0.99)
        self.assertIn("lightweight assistant", memories[0].answer)

    def test_search_cache_ttl_expiry(self) -> None:
        self.store.cache_search(
            "python",
            "duckduckgo",
            [SearchResult(title="Python", url="https://python.org", rank=0)],
        )
        cached = self.store.get_cached_search("python", "duckduckgo")
        self.assertIsNotNone(cached)

        with self.store._connect() as connection:
            connection.execute("UPDATE search_cache SET created_at = 0")

        expired = self.store.get_cached_search("python", "duckduckgo")
        self.assertIsNone(expired)

    def test_page_cache_roundtrip_and_clear(self) -> None:
        self.store.cache_page("https://example.com", "Example", "Useful text", "duckduckgo")
        cached_page = self.store.get_cached_page("https://example.com")
        self.assertIsNotNone(cached_page)
        self.assertEqual(cached_page["title"], "Example")

        self.store.clear_cache()
        self.assertIsNone(self.store.get_cached_search("missing", "duckduckgo"))
        self.assertIsNone(self.store.get_cached_page("https://example.com"))


if __name__ == "__main__":
    unittest.main()

