from __future__ import annotations

import tempfile
import unittest

from rabbit_ai.config import RabbitConfig, RuntimeConfig, SearchConfig
from rabbit_ai.engine import RabbitAI
from rabbit_ai.memory import MemoryStore
from rabbit_ai.reasoning import Reasoner
from rabbit_ai.search import DuckDuckGoSearchProvider, PageFetcher, SearchProvider
from rabbit_ai.types import Passage, SearchResult


SAMPLE_DDG_HTML = """
<html>
  <body>
    <div class="result">
      <a class="result__a" href="https://example.com/rabbit">Rabbit AI overview</a>
      <a class="result__snippet">Rabbit AI combines search and lightweight reasoning.</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://example.com/python">Python docs</a>
      <a class="result__snippet">Official Python documentation.</a>
    </div>
  </body>
</html>
"""

SAMPLE_ARTICLE_HTML = """
<html>
  <head><title>Rabbit AI Guide</title></head>
  <body>
    <script>ignore_me()</script>
    <main>
      <h1>Rabbit AI Guide</h1>
      <p>Rabbit AI is a compact assistant that uses retrieval and rule-based reasoning.</p>
      <p>It favors CPU-friendly components such as TF-IDF, keyword scoring, and cached search.</p>
    </main>
  </body>
</html>
"""


class FakeSearchProvider(SearchProvider):
    def __init__(self, result_map: dict[str, list[SearchResult]]) -> None:
        self.result_map = result_map
        self.calls = 0
        self.name = "fake-search"

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.calls += 1
        return self.result_map.get(query, [])[:max_results]


class FakePageFetcher:
    def __init__(self, passage_map: dict[str, list[Passage]]) -> None:
        self.passage_map = passage_map
        self.calls = 0

    def fetch_passages(
        self,
        query: str,
        result: SearchResult,
        *,
        max_passages: int = 3,
        min_length: int = 80,
    ) -> list[Passage]:
        self.calls += 1
        return self.passage_map.get(result.url, [])

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
        return [
            Passage(
                title=title,
                url=url,
                text=text,
                source=source,
                rank=rank,
            )
        ]


class SearchReasoningEngineTests(unittest.TestCase):
    def test_duckduckgo_html_parser(self) -> None:
        results = DuckDuckGoSearchProvider._parse_html_results(SAMPLE_DDG_HTML)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].url, "https://example.com/rabbit")
        self.assertIn("lightweight reasoning", results[0].snippet)

    def test_page_cleaning_extracts_meaningful_text(self) -> None:
        title, text = PageFetcher._extract_text(SAMPLE_ARTICLE_HTML, "https://example.com/rabbit")
        self.assertEqual(title, "Rabbit AI Guide")
        self.assertNotIn("ignore_me", text)
        self.assertIn("retrieval and rule-based reasoning", text)

    def test_reasoner_classifies_query_types(self) -> None:
        reasoner = Reasoner(RuntimeConfig())
        self.assertEqual(reasoner.classify("what is 1 plus 1"), "calculation")
        self.assertEqual(reasoner.classify("latest Rabbit AI release"), "time_sensitive")
        self.assertEqual(reasoner.classify("Compare HTTP vs HTTPS"), "comparison")
        self.assertEqual(reasoner.classify("How do I boil pasta?"), "how_to")
        self.assertEqual(reasoner.classify("What is Rabbit AI?"), "definition")

    def test_engine_solves_simple_math_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = RabbitAI(
                config=RabbitConfig(db_path=f"{temp_dir}/rabbit.db", search=SearchConfig()),
                memory_store=MemoryStore(db_path=f"{temp_dir}/rabbit.db", cache_ttl_hours=24),
                search_provider=FakeSearchProvider({}),
                fallback_provider=FakeSearchProvider({}),
                page_fetcher=FakePageFetcher({}),
            )
            answer = agent.ask("what is 1 plus 1", use_web=True)
            self.assertEqual(answer.text, "The answer is 2.")
            self.assertFalse(answer.used_memory)
            self.assertFalse(answer.used_web)
            self.assertEqual(answer.query_type, "calculation")

    def test_engine_reuses_memory_for_repeated_non_time_sensitive_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            search_provider = FakeSearchProvider(
                {
                    "What is Rabbit AI?": [
                        SearchResult(title="Rabbit AI overview", url="https://example.com/rabbit", rank=0, source="fake-search")
                    ]
                }
            )
            page_fetcher = FakePageFetcher(
                {
                    "https://example.com/rabbit": [
                        Passage(
                            title="Rabbit AI overview",
                            url="https://example.com/rabbit",
                            text="Rabbit AI is a compact assistant that mixes search, ranking, and cached memory for CPU-friendly answers.",
                            source="fake-search",
                            rank=0,
                        )
                    ]
                }
            )
            agent = RabbitAI(
                config=RabbitConfig(db_path=f"{temp_dir}/rabbit.db", search=SearchConfig()),
                memory_store=MemoryStore(db_path=f"{temp_dir}/rabbit.db", cache_ttl_hours=24),
                search_provider=search_provider,
                fallback_provider=FakeSearchProvider({}),
                page_fetcher=page_fetcher,
            )

            first = agent.ask("What is Rabbit AI?", use_web=True)
            second = agent.ask("What is Rabbit AI?", use_web=True)

            self.assertTrue(first.used_web)
            self.assertTrue(second.used_memory)
            self.assertFalse(second.used_web)
            self.assertEqual(search_provider.calls, 1)

    def test_engine_does_not_reuse_low_confidence_failure_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(db_path=f"{temp_dir}/rabbit.db", cache_ttl_hours=24)
            store.save_interaction(
                "What is Rabbit AI?",
                Answer(
                    text="I could not find enough reliable information to answer that confidently.",
                    confidence=0.1,
                    query_type="definition",
                ),
            )
            search_provider = FakeSearchProvider(
                {
                    "What is Rabbit AI?": [
                        SearchResult(title="Rabbit AI overview", url="https://example.com/rabbit", rank=0, source="fake-search")
                    ]
                }
            )
            page_fetcher = FakePageFetcher(
                {
                    "https://example.com/rabbit": [
                        Passage(
                            title="Rabbit AI overview",
                            url="https://example.com/rabbit",
                            text="Rabbit AI is a compact assistant that mixes search, ranking, and cached memory for CPU-friendly answers.",
                            source="fake-search",
                            rank=0,
                        )
                    ]
                }
            )
            agent = RabbitAI(
                config=RabbitConfig(db_path=f"{temp_dir}/rabbit.db", search=SearchConfig()),
                memory_store=store,
                search_provider=search_provider,
                fallback_provider=FakeSearchProvider({}),
                page_fetcher=page_fetcher,
            )
            answer = agent.ask("What is Rabbit AI?", use_web=True)
            self.assertNotIn("could not find enough reliable information", answer.text.lower())
            self.assertTrue(answer.used_web)

    def test_engine_uses_fallback_provider_when_primary_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            primary = FakeSearchProvider({})
            fallback = FakeSearchProvider(
                {
                    "Rabbit AI summary": [
                        SearchResult(title="Rabbit AI fallback", url="https://example.com/fallback", rank=0, source="wikipedia")
                    ]
                }
            )
            page_fetcher = FakePageFetcher(
                {
                    "https://example.com/fallback": [
                        Passage(
                            title="Rabbit AI fallback",
                            url="https://example.com/fallback",
                            text="Rabbit AI combines lightweight retrieval, memory reuse, and fast rule-based reasoning.",
                            source="wikipedia",
                            rank=0,
                        )
                    ]
                }
            )
            agent = RabbitAI(
                config=RabbitConfig(db_path=f"{temp_dir}/rabbit.db", search=SearchConfig()),
                memory_store=MemoryStore(db_path=f"{temp_dir}/rabbit.db", cache_ttl_hours=24),
                search_provider=primary,
                fallback_provider=fallback,
                page_fetcher=page_fetcher,
            )

            answer = agent.ask("Rabbit AI summary", use_web=True)
            self.assertTrue(answer.used_web)
            self.assertIn("lightweight retrieval", answer.text.lower())
            self.assertEqual(primary.calls, 1)
            self.assertEqual(fallback.calls, 1)

    def test_time_sensitive_query_still_uses_web_on_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            search_provider = FakeSearchProvider(
                {
                    "latest Rabbit AI release": [
                        SearchResult(title="Rabbit AI latest", url="https://example.com/latest", rank=0, source="fake-search")
                    ]
                }
            )
            page_fetcher = FakePageFetcher(
                {
                    "https://example.com/latest": [
                        Passage(
                            title="Rabbit AI latest",
                            url="https://example.com/latest",
                            text="The latest Rabbit AI release focuses on lightweight retrieval, caching, and CPU-only execution.",
                            source="fake-search",
                            rank=0,
                        )
                    ]
                }
            )
            agent = RabbitAI(
                config=RabbitConfig(db_path=f"{temp_dir}/rabbit.db", search=SearchConfig()),
                memory_store=MemoryStore(db_path=f"{temp_dir}/rabbit.db", cache_ttl_hours=24),
                search_provider=search_provider,
                fallback_provider=FakeSearchProvider({}),
                page_fetcher=page_fetcher,
            )

            first = agent.ask("latest Rabbit AI release", use_web=True)
            second = agent.ask("latest Rabbit AI release", use_web=True)

            self.assertTrue(first.used_web)
            self.assertTrue(second.used_web)
            self.assertIn("https://example.com/latest", second.sources)

    def test_engine_answers_conservatively_when_no_evidence_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = RabbitAI(
                config=RabbitConfig(db_path=f"{temp_dir}/rabbit.db", search=SearchConfig()),
                memory_store=MemoryStore(db_path=f"{temp_dir}/rabbit.db", cache_ttl_hours=24),
                search_provider=FakeSearchProvider({}),
                fallback_provider=FakeSearchProvider({}),
                page_fetcher=FakePageFetcher({}),
            )
            answer = agent.ask("Unanswerable test prompt", use_web=True)
            self.assertIn("could not find enough reliable information", answer.text.lower())
            self.assertLessEqual(answer.confidence, 0.2)


if __name__ == "__main__":
    unittest.main()
