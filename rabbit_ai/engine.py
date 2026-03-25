from __future__ import annotations

from .config import DEFAULT_CONFIG, RabbitConfig
from .memory import MemoryStore
from .reasoning import Reasoner
from .retrieval import Ranker, dense_keyword_query, normalize_text
from .search import DuckDuckGoSearchProvider, PageFetcher, SearchProvider, WikipediaSearchProvider
from .types import Answer, Passage, SearchResult


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

        direct_answer = self.reasoner.try_direct_answer(query)
        if direct_answer is not None:
            self.memory.save_interaction(query, direct_answer)
            return direct_answer

        query_type = self.reasoner.classify(query)
        memories = self.memory.recall(query, limit=5)

        if (
            query_type != "time_sensitive"
            and memories
            and memories[0].similarity >= self.config.runtime.memory_reuse_threshold
            and memories[0].confidence >= 0.45
            and not self.reasoner.is_low_signal_answer(memories[0].answer)
        ):
            top_memory = memories[0]
            answer = Answer(
                text=top_memory.answer,
                sources=top_memory.sources,
                confidence=round(min(0.92, 0.7 * top_memory.confidence + 0.3 * top_memory.similarity), 3),
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

    def _memory_support_passages(self, memories: list) -> list[Passage]:
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
                title=cached_page["title"],
                url=result.url,
                text=cached_page["text"],
                source=cached_page["source"],
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
