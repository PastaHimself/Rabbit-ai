from __future__ import annotations

import abc
import re
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import SearchConfig
from .retrieval import top_relevant_chunks
from .types import Passage, SearchResult


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
