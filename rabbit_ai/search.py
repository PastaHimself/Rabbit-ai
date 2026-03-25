from __future__ import annotations

import abc
import re
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
from urllib.request import OpenerDirector, Request, build_opener

from .config import SearchConfig
from .retrieval import top_relevant_chunks
from .types import Passage, SearchResult


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _class_tokens(attrs: dict[str, str]) -> set[str]:
    return {token for token in attrs.get("class", "").split() if token}


def _attrs_to_dict(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
    return {key: value or "" for key, value in attrs}


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._current_href = ""
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._current_href = _attrs_to_dict(attrs).get("href", "")
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href:
            text = _clean_spaces("".join(self._current_text))
            self.links.append((self._current_href, text))
            self._current_href = ""
            self._current_text = []


class _DuckDuckGoHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._seen_urls: set[str] = set()
        self._result_depth = 0
        self._link_depth = 0
        self._snippet_depth = 0
        self._current_href = ""
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = _attrs_to_dict(attrs)
        classes = _class_tokens(attr_map)
        if tag == "div" and "result" in classes:
            if self._result_depth == 0:
                self._current_href = ""
                self._current_title = []
                self._current_snippet = []
            self._result_depth += 1
            return

        if self._result_depth == 0:
            return

        if tag == "a":
            href = attr_map.get("href", "")
            if "result__a" in classes or (href and not self._current_href):
                self._link_depth += 1
                if not self._current_href:
                    self._current_href = href
                    self._current_title = []
            if "result__snippet" in classes or "result__extras__url" in classes:
                self._snippet_depth += 1
            return

        if "result__snippet" in classes or "result__extras__url" in classes:
            self._snippet_depth += 1

    def handle_data(self, data: str) -> None:
        if self._link_depth > 0:
            self._current_title.append(data)
        if self._snippet_depth > 0:
            self._current_snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._result_depth == 0:
            return

        if tag == "a":
            if self._link_depth > 0:
                self._link_depth -= 1
            if self._snippet_depth > 0:
                self._snippet_depth -= 1
            return

        if self._snippet_depth > 0 and tag in {"span", "div"}:
            self._snippet_depth -= 1

        if tag == "div":
            self._result_depth -= 1
            if self._result_depth == 0:
                self._finalize_result()

    def _finalize_result(self) -> None:
        title = _clean_spaces("".join(self._current_title))
        snippet = _clean_spaces("".join(self._current_snippet))
        url = DuckDuckGoSearchProvider._unwrap_url(self._current_href)
        if title and DuckDuckGoSearchProvider._is_external_url(url) and url not in self._seen_urls:
            self._seen_urls.add(url)
            self.results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    rank=len(self.results),
                    source=DuckDuckGoSearchProvider.name,
                )
            )
        self._current_href = ""
        self._current_title = []
        self._current_snippet = []


class _WikipediaSearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.heading = ""
        self.first_paragraph = ""
        self.results: list[SearchResult] = []
        self._capture_heading = False
        self._capture_paragraph = False
        self._paragraph_parts: list[str] = []
        self._in_result_item = False
        self._current_link = ""
        self._current_title_parts: list[str] = []
        self._current_snippet_parts: list[str] = []
        self._capture_title = False
        self._capture_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = _attrs_to_dict(attrs)
        classes = _class_tokens(attr_map)
        if tag == "h1" and attr_map.get("id") == "firstHeading":
            self._capture_heading = True
            return
        if tag == "p" and not self.first_paragraph:
            self._capture_paragraph = True
            self._paragraph_parts = []
            return
        if tag == "li" and "mw-search-result" in classes:
            self._in_result_item = True
            self._current_link = ""
            self._current_title_parts = []
            self._current_snippet_parts = []
            return
        if not self._in_result_item:
            return
        if tag == "a" and not self._current_link:
            self._current_link = attr_map.get("href", "")
            self._capture_title = True
            return
        if tag == "div" and "searchresult" in classes:
            self._capture_snippet = True

    def handle_data(self, data: str) -> None:
        if self._capture_heading:
            self.heading += data
        if self._capture_paragraph:
            self._paragraph_parts.append(data)
        if self._capture_title:
            self._current_title_parts.append(data)
        if self._capture_snippet:
            self._current_snippet_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self._capture_heading:
            self.heading = _clean_spaces(self.heading)
            self._capture_heading = False
            return
        if tag == "p" and self._capture_paragraph:
            paragraph = _clean_spaces("".join(self._paragraph_parts))
            if paragraph and not self.first_paragraph:
                self.first_paragraph = paragraph
            self._capture_paragraph = False
            self._paragraph_parts = []
            return
        if tag == "a" and self._capture_title:
            self._capture_title = False
            return
        if tag == "div" and self._capture_snippet:
            self._capture_snippet = False
            return
        if tag == "li" and self._in_result_item:
            title = _clean_spaces("".join(self._current_title_parts))
            snippet = _clean_spaces("".join(self._current_snippet_parts))
            if title and self._current_link:
                self.results.append(
                    SearchResult(
                        title=title,
                        url=urljoin("https://en.wikipedia.org/", self._current_link),
                        snippet=snippet,
                        rank=len(self.results),
                        source=WikipediaSearchProvider.name,
                    )
                )
            self._in_result_item = False


class _ContentExtractor(HTMLParser):
    EXCLUDED_TAGS = {"script", "style", "noscript", "svg", "img", "header", "footer", "nav", "aside", "form"}
    PREFERRED_CONTAINERS = {"main", "article", "body"}
    TEXT_TAGS = {"p", "li", "h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._title_parts: list[str] = []
        self._capture_title = False
        self._container_stack: list[str] = []
        self._ignore_depth = 0
        self._collect_text = False
        self._text_parts: list[str] = []
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._capture_title = True
            self._title_parts = []
            return

        if tag in self.PREFERRED_CONTAINERS:
            self._container_stack.append(tag)

        if tag in self.EXCLUDED_TAGS:
            self._ignore_depth += 1
            return

        if self._ignore_depth == 0 and self._container_stack and tag in self.TEXT_TAGS:
            self._collect_text = True
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._title_parts.append(data)
        if self._collect_text and self._ignore_depth == 0:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.title = _clean_spaces("".join(self._title_parts))
            self._capture_title = False
            self._title_parts = []
            return

        if tag in self.EXCLUDED_TAGS and self._ignore_depth > 0:
            self._ignore_depth -= 1
            return

        if self._collect_text and tag in self.TEXT_TAGS:
            text = _clean_spaces("".join(self._text_parts))
            if len(text) >= 40:
                self.blocks.append(text)
            self._collect_text = False
            self._text_parts = []

        if tag in self.PREFERRED_CONTAINERS and self._container_stack:
            self._container_stack.pop()


class SearchProvider(abc.ABC):
    name = "search"

    @abc.abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class _BaseHttpClient:
    def __init__(self, config: SearchConfig | None = None, opener: OpenerDirector | None = None) -> None:
        self.config = config or SearchConfig()
        self.opener = opener or build_opener()
        self.headers = {
            "User-Agent": self.config.user_agent,
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _get(self, url: str, params: dict[str, str] | None = None) -> tuple[str, str, str]:
        if params:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urlencode(params)}"

        request = Request(url, headers=self.headers)
        try:
            with self.opener.open(request, timeout=self.config.timeout_seconds) as response:
                content_type = response.headers.get("content-type", "")
                final_url = response.geturl()
                body = response.read().decode("utf-8", errors="ignore")
                return final_url, body, content_type
        except (HTTPError, URLError, TimeoutError, OSError):
            return "", "", ""


class DuckDuckGoSearchProvider(_BaseHttpClient, SearchProvider):
    name = "duckduckgo"
    HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
    LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        for endpoint, parser in (
            (self.HTML_ENDPOINT, self._parse_html_results),
            (self.LITE_ENDPOINT, self._parse_lite_results),
        ):
            _, html, _ = self._get(endpoint, {"q": query})
            if not html:
                continue
            results = parser(html)
            if results:
                return results[:max_results]
        return []

    @staticmethod
    def _looks_like_challenge(html: str) -> bool:
        lowered = html.lower()
        return "anomaly.js" in lowered or "challenge-form" in lowered or "botnet" in lowered

    @classmethod
    def _parse_html_results(cls, html: str) -> list[SearchResult]:
        if cls._looks_like_challenge(html):
            return []

        parser = _DuckDuckGoHtmlParser()
        parser.feed(html)
        if parser.results:
            return parser.results

        collector = _AnchorCollector()
        collector.feed(html)
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for href, title in collector.links:
            url = cls._unwrap_url(href)
            if not title or not cls._is_external_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(SearchResult(title=title, url=url, rank=len(results), source=cls.name))
            if len(results) >= 10:
                break
        return results

    @classmethod
    def _parse_lite_results(cls, html: str) -> list[SearchResult]:
        if cls._looks_like_challenge(html):
            return []

        collector = _AnchorCollector()
        collector.feed(html)
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for href, title in collector.links:
            url = cls._unwrap_url(href)
            if not title or not cls._is_external_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(SearchResult(title=title, url=url, rank=len(results), source=cls.name))
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


class WikipediaSearchProvider(_BaseHttpClient, SearchProvider):
    name = "wikipedia"
    SEARCH_ENDPOINT = "https://en.wikipedia.org/w/index.php"

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        final_url, html, _ = self._get(
            self.SEARCH_ENDPOINT,
            {"search": query, "title": "Special:Search", "ns0": "1"},
        )
        if not html:
            return []

        parser = _WikipediaSearchParser()
        parser.feed(html)
        if parser.heading and "wikipedia.org/wiki/" in final_url:
            return [
                SearchResult(
                    title=parser.heading,
                    url=final_url,
                    snippet=parser.first_paragraph,
                    rank=0,
                    source=self.name,
                )
            ]
        return parser.results[:max_results]


class PageFetcher(_BaseHttpClient):
    def fetch(self, url: str) -> tuple[str, str]:
        final_url, html, content_type = self._get(url)
        if "html" not in content_type and "text/plain" not in content_type:
            return "", ""
        return self._extract_text(html, final_url or url)

    @staticmethod
    def _extract_text(html: str, url: str) -> tuple[str, str]:
        parser = _ContentExtractor()
        parser.feed(html)
        title = parser.title or url
        text = _clean_spaces("\n".join(parser.blocks))
        return title, text

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
