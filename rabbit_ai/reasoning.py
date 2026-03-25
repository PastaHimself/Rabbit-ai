from __future__ import annotations

import re
from collections import OrderedDict

from .config import RuntimeConfig
from .retrieval import keyword_overlap_score
from .types import Answer, MemoryRecord, Passage

TIME_SENSITIVE_PATTERN = re.compile(r"\b(latest|today|current|now|recent|news|this year|yesterday|tomorrow|202\d|203\d)\b")


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
