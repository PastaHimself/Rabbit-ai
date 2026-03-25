# Rabbit AI

## Architecture Options

### 1. Pipeline-first
- Flow: classify query -> search -> fetch -> rank -> answer.
- Strength: easiest to debug because each stage is isolated.
- Tradeoff: weaker reuse because memory is added late.

### 2. Retrieval-first
- Flow: pull candidate evidence from local memory and the web, rank everything together, then compose a concise answer.
- Strength: best quality-to-compute ratio for a CPU-only assistant.
- Tradeoff: ranking and evidence management need slightly more code.

### 3. Memory-first
- Flow: answer from local memory unless confidence is low or the query is time-sensitive.
- Strength: very fast after warmup.
- Tradeoff: cold-start quality is weaker, and stale answers are a bigger risk.

## Recommended Design

Rabbit AI uses the retrieval-first hybrid because it maximizes intelligence per compute. The engine first recalls similar local answers, then optionally searches the web, turns both into passages, ranks them with a lightweight TF-IDF scorer, and uses rule-based templates to produce a concise answer.

Locked behavior:
- `RabbitAI.ask(query, use_web=True) -> Answer`
- Memory answers are reused directly only when similarity is `>= 0.82` and the query is not time-sensitive.
- Ranking score = `0.55 * cosine + 0.20 * title overlap + 0.15 * search-rank prior + 0.10 * source-quality bonus`
- Web fallback chain = DuckDuckGo HTML/Lite -> dense-keyword retry -> Wikipedia HTML search

## Environment Setup

This workspace currently does not have a working `python` executable on `PATH`, so start there.

1. Install Python 3.11 or newer on Windows.
```powershell
winget install Python.Python.3.11
```

2. Create and activate a virtual environment.
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install Rabbit AI from this local checkout.
```powershell
pip install -r requirements.txt
pip install --no-deps --no-build-isolation -e .
```

Rabbit AI now uses a PyTorch backend for retrieval and ranking. The package install needs `torch`, and the single-file fallback path also keeps `requests` and `beautifulsoup4`.

Do not run `pip install rabbit-ai` on a private or offline server unless you have mirrored the package to your own index. That command asks `pip` to fetch `rabbit-ai` from the configured package index, which is why you see retries against `pypi.org`.

If you prefer a one-step Windows install for private servers, run:
```powershell
.\install_private.ps1
```

4. Run the CLI assistant.
```powershell
rabbit-ai
```

You can also run it as a module:
```powershell
python -m rabbit_ai.cli
```

5. Run the included tests.
```powershell
python -m unittest discover -s tests -v
```

## Module-by-Module Implementation

### 1. `config/types`
These files define the stable interfaces and the CPU-only runtime defaults.

Files:
- `rabbit_ai/config.py`
- `rabbit_ai/types.py`

```python
from rabbit_ai.config import DEFAULT_CONFIG, RabbitConfig
from rabbit_ai.types import Answer, MemoryRecord, Passage, SearchResult

config = RabbitConfig()
answer = Answer(text="Short answer", confidence=0.7)
```

### 2. `retrieval`
This module implements tokenization, TF-IDF, cosine similarity, keyword overlap, query rewriting, and the `Ranker`.

File:
- `rabbit_ai/retrieval.py`

```python
from rabbit_ai.retrieval import Ranker, dense_keyword_query, tokenize

tokens = tokenize("Rabbit AI uses fast retrieval and ranking.")
query = dense_keyword_query("What is the latest Rabbit AI design?")
ranked = Ranker().rank(query, passages)
```

### 3. `memory`
This module stores interactions, search-result cache entries, and fetched-page cache entries in `sqlite3`.

File:
- `rabbit_ai/memory.py`

```python
from rabbit_ai.memory import MemoryStore

store = MemoryStore("rabbit_ai.db")
store.save_interaction("What is Rabbit AI?", answer)
memories = store.recall("What is Rabbit AI?")
```

### 4. `search`
This module provides the pluggable search interface, DuckDuckGo HTML/Lite parsing, Wikipedia fallback search, page fetching, HTML cleanup, and passage extraction.

File:
- `rabbit_ai/search.py`

```python
from rabbit_ai.search import DuckDuckGoSearchProvider, PageFetcher

provider = DuckDuckGoSearchProvider()
results = provider.search("rabbit ai python", max_results=5)
passages = PageFetcher().fetch_passages("rabbit ai python", results[0])
```

### 5. `reasoning`
This module classifies query intent and composes concise answers from ranked passages using deterministic templates.

File:
- `rabbit_ai/reasoning.py`

```python
from rabbit_ai.reasoning import Reasoner

reasoner = Reasoner()
query_type = reasoner.classify("Compare HTTP vs HTTPS")
answer = reasoner.compose(query, ranked_passages, memories)
```

### 6. `engine`
This is the orchestrator. It joins memory, search, ranking, and answer composition behind the public `RabbitAI` interface.

File:
- `rabbit_ai/engine.py`

```python
from rabbit_ai.engine import RabbitAI

rabbit = RabbitAI()
answer = rabbit.ask("What is Python?", use_web=True)
print(answer.text)
```

### 7. `cli`
This module exposes the terminal assistant with `exit`, `sources on`, `sources off`, `clear-cache`, and `stats`.

File:
- `rabbit_ai/cli.py`

```python
from rabbit_ai.cli import run_cli

run_cli()
```

### 8. `evaluation`
This module runs benchmark queries and reports keyword-hit score, latency, confidence, and cache-friendly reuse behavior.

File:
- `rabbit_ai/evaluation.py`

```python
from rabbit_ai.engine import RabbitAI
from rabbit_ai.evaluation import evaluate_agent, summarize_report

report = evaluate_agent(RabbitAI(), use_web=True)
summary = summarize_report(report)
```

## Evaluation

The project includes:
- Example benchmark queries in `rabbit_ai/evaluation.py`
- `keyword_hit_score` for rough relevance checking
- Latency measurements per query
- Confidence tracking from the rule-based response engine
- Cache and memory reuse through `RabbitAI.stats()`

Expected test coverage:
- Retrieval math and ranking determinism
- SQLite schema, TTL expiration, and recall
- DuckDuckGo parser and HTML text cleaning
- End-to-end engine behavior for memory reuse, fallback search, and conservative failure handling

## Final Combined Script

If you want a single-file build instead of the package layout, use `rabbit_ai_combined.py`.

```powershell
python rabbit_ai_combined.py
```

The single-file script is a convenience snapshot and may lag behind the modular package. It includes:
- Config and dataclasses
- Retrieval and ranking
- SQLite memory
- DuckDuckGo and Wikipedia search
- Rule-based reasoning
- CLI and evaluation helpers

## Lightweight Correctness Check

Completed:
- Kept the modular package as the primary supported install target
- Added `unittest` coverage for retrieval, memory, search parsing, reasoning, and engine fallbacks
- Switched the retrieval and ranking backend to PyTorch
- Confirmed there is no required GPU usage, no TensorFlow, and no external AI API dependency

Blocked in this workspace:
- Running `python -m unittest ...`
- Running a live CLI session
- Running a syntax/import check with `python -m py_compile`

Reason:
- `python` is not currently installed or not available on `PATH` in this environment

## Packaging

Rabbit AI is now installable as a Python package with:
- Package metadata in `pyproject.toml`
- CLI entry point: `rabbit-ai`
- Module entry point: `python -m rabbit_ai`

To build distributable artifacts later:
```powershell
python -m build
```
