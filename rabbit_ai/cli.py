from __future__ import annotations

import json

from .engine import RabbitAI


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

