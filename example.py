from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rabbit_ai import RabbitAI
except ModuleNotFoundError as exc:
    if exc.name != "rabbit_ai":
        raise
    if not (ROOT / "rabbit_ai_combined.py").exists():
        raise RuntimeError(
            "rabbit_ai is not available. Deploy the 'rabbit_ai' folder or place "
            "'rabbit_ai_combined.py' next to this file."
        ) from exc
    from rabbit_ai_combined import RabbitAI


APP: RabbitAI | None = None
APP_INIT_ERROR: str | None = None
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rabbit AI Web Example</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4efe7;
      --panel: #fffaf3;
      --text: #1e1e1e;
      --muted: #6a6258;
      --accent: #b25b2a;
      --accent-dark: #8d431d;
      --border: #e4d8c6;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top, rgba(178, 91, 42, 0.16), transparent 28rem),
        linear-gradient(180deg, #f8f2e9, var(--bg));
      color: var(--text);
    }

    main {
      max-width: 56rem;
      margin: 0 auto;
      padding: 3rem 1.25rem 4rem;
    }

    .hero {
      margin-bottom: 1.5rem;
    }

    h1 {
      margin: 0;
      font-size: clamp(2.25rem, 7vw, 4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }

    .subhead {
      max-width: 38rem;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.6;
      margin-top: 0.9rem;
    }

    .panel {
      background: color-mix(in srgb, var(--panel) 88%, white);
      border: 1px solid var(--border);
      border-radius: 1.25rem;
      padding: 1rem;
      box-shadow: 0 18px 50px rgba(73, 47, 23, 0.08);
    }

    textarea {
      width: 100%;
      min-height: 9rem;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 0.9rem;
      padding: 1rem;
      font: inherit;
      background: #fffdfa;
      color: var(--text);
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.75rem;
      margin-top: 0.9rem;
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 0.8rem 1.2rem;
      background: var(--accent);
      color: white;
      font: inherit;
      cursor: pointer;
    }

    button:hover {
      background: var(--accent-dark);
    }

    label {
      color: var(--muted);
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }

    .status {
      min-height: 1.5rem;
      margin-top: 0.8rem;
      color: var(--muted);
    }

    .answer {
      margin-top: 1rem;
      white-space: pre-wrap;
      line-height: 1.65;
    }

    .meta {
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      color: var(--muted);
      display: grid;
      gap: 0.4rem;
    }

    ul {
      margin: 0.4rem 0 0;
      padding-left: 1.25rem;
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Rabbit AI</h1>
      <p class="subhead">
        Minimal Python web example for your package. Ask a question below and this page
        will call <code>RabbitAI.ask(...)</code> on the server.
      </p>
    </section>

    <section class="panel">
      <form id="ask-form">
        <textarea id="query" name="query" placeholder="Ask Rabbit AI something..." required></textarea>
        <div class="controls">
          <button type="submit">Ask Rabbit AI</button>
          <label>
            <input id="use-web" type="checkbox" checked>
            Use web search
          </label>
        </div>
      </form>

      <div class="status" id="status"></div>
      <div class="answer" id="answer"></div>
      <div class="meta" id="meta" hidden>
        <div id="confidence"></div>
        <div id="flags"></div>
        <div id="sources"></div>
      </div>
    </section>
  </main>

  <script>
    const form = document.getElementById("ask-form");
    const queryInput = document.getElementById("query");
    const useWebInput = document.getElementById("use-web");
    const statusEl = document.getElementById("status");
    const answerEl = document.getElementById("answer");
    const metaEl = document.getElementById("meta");
    const confidenceEl = document.getElementById("confidence");
    const flagsEl = document.getElementById("flags");
    const sourcesEl = document.getElementById("sources");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const query = queryInput.value.trim();
      if (!query) {
        statusEl.textContent = "Enter a question first.";
        return;
      }

      statusEl.textContent = "Thinking...";
      answerEl.textContent = "";
      metaEl.hidden = true;

      try {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            use_web: useWebInput.checked
          })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Request failed.");
        }

        statusEl.textContent = "";
        answerEl.textContent = data.text;
        confidenceEl.textContent = `Confidence: ${data.confidence.toFixed(2)} | Type: ${data.query_type}`;
        flagsEl.textContent = `Used memory: ${data.used_memory} | Used web: ${data.used_web}`;

        sourcesEl.replaceChildren();
        const sourcesTitle = document.createElement("strong");
        sourcesTitle.textContent = "Sources";
        sourcesEl.appendChild(sourcesTitle);

        if (data.sources.length) {
          const list = document.createElement("ul");
          for (const source of data.sources) {
            const item = document.createElement("li");
            item.textContent = source;
            list.appendChild(item);
          }
          sourcesEl.appendChild(list);
        } else {
          const empty = document.createElement("div");
          empty.textContent = "None";
          sourcesEl.appendChild(empty);
        }

        metaEl.hidden = false;
      } catch (error) {
        statusEl.textContent = error.message;
      }
    });
  </script>
</body>
</html>
"""


class RabbitWebHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        body = HTML_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        app = get_app()
        if app is None:
            self._send_json(
                {"error": APP_INIT_ERROR or "Rabbit AI failed to initialize."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, HTTPStatus.BAD_REQUEST)
            return

        query = str(payload.get("query", "")).strip()
        use_web = bool(payload.get("use_web", True))

        if not query:
            self._send_json({"error": "Query is required."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            answer = app.ask(query, use_web=use_web)
        except Exception as exc:
            self._send_json({"error": f"Rabbit AI failed to process the request: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(
            {
                "text": answer.text,
                "sources": answer.sources,
                "confidence": answer.confidence,
                "used_memory": answer.used_memory,
                "used_web": answer.used_web,
                "query_type": answer.query_type,
            },
            HTTPStatus.OK,
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: dict[str, object], status: HTTPStatus) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def get_app() -> RabbitAI | None:
    global APP
    global APP_INIT_ERROR

    if APP is not None:
        return APP
    if APP_INIT_ERROR is not None:
        return None

    try:
        APP = RabbitAI()
    except Exception as exc:
        APP_INIT_ERROR = str(exc)
        return None
    return APP


def run() -> None:
    server = ThreadingHTTPServer((HOST, PORT), RabbitWebHandler)
    print(f"Rabbit AI web example running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
