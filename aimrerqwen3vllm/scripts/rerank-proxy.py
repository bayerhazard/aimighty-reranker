#!/usr/bin/env python3
"""rerank-proxy.py - Cohere-compatible rerank proxy that applies the
Qwen3-Reranker instruction template before calling vLLM.

Why: vLLM 0.29 implements no score template for
`Qwen3ForSequenceClassification` (only `jina_vl` does), so its `/rerank`
endpoint concatenates `query` + `document` raw. `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
was trained with the Qwen3-Reranker prompt, so raw input yields
miscalibrated scores (~0.30 relevant / 0.12 irrelevant instead of
~0.9997 / 0.0001). RAGFlow's blend replaces the embedding term with the
rerank score, so miscalibrated scores demote relevant chunks.

This proxy accepts RAGFlow's Cohere-style request:
    POST /v1/rerank {"model","query","documents":[...],"top_n":N}
and forwards it to vLLM `/rerank` with the query and each document already
wrapped in the Qwen3-Reranker template (vLLM then only concatenates them,
which produces the exact expected sequence).

Stdlib only - no pip dependencies.
Env:
  VLLM_URL   upstream base, default http://127.0.0.1:30000/v1
  PORT       listen port, default 8080
  RERANK_INSTRUCT  instruction line (default generic English)
"""
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:30000/v1").rstrip("/")
PORT = int(os.environ.get("PORT", "8080"))
INSTRUCT = os.environ.get(
    "RERANK_INSTRUCT",
    "Given a web search query, retrieve relevant passages that answer the query",
)

PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def fmt_query(q: str) -> str:
    return f"{PREFIX}<Instruct>: {INSTRUCT}\n<Query>: {q}\n"


def fmt_document(d: str) -> str:
    return f"<Document>: {d}{SUFFIX}"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "rerank-template-proxy/1.0"

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep logs terse
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("/health", "/healthz"):
            return self._json(200, {"status": "ok"})
        if path in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": []})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path not in ("/v1/rerank", "/rerank"):
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"error": f"bad body: {e}"})

        query = body.get("query") or ""
        documents = body.get("documents") or []
        if not query or not documents:
            return self._json(400, {"error": "query and documents are required"})
        model = body.get("model") or "Qwen/Qwen3-Reranker-0.6B"
        top_n = body.get("top_n") or len(documents)

        upstream_body = {
            "model": model,
            "query": fmt_query(query),
            "documents": [fmt_document(d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)) for d in documents],
        }
        req = urllib.request.Request(
            VLLM_URL + "/rerank",
            data=json.dumps(upstream_body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer proxy"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                res = json.load(r)
        except urllib.error.HTTPError as e:
            return self._json(502, {"error": f"upstream {e.code}: {e.read().decode()[:300]}"})
        except Exception as e:  # noqa: BLE001
            return self._json(502, {"error": f"upstream: {type(e).__name__}: {e}"})

        results = []
        for item in res.get("results", []):
            idx = item.get("index", 0)
            results.append({
                "index": idx,
                "relevance_score": item.get("relevance_score", 0.0),
                "document": {"text": documents[idx] if isinstance(documents[idx], str) else ""},
            })
        results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        if top_n:
            results = results[:top_n]
        return self._json(200, {
            "id": res.get("id", ""),
            "model": model,
            "results": results,
            "usage": res.get("usage", {}),
        })


if __name__ == "__main__":
    print(f"rerank-template-proxy listening on :{PORT} -> {VLLM_URL}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
