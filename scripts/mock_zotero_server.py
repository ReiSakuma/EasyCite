#!/usr/bin/env python3
"""Mock Zotero HTTP endpoint for manually testing EasyCite v2.

Run from the v2 project root:

    .venv/bin/python scripts/mock_zotero_server.py

Then configure VS Code:

    citation-agent.zoteroEnabled = true
    citation-agent.zoteroEndpoint = http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


MOCK_ITEMS = [
    {
        "key": "ZOTERO-GAUSSIAN-SLAM",
        "title": "Distributed Gaussian SLAM for Collaborative Robot Teams",
        "citationKey": "chen2025distributedGaussianSlam",
        "creators": [
            {"firstName": "Mira", "lastName": "Chen"},
            {"firstName": "Noah", "lastName": "Patel"},
        ],
        "date": "2025-03-04",
        "publicationTitle": "IEEE Robotics and Automation Letters",
        "DOI": "10.0000/mock-zotero-gaussian-slam",
        "url": "https://example.org/mock-zotero-gaussian-slam",
        "pdfPath": "/mock/zotero/pdfs/chen2025distributedGaussianSlam.pdf",
        "abstractNote": (
            "A mock Zotero item about distributed Gaussian SLAM, "
            "collaborative robot teams, and global optimization."
        ),
    },
    {
        "key": "ZOTERO-MULTI-AGENT-MAPPING",
        "title": "Multi-Agent Mapping with Offline Pose Graph Optimization",
        "citationKey": "rivera2024multiAgentMapping",
        "creators": [
            {"firstName": "Elena", "lastName": "Rivera"},
            {"firstName": "Bo", "lastName": "Zhang"},
        ],
        "date": "2024",
        "conferenceName": "International Conference on Robotics and Automation",
        "DOI": "10.0000/mock-zotero-mapping",
        "url": "https://example.org/mock-zotero-mapping",
        "pdfPath": "/mock/zotero/pdfs/rivera2024multiAgentMapping.pdf",
        "abstractNote": (
            "A mock Zotero item describing multi-agent mapping systems "
            "that rely on offline pose graph and global optimization."
        ),
    },
]


class MockZoteroHandler(BaseHTTPRequestHandler):
    server_version = "EasyCiteMockZotero/0.1"

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/search", "/items"}:
            self._send_json({"error": "not found", "path": parsed.path}, status=404)
            return

        query = parse_qs(parsed.query)
        claim = query.get("q", [""])[0]
        limit = _safe_int(query.get("limit", ["5"])[0], default=5)
        print(f"[mock-zotero] {parsed.path} q={claim!r} limit={limit}", flush=True)

        items = _rank_items(claim)[: max(limit, 0)]
        self._send_json({"items": items})

    def log_message(self, fmt, *args):
        print(f"[mock-zotero] {fmt % args}", flush=True)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _rank_items(claim: str) -> list[dict]:
    terms = {
        token.lower()
        for token in claim.replace("-", " ").replace(".", " ").split()
        if len(token) > 2
    }

    def score(item: dict) -> int:
        haystack = " ".join(
            str(item.get(field, ""))
            for field in ["title", "abstractNote", "publicationTitle", "conferenceName"]
        ).lower()
        return sum(1 for term in terms if term in haystack)

    return sorted(MOCK_ITEMS, key=score, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mock Zotero endpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MockZoteroHandler)
    url = f"http://{args.host}:{args.port}"
    print("[mock-zotero] running", url, flush=True)
    print("[mock-zotero] supports GET /search?q=...&limit=...", flush=True)
    print("[mock-zotero] press Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock-zotero] stopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
