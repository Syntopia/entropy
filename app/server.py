"""Minimal HTTP server for the transition state visualiser."""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from . import compute_visualisation, default_configuration


def _load_index_template() -> str:
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text(encoding="utf-8")


INDEX_TEMPLATE = _load_index_template()
DEFAULT_CONFIG_JSON = json.dumps(default_configuration())


class TransitionStateHandler(BaseHTTPRequestHandler):
    server_version = "TransitionStateServer/1.0"

    def _set_common_headers(self, status: HTTPStatus = HTTPStatus.OK, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802 - required name by BaseHTTPRequestHandler
        self._set_common_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            content = INDEX_TEMPLATE.replace("__DEFAULT_CONFIG__", DEFAULT_CONFIG_JSON)
            encoded = content.encode("utf-8")
            self._set_common_headers(content_type="text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/pes":
            self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload: Dict[str, Any] = json.loads(body or "{}")
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON payload")
            return
        response = compute_visualisation(payload)
        encoded = json.dumps(response).encode("utf-8")
        self._set_common_headers()
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run(host: str = "127.0.0.1", port: int = 5000) -> None:
    server = ThreadingHTTPServer((host, port), TransitionStateHandler)
    print(f"Serving Transition State Visualiser on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server…")
    finally:
        server.server_close()


__all__ = ["run"]
