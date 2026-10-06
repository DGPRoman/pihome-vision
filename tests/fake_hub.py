"""A stand-in for the hub's relay routes, on a local port, for tests.

It answers ``GET`` and ``PUT /v1/relays/{id}`` as the hub does, keeps every request
it was sent, and can be told to answer the next few with a status of the test's
choosing instead.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, override

PREFIX = "/v1/relays/"


@dataclass
class Request:
    method: str
    path: str
    key: str | None
    body: dict[str, bool] | None


@dataclass
class FakeHub:
    key: str
    relays: dict[str, bool] = field(default_factory=dict)
    requests: list[Request] = field(default_factory=list)
    #: Statuses to answer with, one per request, before answering properly again.
    statuses: deque[int] = field(default_factory=deque)
    #: Headers to add to every answer: a Location, say.
    headers: dict[str, str] = field(default_factory=dict)
    #: A body to send instead of the relay's state.
    body: bytes | None = None
    #: Set to hold every answer back until it is set.
    released: threading.Event = field(default_factory=threading.Event)
    origin: str = ""

    def __post_init__(self) -> None:
        self.released.set()

    @property
    def puts(self) -> list[tuple[str, bool]]:
        return [
            (r.path.removeprefix(PREFIX), r.body["on"])
            for r in self.requests
            if r.method == "PUT" and r.body is not None
        ]


def serve(hub: FakeHub) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._answer(None)

        def do_PUT(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self._answer(json.loads(self.rfile.read(length)))

        def _answer(self, body: dict[str, bool] | None) -> None:
            hub.requests.append(Request(self.command, self.path, self.headers["X-API-Key"], body))
            hub.released.wait()
            relay = self.path.removeprefix(PREFIX)
            if hub.statuses:
                self._send(hub.statuses.popleft(), {"detail": "as the test asked"})
            elif hub.key != self.headers["X-API-Key"]:
                self._send(HTTPStatus.UNAUTHORIZED, {"detail": "Invalid or missing API key"})
            elif relay not in hub.relays:
                self._send(HTTPStatus.NOT_FOUND, {"detail": "no such relay"})
            else:
                if body is not None:
                    hub.relays[relay] = body["on"]
                self._send(HTTPStatus.OK, {"id": relay, "on": hub.relays[relay]})

        def _send(self, status: int, content: dict[str, Any]) -> None:
            payload = hub.body if hub.body is not None else json.dumps(content).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for name, value in hub.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        @override
        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    hub.origin = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    return server
