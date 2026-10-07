"""A stand-in for the hub's relay routes, on a local port, for tests.

It answers ``GET`` and ``PUT /v1/relays/{id}`` as the hub does, on connections kept
open between requests as the hub's are, keeps every request it was sent, and can be
told to answer the next few with a status of the test's choosing instead. It also
hands out real clients of itself, which the ``hub`` fixture closes when the test ends.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, override

from pihome_vision.hub import TIMEOUT, Hub

PREFIX = "/v1/relays/"


@dataclass
class Request:
    method: str
    path: str
    key: str | None
    body: dict[str, bool] | None
    #: The client's port: the same for requests on the same connection.
    peer: int


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
    #: Close the connection after the next answer without saying so, as the hub
    #: does with one left idle.
    hang_up: bool = False
    origin: str = ""
    clients: list[Hub] = field(default_factory=list)

    def client(self, key: str | None = None, *, timeout: float = TIMEOUT) -> Hub:
        """A client of this hub, with its key unless told otherwise."""
        self.clients.append(Hub(self.origin, self.key if key is None else key, timeout=timeout))
        return self.clients[-1]

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
        protocol_version = "HTTP/1.1"
        # The headers and the body go out in two writes, and on a connection that
        # stays open the second would otherwise wait for the client's delayed ACK.
        disable_nagle_algorithm = True

        def do_GET(self) -> None:
            self._answer(None)

        def do_PUT(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self._answer(json.loads(self.rfile.read(length)))

        def _answer(self, body: dict[str, bool] | None) -> None:
            hub.requests.append(
                Request(
                    self.command, self.path, self.headers["X-API-Key"], body, self.client_address[1]
                )
            )
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
            if hub.hang_up:
                hub.hang_up = False
                self.close_connection = True

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

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        @override
        def handle_error(self, request: Any, client_address: Any) -> None:
            # A client that stopped waiting for its answer, as tests about timeouts
            # make them, is not this hub's error.
            if not isinstance(sys.exc_info()[1], ConnectionError):
                super().handle_error(request, client_address)

    server = Server(("127.0.0.1", 0), Handler)
    hub.origin = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    return server
