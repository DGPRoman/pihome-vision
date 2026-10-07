"""Reading and switching the hub's relays: ``GET`` and ``PUT /v1/relays/{id}``.

Nothing else of the hub's API is used. Every request carries the relay key in
``X-API-Key`` and is sent to the configured origin only: redirects are refused
rather than followed, because following one would hand the key to wherever it
pointed, and proxies from the environment are ignored for the same reason.

One connection is kept open from request to request. Switching a light on is a
read and then a switch, and on a connection already open the two take about a
third of the time they take on two new ones, TLS handshakes and all.
"""

from __future__ import annotations

import json
import ssl
import threading
from enum import Enum
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from types import TracebackType
from typing import Final, Protocol, Self
from urllib.parse import quote, urlsplit

#: Seconds to wait for the hub to answer one request.
TIMEOUT: Final = 5.0


class Refusal(Enum):
    """Why a request to the hub did not work. Each value says it to a person."""

    #: 401. The hub counts these per address and, after a few, refuses every client
    #: behind the same router for a while, so the key is not tried again.
    UNAUTHORIZED = "the hub turned down the relay key"
    #: 404.
    NO_SUCH_RELAY = "the hub has no relay by that name"
    #: 429: too many failed keys from this address, not necessarily this program's.
    THROTTLED = "the hub is refusing this address for a while after failed keys"
    #: Not there, not answering, or failing on its side (5xx). Worth trying again.
    UNAVAILABLE = "the hub could not be reached"
    #: Any other answer. Trying again would get the same one.
    REJECTED = "the hub refused the request"


class HubError(Exception):
    def __init__(self, refusal: Refusal, detail: str) -> None:
        super().__init__(f"{refusal.value}: {detail}")
        self.refusal = refusal


class Relays(Protocol):
    """What switching a light needs of the hub."""

    def read(self, relay: str) -> bool:
        """Whether ``relay`` is on."""
        ...

    def switch(self, relay: str, *, on: bool) -> bool:
        """Switch ``relay`` and say whether it is on now."""
        ...


_UNAVAILABLE_STATUSES: Final = frozenset({500, 502, 503, 504})


class Hub:
    """The hub at ``origin``, asked with ``key``, one request at a time."""

    def __init__(self, origin: str, key: str, *, timeout: float = TIMEOUT) -> None:
        # The origin is checked when the settings are loaded: http or https, a host
        # and maybe a port, nothing else.
        parts = urlsplit(origin)
        self._https = parts.scheme == "https"
        self._host = parts.hostname or ""
        # Given apart from the host, which http.client would otherwise read the
        # last group of an IPv6 address from as a port.
        self._port = parts.port or (443 if self._https else 80)
        self._key = key
        self._timeout = timeout
        self._context = ssl.create_default_context() if self._https else None
        self._connection: HTTPConnection | None = None
        self._lock = threading.Lock()

    def read(self, relay: str) -> bool:
        return self._request("GET", relay, None)

    def switch(self, relay: str, *, on: bool) -> bool:
        return self._request("PUT", relay, {"on": on})

    def close(self) -> None:
        """Close the connection kept open; the next request opens another."""
        with self._lock:
            self._drop()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _request(self, method: str, relay: str, body: dict[str, bool] | None) -> bool:
        path = f"/v1/relays/{quote(relay, safe='')}"
        headers = {"X-API-Key": self._key, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        with self._lock:
            status, payload = self._exchange(method, path, data, headers)
        if status != HTTPStatus.OK:
            raise _refusal(status)
        return _on(payload)

    def _exchange(
        self, method: str, path: str, data: bytes | None, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        if self._connection is not None:
            try:
                return self._send(self._connection, method, path, data, headers)
            except TimeoutError as exc:
                # A hub too slow to answer is not asked again at once: the lights
                # retry when it is worth it.
                raise _unavailable(exc) from None
            except (OSError, HTTPException):
                # Closed by the hub after lying idle, which it may have done just as
                # this was sent. Both requests can be repeated safely, so this one
                # goes again on a new connection.
                pass
        self._connection = (
            HTTPSConnection(self._host, self._port, timeout=self._timeout, context=self._context)
            if self._https
            else HTTPConnection(self._host, self._port, timeout=self._timeout)
        )
        try:
            return self._send(self._connection, method, path, data, headers)
        except (OSError, HTTPException) as exc:
            raise _unavailable(exc) from None

    def _send(
        self,
        connection: HTTPConnection,
        method: str,
        path: str,
        data: bytes | None,
        headers: dict[str, str],
    ) -> tuple[int, bytes]:
        try:
            connection.request(method, path, data, headers)
            response = connection.getresponse()
            payload = response.read()
        except BaseException:
            self._drop()
            raise
        if response.will_close:
            self._drop()
        return response.status, payload

    def _drop(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def _unavailable(exc: OSError | HTTPException) -> HubError:
    return HubError(Refusal.UNAVAILABLE, str(exc) or type(exc).__name__)


def _refusal(status: int) -> HubError:
    detail = f"HTTP {status}"
    match status:
        case HTTPStatus.UNAUTHORIZED:
            return HubError(Refusal.UNAUTHORIZED, detail)
        case HTTPStatus.NOT_FOUND:
            return HubError(Refusal.NO_SUCH_RELAY, detail)
        case HTTPStatus.TOO_MANY_REQUESTS:
            return HubError(Refusal.THROTTLED, detail)
        case _ if status in _UNAVAILABLE_STATUSES:
            return HubError(Refusal.UNAVAILABLE, detail)
        case _ if HTTPStatus.MULTIPLE_CHOICES <= status < HTTPStatus.BAD_REQUEST:
            detail += ", a redirect, which is not followed: is the hub's address right?"
    return HubError(Refusal.REJECTED, detail)


def _on(payload: bytes) -> bool:
    try:
        state = json.loads(payload)
    except ValueError:
        state = None
    on = state.get("on") if isinstance(state, dict) else None
    if not isinstance(on, bool):
        raise HubError(Refusal.REJECTED, "the answer does not say whether the relay is on")
    return on
