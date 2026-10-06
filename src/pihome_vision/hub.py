"""Reading and switching the hub's relays: ``GET`` and ``PUT /v1/relays/{id}``.

Nothing else of the hub's API is used. Every request carries the relay key in
``X-API-Key`` and is sent to the configured origin only: redirects are refused
rather than followed, because following one would hand the key to wherever it
pointed, and proxies from the environment are ignored for the same reason.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from enum import Enum
from http import HTTPStatus
from http.client import HTTPException, HTTPMessage
from typing import IO, Final, Protocol, override
from urllib.parse import quote

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


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    @override
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> None:
        return None


_UNAVAILABLE_STATUSES: Final = frozenset({500, 502, 503, 504})


class Hub:
    """The hub at ``origin``, asked with ``key``."""

    def __init__(self, origin: str, key: str, *, timeout: float = TIMEOUT) -> None:
        self._origin = origin.rstrip("/")
        self._key = key
        self._timeout = timeout
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
            _NoRedirects(),
        )

    def read(self, relay: str) -> bool:
        return self._request("GET", relay, None)

    def switch(self, relay: str, *, on: bool) -> bool:
        return self._request("PUT", relay, {"on": on})

    def _request(self, method: str, relay: str, body: dict[str, bool] | None) -> bool:
        url = f"{self._origin}/v1/relays/{quote(relay, safe='')}"
        headers = {"X-API-Key": self._key, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        # The origin is checked when the settings are loaded: http or https only.
        request = urllib.request.Request(url, data, headers, method=method)  # noqa: S310
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            exc.close()
            raise _refusal(exc.code) from None
        except (OSError, HTTPException) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            raise HubError(Refusal.UNAVAILABLE, str(reason)) from None
        return _on(payload)


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
