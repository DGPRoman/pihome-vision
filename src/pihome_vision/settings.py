"""Settings that come from the environment: the secrets, and where the rest is.

Every variable is prefixed with ``PIHOME_VISION_`` so the service cannot pick up
unrelated ones by accident. The camera address and the hub key are typed as
:class:`~pydantic.SecretStr`, which keeps them out of ``repr()``, tracebacks and log
lines. Everything that is not a secret lives in ``vision.yaml`` instead — see
:mod:`pihome_vision.config`.
"""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from pydantic import SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pihome_vision.redact import mask_url

ENV_PREFIX: Final = "PIHOME_VISION_"

#: The hub refuses keys shorter than this, so a shorter one here can only be a typo.
MIN_HUB_KEY_LENGTH: Final = 32

#: Values that look like a key but are really left over from example configuration.
_REJECTED_KEY_MARKERS: Final = (
    "changeme",
    "change_me",
    "change-me",
    "example",
    "placeholder",
    "replaceme",
    "replace-me",
    "replace_me",
    "yourkeyhere",
)

#: Where ``vision.yaml`` is unless the environment says otherwise.
DEFAULT_CONFIG_PATH: Final = Path("config/vision.yaml")

CAMERA_SCHEMES: Final = frozenset({"rtsp", "rtsps", "http", "https"})
_LOCAL_CAMERA = re.compile(r"cam:\d+")

#: Characters a camera's address cannot hold: ffmpeg reads it as a line of a playlist.
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

#: Where plain HTTP to the hub is allowed: addresses that cannot be reached across
#: the internet, so the key cannot be read off the wire by anybody outside the house.
#: The same set the Android client allows, 100.64.0.0/10 included for VPNs that hand
#: out addresses from it.
_PRIVATE_NETWORKS: Final = tuple(
    ipaddress.ip_network(net)
    for net in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)


def _is_private_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in network for network in _PRIVATE_NETWORKS)


class CameraSettings(BaseSettings):
    """What a command that only looks through the camera reads from its environment.

    Not the hub: a desktop used to draw zones need not hold the key to every relay.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    #: The camera, with its credentials: ``rtsp://user:password@host/path``, an
    #: ``http(s)`` stream, or ``cam:N`` for a webcam on this machine.
    camera_url: SecretStr
    #: Everything that is not a secret.
    config_path: Path = DEFAULT_CONFIG_PATH

    @field_validator("camera_url")
    @classmethod
    def _camera_url(cls, value: SecretStr) -> SecretStr:
        # No message here may quote the value: it is the one with the password in it.
        url = value.get_secret_value().strip()
        if _CONTROL.search(url):
            msg = "must be one line, without control characters"
            raise ValueError(msg)
        if _LOCAL_CAMERA.fullmatch(url):
            return SecretStr(url)
        try:
            parts = urlsplit(url)
            host = parts.hostname
        except ValueError:
            msg = "is not a URL that can be read"
            raise ValueError(msg) from None
        if parts.scheme not in CAMERA_SCHEMES:
            msg = f"must start with one of {', '.join(sorted(CAMERA_SCHEMES))}, or be cam:N"
            raise ValueError(msg)
        if not host:
            msg = "names no host"
            raise ValueError(msg)
        return SecretStr(url)


class Settings(CameraSettings):
    """What the service reads from its environment: the camera's, and the hub's."""

    #: The hub's origin, nothing after it: ``https://hub.example``.
    hub_url: str
    #: The hub's relay key. It opens every relay, which is why it is a secret here
    #: and why SECURITY.md says how to keep it.
    hub_key: SecretStr

    @field_validator("hub_url")
    @classmethod
    def _hub_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        # Quoted through mask_url: a key pasted in as user:password@ must not be
        # echoed by the message that refuses it.
        shown = repr(mask_url(url))
        try:
            parts = urlsplit(url)
            host = parts.hostname
            parts.port  # noqa: B018 - raises on a port that is not a number
        except ValueError:
            msg = f"{shown} is not a URL that can be read"
            raise ValueError(msg) from None
        if parts.username is not None or parts.password is not None:
            msg = "must not carry credentials; the key goes in PIHOME_VISION_HUB_KEY"
            raise ValueError(msg)
        if parts.scheme not in {"http", "https"} or not host:
            msg = f"{shown} must be an http or https address of the hub"
            raise ValueError(msg)
        if parts.path or parts.query or parts.fragment:
            msg = f"{shown} must be the hub's origin, with nothing after the host and port"
            raise ValueError(msg)
        if parts.scheme == "http" and not _is_private_host(host):
            msg = (
                f"{shown} would send the relay key in the clear to an address outside "
                "this network; use https, or a private address"
            )
            raise ValueError(msg)
        return url

    @field_validator("hub_key")
    @classmethod
    def _hub_key(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if len(secret) < MIN_HUB_KEY_LENGTH:
            msg = f"must be at least {MIN_HUB_KEY_LENGTH} characters, got {len(secret)}"
            raise ValueError(msg)
        normalised = secret.casefold().replace(" ", "")
        for marker in _REJECTED_KEY_MARKERS:
            if marker in normalised:
                msg = f"looks like example configuration (contains {marker!r})"
                raise ValueError(msg)
        return value


def render_settings_error(exc: ValidationError) -> str:
    """Name each variable that is wrong and say why, and nothing more.

    Pydantic's own rendering quotes the value it was given, which for the camera is
    the address with the password in it. Only the field and the message are used.
    """
    lines = ["pihome-vision: the environment is incomplete or invalid", ""]
    for error in exc.errors(include_input=False, include_url=False):
        field = str(error["loc"][0]) if error["loc"] else ""
        message = error["msg"].removeprefix("Value error, ")
        name = f"{ENV_PREFIX}{field}".upper()
        lines.append(f"  {name}: {message}")
    lines += ["", "Every variable is described in .env.example."]
    return "\n".join(lines)


def config_path_from_environment() -> Path:
    """Where ``vision.yaml`` is, for a command that needs it but none of the secrets.

    Reads the same variable :class:`Settings` would. A ``.env`` file is not consulted:
    it exists for development runs of the service, and holds the secrets this avoids.
    """
    override = os.environ.get(f"{ENV_PREFIX}CONFIG_PATH", "")
    return Path(override) if override else DEFAULT_CONFIG_PATH
