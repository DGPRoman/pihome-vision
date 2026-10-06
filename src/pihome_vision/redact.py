"""Keeping credentials out of anything a person or a log might read.

A camera's address carries its user name and password, and it has to be printed
somewhere: in a log line saying which camera was lost, in an error saying which one
could not be opened. Every such place goes through :func:`mask_url`, so there is one
function to get right rather than one per message.
"""

from __future__ import annotations

from typing import Final
from urllib.parse import urlsplit, urlunsplit

#: What replaces the credentials. Says that some were there, which matters when the
#: reason a camera refuses is that they are wrong.
MASK: Final = "***"


def mask_url(url: str) -> str:
    """``url`` with any user name and password replaced by :data:`MASK`.

    Anything that cannot be parsed as a URL is masked whole: a string that defeats the
    parser is exactly the one whose credentials the parser cannot find.
    """
    if "@" not in url:
        return url
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return MASK
    # An @ outside the authority means the parser read the string some other way
    # than as scheme://user:password@host — "user:secret@host" has the scheme "user".
    if "@" not in parts.netloc or not host:
        return MASK
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority += f":{port}"
    return urlunsplit(
        (parts.scheme, f"{MASK}@{authority}", parts.path, parts.query, parts.fragment)
    )
