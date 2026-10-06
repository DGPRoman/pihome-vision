"""Keeping credentials out of anything a person or a log might read.

A camera's address carries its user name and password, and it has to be printed
somewhere: in a log line saying which camera was lost, in an error saying which one
could not be opened. Every such place goes through :func:`mask_url`, so there is one
function to get right rather than one per message. What ffmpeg says about the camera
goes through :func:`scrub`, because ffmpeg repeats the address it was given.
"""

from __future__ import annotations

import re
from typing import Final
from urllib.parse import unquote, urlsplit, urlunsplit

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


#: The user information of any URL in a piece of text: ``scheme://`` up to the ``@``.
_USERINFO: Final = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/@]*@")


def scrub(text: str, url: str) -> str:
    """``text``, which another program wrote after being handed ``url``, without its
    credentials.

    Any URL's user information is masked, and so is ``url``'s password wherever else
    it turns up, as typed or percent-decoded.
    """
    text = _USERINFO.sub(rf"\1{MASK}@", text)
    try:
        password = urlsplit(url).password
    except ValueError:
        password = None
    if password:
        for secret in {password, unquote(password)}:
            text = text.replace(secret, MASK)
    return text
