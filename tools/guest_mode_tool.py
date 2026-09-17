"""One-shot delivery tokens for Telegram guest-mode (Bot API 10.0) attachments.

A guest bot can only speak into a chat it has not joined by *answering* a
``guest_query_id``, and each query answers exactly once. That makes the normal
"text reply, then upload the file" sequence impossible: the text reply already
spent the turn's only answer.

The way around it is to hand the caller a second query. The reply carries a
``switch_inline_query_current_chat`` button whose payload is ``deliver_<token>``;
tapping it pre-fills that text in the caller's input box, and sending it arrives
as a *new* guest message with a *fresh* query id — which the adapter answers with
the cached media instead of text.

The token is what connects the two turns. It is opaque, single-use and
short-lived, and it resolves to a Telegram ``file_id`` that was staged in the
operator's own home channel, never to a host path: a leaked or guessed token
cannot name a file, only redeem one the bot already decided to offer. Caller
authorization is enforced again at redemption (see the adapter's
``deliver_<token>`` branch) — the token is a delivery handle, not a capability
grant.

In-memory by design: tokens are scoped to one conversation turn, so surviving a
restart has no value and persisting them would only widen the window.
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# token -> {"file_id", "media_kind", "caption", "file_name", "expires_at"}
_TOKEN_STORE: Dict[str, Dict[str, Any]] = {}

# Long enough for a human to read the reply and tap the button, short enough
# that a token found in a screenshot later is already dead.
TOKEN_TTL_SECONDS = 15 * 60
# Hard cap on live tokens; a busy bot must not grow this dict without bound.
_MAX_TOKENS = 256

MEDIA_KINDS = ("photo", "video", "audio", "voice", "document", "animation")


def _purge_expired(now: Optional[float] = None) -> None:
    """Drop expired entries, then the oldest ones if the store is still over cap."""
    now = time.monotonic() if now is None else now
    for token in [t for t, rec in _TOKEN_STORE.items() if rec.get("expires_at", 0) <= now]:
        _TOKEN_STORE.pop(token, None)
    while len(_TOKEN_STORE) > _MAX_TOKENS:
        oldest = min(_TOKEN_STORE, key=lambda t: _TOKEN_STORE[t].get("expires_at", 0))
        _TOKEN_STORE.pop(oldest, None)


def mint_token(
    file_id: str, media_kind: str, *, caption: Optional[str] = None,
    file_name: Optional[str] = None, ttl: float = TOKEN_TTL_SECONDS,
) -> str:
    """Mint a single-use delivery token for an already-staged Telegram ``file_id``.

    ``media_kind`` is one of :data:`MEDIA_KINDS` and decides which
    ``InlineQueryResultCached*`` the redemption builds.
    """
    _purge_expired()
    # URL-safe and button-payload-safe: the token rides inside the 64-char
    # switch_inline_query_current_chat payload, after the "deliver_" prefix.
    token = secrets.token_urlsafe(16)
    _TOKEN_STORE[token] = {
        "file_id": file_id,
        "media_kind": media_kind,
        "caption": caption,
        "file_name": file_name,
        "expires_at": time.monotonic() + ttl,
    }
    return token


def resolve_token(token: str, *, consume: bool = True) -> Optional[Dict[str, Any]]:
    """Return the record for *token*, or ``None`` when unknown or expired.

    Single-use by default: a successful resolve removes the token, so a replay
    of the same ``deliver_<token>`` message (or a second person sending the
    payload they saw on screen) lands on the expired branch.
    """
    _purge_expired()
    record = _TOKEN_STORE.get(token)
    if record is None:
        return None
    if record.get("expires_at", 0) <= time.monotonic():
        _TOKEN_STORE.pop(token, None)
        return None
    if consume:
        _TOKEN_STORE.pop(token, None)
    return record
