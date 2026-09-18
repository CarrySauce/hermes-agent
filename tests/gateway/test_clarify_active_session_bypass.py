"""Regression tests for clarify replies while a gateway session is busy."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _ClarifyBypassAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(text="custom answer"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


@pytest.mark.asyncio
async def test_active_session_routes_typed_choice_clarify_reply_to_runner_not_busy_queue():
    """Typed text must resolve a pending choice clarify even while the agent is busy.

    Telegram button clarifies keep the adapter session active while the agent
    thread blocks on ``wait_for_response``.  If the adapter only bypasses for
    entries already marked ``awaiting_text``, typed replies to the visible
    multi-choice prompt are handled as busy follow-ups and the clarify wait is
    never resolved.
    """
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_active_session_bypass_uses_profile_namespaced_key_under_multiplex():
    """Regression for issue #82975: under a named-profile multiplex, the
    adapter's clarify bypass lookup must use the SAME profile-namespaced
    session key that the runner registers pending clarifies under
    (SessionStore._generate_session_key() includes
    profile=self._resolve_profile_for_key(source)), not the legacy
    unnamespaced key. Otherwise the lookup misses, and a user's answer to
    a pending clarify is routed to the busy-session queue instead of
    resolving it -- the turn then hangs until the clarify's 3600s timeout."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")

    # A session_store configured for profile multiplexing, matching what
    # the runner's SessionStore._generate_session_key() actually produces.
    session_store = MagicMock()
    session_store._resolve_profile_for_key.return_value = "ops"
    adapter._session_store = session_store

    profile_namespaced_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
        profile="ops",
    )
    # Sanity: the profile-namespaced key really is different from the
    # legacy unnamespaced one -- otherwise this test wouldn't distinguish
    # the fixed behavior from the bug.
    legacy_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    assert profile_namespaced_key != legacy_key

    adapter._active_sessions[profile_namespaced_key] = asyncio.Event()
    # The runner registers the pending clarify under its own
    # profile-namespaced key, exactly as it would in a real multiplexed
    # deployment.
    cm.register("clarify-1", profile_namespaced_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}




# ---------------------------------------------------------------------------
# would_accept_text_response — the routing pre-check
#
# Callers that must decide how to route a message BEFORE the gateway's text
# intercept sees it (the Telegram guest path, where a message routed as an
# ordinary turn has no reply surface) need the same verdict the intercept will
# reach. These pin the two to each other.
# ---------------------------------------------------------------------------

@pytest.fixture
def clarify_module():
    from tools import clarify_gateway

    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()
    yield clarify_gateway
    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()


def test_would_accept_matches_the_awaiting_text_entry(clarify_module):
    """After "Other" (or an open-ended prompt) any real text is the answer."""
    clarify_module.register("c1", "sess", "Which day?", None)

    assert clarify_module.would_accept_text_response("sess", "next Friday") is True
    # …and the check does not consume the entry.
    assert clarify_module.has_pending("sess") is True


def test_would_accept_rejects_what_the_intercept_declines(clarify_module):
    """Empty text and slash commands fall through the intercept, so they are not answers."""
    clarify_module.register("c2", "sess", "Which day?", None)

    assert clarify_module.would_accept_text_response("sess", "") is False
    assert clarify_module.would_accept_text_response("sess", "   ") is False
    assert clarify_module.would_accept_text_response("sess", "/status") is False


def test_would_accept_follows_choice_coercion(clarify_module):
    """A choice prompt takes numbers and exact labels; unmatched prose it rejects."""
    clarify_module.register("c3", "sess", "Which day?", ["Mon", "Tue"])

    assert clarify_module.would_accept_text_response("sess", "2") is True
    assert clarify_module.would_accept_text_response("sess", "Tue") is True
    assert clarify_module.would_accept_text_response("sess", "what's the weather?") is False
    # Selection-shaped but out of range: still not an answer (the intercept retains the prompt).
    assert clarify_module.would_accept_text_response("sess", "9") is False


def test_would_accept_agrees_with_attempt_on_the_same_input(clarify_module):
    """The pre-check and the resolution must not drift apart."""
    for text, expected in (("2", True), ("Mon", True), ("nonsense prose here", False)):
        clarify_module._entries.clear()
        clarify_module._session_index.clear()
        clarify_module.register("c4", "sess", "Which day?", ["Mon", "Tue"])
        predicted = clarify_module.would_accept_text_response("sess", text)
        resolved = clarify_module.attempt_text_response_for_session("sess", text) == clarify_module.TEXT_RESOLVED
        assert predicted is expected and resolved is expected


def test_would_accept_is_false_without_a_pending_clarify(clarify_module):
    assert clarify_module.would_accept_text_response("sess", "anything") is False
