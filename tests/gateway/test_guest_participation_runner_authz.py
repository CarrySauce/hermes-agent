"""Reproduction: does the gateway runner admit an Allow-once replay?

The adapter's own guest gate and the runner's inbound gate are the same authorization chain, so a
caller the gate refused is refused again when the replayed event reaches ``run_inbound``. This
walks the REAL source the adapter builds for the replay through the REAL
``_principal_authorized``, rather than stopping at ``_enqueue_text_event`` the way the
participation tests do.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.session import Platform, SessionSource
from tests.gateway.test_telegram_guest_reply import (  # noqa: E402
    _make_guest_update, _participation_adapter, _request_id,
)

STRANGER = "8446220098"
OPERATOR = "525723850"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "TELEGRAM_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS", "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _runner(*, paired: bool = False):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: paired)
    return runner


async def _allow_once_event(adapter, *, choice="o"):
    """The event an Allow tap's replay hands the gateway (``choice`` "o" once, "a" always)."""
    adapter._bot.answer_guest_query = AsyncMock(
        return_value=SimpleNamespace(inline_message_id="imi_wait"))
    update, msg = _make_guest_update(update_id=901, gqid="gq_repro", caller_id=STRANGER)
    msg.reply_to_message = None
    with patch.object(adapter, "_is_callback_user_authorized", return_value=False), \
         patch.object(adapter, "_should_process_message"):
        await adapter._handle_guest_message_update(update, MagicMock())
    query = MagicMock()
    query.message = None
    query.inline_message_id = "imi_wait"
    query.data = f"gp:{choice}:{_request_id(adapter)}"
    query.from_user = MagicMock(id=int(OPERATOR), first_name="Owner", username="owner")
    query.answer, query.edit_message_text = AsyncMock(), AsyncMock()
    captured = []
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True), \
         patch.object(adapter, "_should_process_message", return_value=True), \
         patch.object(adapter, "_enqueue_text_event", side_effect=captured.append):
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())
    assert len(captured) == 1
    return captured[0]


@pytest.mark.asyncio
async def test_allow_once_event_survives_runner_authorization(monkeypatch):
    """The bug this file exists for: the adapter handed the event over and the runner binned it,
    so the operator was told "it is running" and nothing ever answered."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    adapter = _participation_adapter()

    event = await _allow_once_event(adapter)

    assert event.source.user_id == STRANGER          # still the refused stranger
    assert _runner()._is_user_authorized(event.source) is True


@pytest.mark.asyncio
async def test_only_the_approved_event_carries_it(monkeypatch):
    """The stamp is on one event, not on the sender: their next message is refused again."""
    from gateway.platforms.event import MessageType

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    adapter = _participation_adapter()
    event = await _allow_once_event(adapter)
    runner = _runner()

    # The very next thing that user sends, built the ordinary way.
    _update, msg = _make_guest_update(update_id=902, gqid="gq_next", caller_id=STRANGER)
    follow_up = adapter._build_message_event(msg, MessageType.TEXT, update_id=902)

    assert event.source.participation_authorized is True
    assert follow_up.source.participation_authorized is False
    # This is the call run_inbound.py:218 and run_busy.py:709 both make, so a queued follow-up in
    # the still-running turn is refused too.
    assert runner._is_user_authorized(follow_up.source) is False


@pytest.mark.asyncio
async def test_the_stranger_cannot_send_a_second_request_at_all(monkeypatch):
    """Before authorization even matters: the guest gate refuses their next mention, so nothing
    reaches the runner to be authorized."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    adapter = _participation_adapter()
    await _allow_once_event(adapter)
    update, msg = _make_guest_update(update_id=903, gqid="gq_second", caller_id=STRANGER)
    msg.reply_to_message = None
    enqueued = []

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False), \
         patch.object(adapter, "_enqueue_text_event", side_effect=enqueued.append):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert enqueued == []


@pytest.mark.asyncio
async def test_a_clarify_prompt_inside_that_turn_stays_operator_only(monkeypatch):
    """Allow once authorized a request, not a person: a question it raises is not theirs to answer."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    adapter = _participation_adapter()
    await _allow_once_event(adapter)
    adapter._remember_guest_inline_message("42", "imi_wait", prompt_text="Which one?")
    tap = MagicMock()
    tap.message, tap.inline_message_id, tap.data = None, "imi_wait", "cl:q0:0"
    tap.from_user = MagicMock(id=int(STRANGER), first_name="Stranger", username="someone")
    tap.answer, tap.edit_message_text = AsyncMock(), AsyncMock()

    allowed = await adapter._callback_authorized(tap, adapter._callback_ctx(tap), "nope")

    assert allowed is False


@pytest.mark.asyncio
async def test_allow_always_needs_no_stamp_because_it_wrote_a_grant(monkeypatch, tmp_path):
    """Two different mechanisms, and the durable one must not quietly rely on the one-shot flag."""
    import hermes_constants
    from gateway.pairing import PairingStore

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    adapter = _participation_adapter(allow_always=True)

    event = await _allow_once_event(adapter, choice="a")

    assert event.source.participation_authorized is False
    # Authorized by the pairing grant that tap wrote, which the runner honours as a union.
    assert PairingStore().is_approved("telegram", STRANGER) is True
    assert _runner(paired=True)._is_user_authorized(event.source) is True


def test_the_stamp_is_ignored_without_adapter_delegation(monkeypatch):
    """The plugin-injection path (run_inbound.py:1852, allow_adapter_delegation=False) rebuilds a
    source from a persisted origin; a one-shot operator tap must not authorize it."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", chat_type="group", user_id=STRANGER,
        participation_authorized=True)

    assert _runner()._is_user_authorized(source) is True
    assert _runner()._is_user_authorized(source, allow_adapter_delegation=False) is False


def test_a_truthy_non_boolean_never_passes(monkeypatch):
    """Same discipline as role_authorized: a MagicMock stand-in must not auto-truthy into authz."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", chat_type="group", user_id=STRANGER)
    source.participation_authorized = MagicMock()

    assert _runner()._is_user_authorized(source) is False


def test_the_stamp_cannot_arrive_from_outside():
    """Absent from the serialized field set, so no persisted origin, relay payload or API body can
    set it and no stored session can resurrect it."""
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", chat_type="group", user_id=STRANGER,
        participation_authorized=True)

    assert "participation_authorized" not in source.to_dict()
    hostile = dict(source.to_dict(), participation_authorized=True)
    assert SessionSource.from_dict(hostile).participation_authorized is False


# ---------------------------------------------------------------------------
# The other half: an approved request the gateway never picks up
#
# _start_guest_turn returning True means the event was handed over, not admitted. When the
# runner drops it there is no callback, so without a watchdog the caller keeps staring at the
# card they tapped, the operator was told it is running, and the turn stays registered — three
# of those and the chat sits at its concurrency ceiling with nothing running.
# ---------------------------------------------------------------------------

async def _drain(adapter):
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_a_request_the_gateway_never_picks_up_fails_on_its_own_card(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    adapter = _participation_adapter()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=4242))
    monkeypatch.setattr(type(adapter), "_GUEST_REPLAY_WATCHDOG_SECONDS", 0.0)

    await _allow_once_event(adapter)
    assert adapter._guest_live_turn_keys("42") == ["42"]  # registered, waiting on a gateway
    await _drain(adapter)

    # Released, so the chat is not left one slot short of its ceiling for good.
    assert adapter._guest_live_turn_keys("42") == []
    edits = [c.kwargs for c in reversed(adapter._bot.edit_message_text.await_args_list)]
    in_chat = next(c for c in edits if c.get("inline_message_id") == "imi_wait")
    assert "never ran" in in_chat["text"] and "again" in in_chat["text"]
    # The operator was told "it is running now"; that claim is what their copy ends up corrected to.
    operator = next(c for c in edits if c.get("message_id") is not None)
    assert "never ran" in operator["text"] and "Nothing was granted" in operator["text"]


@pytest.mark.asyncio
async def test_a_turn_the_gateway_is_driving_is_left_alone(monkeypatch):
    """Any outbound call for that turn proves the gateway picked it up; a slow turn must not be
    torn down under it."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    adapter = _participation_adapter()
    monkeypatch.setattr(type(adapter), "_GUEST_REPLAY_WATCHDOG_SECONDS", 0.0)

    await _allow_once_event(adapter)
    await adapter.send_typing("42")          # what the gateway does the moment a turn starts
    adapter._bot.edit_message_text.reset_mock()
    await _drain(adapter)

    assert adapter._guest_live_turn_keys("42") == ["42"]
    adapter._bot.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_finished_turn_is_not_reported_as_never_run(monkeypatch):
    """A turn that answered fast is already released; the watchdog must not speak for it."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OPERATOR)
    adapter = _participation_adapter()
    monkeypatch.setattr(type(adapter), "_GUEST_REPLAY_WATCHDOG_SECONDS", 0.0)

    await _allow_once_event(adapter)
    adapter._release_guest_turn("42")
    adapter._bot.edit_message_text.reset_mock()
    await _drain(adapter)

    adapter._bot.edit_message_text.assert_not_awaited()
