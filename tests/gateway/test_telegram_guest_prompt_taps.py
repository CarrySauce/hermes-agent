"""Guest button taps by anyone other than the person whose query drew the message.

``inline_message_id`` is not a message id — it is a reference into the VIEWER's own mailbox, so
every member of a basic group gets a different id for the same message. The adapter only ever knew
the id ``answerGuestQuery`` returned, which is the query author's view, so a tap by anybody else
was "unattributable" and answered "⚠️ This prompt expired". Attribution therefore has to come from
the PROMPT (the callback data identifies it) rather than from the surface the tap happened on.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import plugins.platforms.telegram.adapter as _tg_adapter_mod  # noqa: E402
from tests.gateway.test_telegram_guest_reply import (  # noqa: E402
    _FakeInlineKeyboardButton, _FakeInlineKeyboardMarkup, _FakeInlineQueryResultArticle,
    _FakeInputTextMessageContent, _clarify_kwargs, _make_adapter, _make_guest_update,
    _participation_adapter, _register_guest_chat, _request_id,
)


@pytest.fixture(autouse=True)
def _real_inline_classes(monkeypatch):
    """Same stand-ins the sibling module installs: the shared ``telegram`` MagicMock would hand back
    auto-created attributes, so a button's ``callback_data`` would not be the string the adapter set."""
    monkeypatch.setattr(_tg_adapter_mod, "InputTextMessageContent", _FakeInputTextMessageContent)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultArticle", _FakeInlineQueryResultArticle)
    monkeypatch.setattr(_tg_adapter_mod, "InlineKeyboardButton", _FakeInlineKeyboardButton)
    monkeypatch.setattr(_tg_adapter_mod, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)

AUTHOR = "999"            # whose @mention drew the message (and whose imi we hold)
OTHER = "8446220098"      # another member of the group, with their own imi for it
OPERATOR = "525723850"

# The id the tapper's client sends: the same message, from their mailbox.
OTHER_IMI = "BQAAAEIXb_cBAAAAeQYAANc8ACWcEiBc"


@pytest.fixture
def clarify_turn(monkeypatch):
    """A guest turn of AUTHOR with a clarify prompt drawn on its surface."""
    import tools.clarify_gateway as clarify_gateway

    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._guest_inline_message_ids["42"] = "imi_author"
    adapter._guest_chat_types["42"] = "supergroup"
    session_key = "agent:main:telegram:42:999"
    monkeypatch.setattr(adapter, "_gateway_session_key", lambda event: session_key)
    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()
    yield adapter, clarify_gateway, session_key
    clarify_gateway.clear_session(session_key)
    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()


async def _draw_clarify(adapter, clarify_gateway, session_key, *, clarify_id="c1"):
    clarify_gateway.register(clarify_id, session_key, "Which one?", ["Left", "Right"])
    adapter._clarify_state[clarify_id] = session_key
    await adapter.send_clarify(**_clarify_kwargs(
        clarify_id=clarify_id, session_key=session_key, question="Which one?",
        choices=["Left", "Right"]))


def _foreign_tap(*, data, user_id, imi=OTHER_IMI):
    """A tap carrying an inline_message_id the adapter has never seen."""
    query = MagicMock()
    query.message = None
    query.inline_message_id = imi
    query.data = data
    query.from_user = MagicMock(id=int(user_id), first_name="Someone", username="someone")
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


@pytest.mark.asyncio
async def test_repro_a_tap_from_another_viewer_is_not_refused_as_expired(clarify_turn):
    """The reported failure: 7 refusals on a live prompt, all taps by a second person."""
    adapter, clarify_gateway, session_key = clarify_turn
    await _draw_clarify(adapter, clarify_gateway, session_key)
    query = _foreign_tap(data="cl:c1:0", user_id=OPERATOR)

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    answered = query.answer.await_args.kwargs.get("text", "") if query.answer.await_args else ""
    assert "expired" not in answered.lower(), answered


@pytest.mark.asyncio
async def test_an_authorized_non_author_answers_the_prompt(clarify_turn):
    """Attribution comes from the prompt, so a second allowed person resolves it normally."""
    adapter, clarify_gateway, session_key = clarify_turn
    await _draw_clarify(adapter, clarify_gateway, session_key)
    query = _foreign_tap(data="cl:c1:0", user_id=OPERATOR)

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True) as gate:
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    # Judged against the chat and turn the PROMPT names, identically to a tap by its author.
    assert gate.call_args.kwargs["chat_id"] == "42"
    assert gate.call_args.kwargs["chat_type"] == "supergroup"
    assert clarify_gateway.get_pending_for_session(session_key) is None  # resolved


@pytest.mark.asyncio
async def test_the_attributed_context_matches_the_authors_own(clarify_turn):
    """"Identical ctx" is the contract: same gate, same turn key, same generation semantics."""
    adapter, clarify_gateway, session_key = clarify_turn
    await _draw_clarify(adapter, clarify_gateway, session_key)

    author = adapter._callback_ctx(_foreign_tap(data="cl:c1:0", user_id=AUTHOR, imi="imi_author"))
    other = adapter._callback_ctx(_foreign_tap(data="cl:c1:0", user_id=OTHER))

    assert other["chat_id"] == author["chat_id"] == "42"
    assert other["chat_type"] == author["chat_type"]
    assert other["guest_turn_key"] == author["guest_turn_key"] == "42"
    assert other["guest_generation"] == author["guest_generation"]


@pytest.mark.asyncio
async def test_an_unknown_prompt_is_still_refused_as_expired(clarify_turn):
    """A restart or a superseded question leaves nothing to attribute, and "expired" is right."""
    adapter, _clarify_gateway, _session_key = clarify_turn
    query = _foreign_tap(data="cl:ancient:0", user_id=OPERATOR)

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True) as gate:
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    gate.assert_not_called()
    assert "expired" in query.answer.await_args.kwargs["text"].lower()


@pytest.mark.asyncio
async def test_a_resolved_exec_prompt_stops_being_tappable(clarify_turn):
    """A one-shot decision takes its buttons with it, so a second press cannot re-decide it."""
    adapter, _clarify_gateway, session_key = clarify_turn
    await adapter.send_exec_approval(chat_id="42", command="rm -rf /", session_key=session_key)
    data = next(d for d in adapter._guest_prompt_taps if d.startswith("ea:once:"))

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True), \
         patch("tools.approval.resolve_gateway_approval", return_value=1):
        await adapter._handle_callback_query(
            MagicMock(callback_query=_foreign_tap(data=data, user_id=OPERATOR)), MagicMock())

    assert adapter._guest_prompt_tap_context(data) is None
    second = _foreign_tap(data=data, user_id=OPERATOR)
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_callback_query(MagicMock(callback_query=second), MagicMock())
    assert "expired" in second.answer.await_args.kwargs["text"].lower()


@pytest.mark.asyncio
async def test_a_superseded_question_is_no_longer_tappable(clarify_turn):
    """One guest surface shows one prompt: the question that was replaced is genuinely gone."""
    adapter, clarify_gateway, session_key = clarify_turn
    await _draw_clarify(adapter, clarify_gateway, session_key, clarify_id="q1")
    await _draw_clarify(adapter, clarify_gateway, session_key, clarify_id="q2")

    assert adapter._guest_prompt_tap_context("cl:q1:0") is None
    assert adapter._guest_prompt_tap_context("cl:q2:0") is not None


@pytest.mark.asyncio
async def test_releasing_the_turn_forgets_its_prompts(clarify_turn):
    adapter, clarify_gateway, session_key = clarify_turn
    await _draw_clarify(adapter, clarify_gateway, session_key)

    adapter._release_guest_turn("42")

    assert adapter._guest_prompt_taps == {}


# ---------------------------------------------------------------------------
# An unauthorized tap now reaches the participation path, and Allow once means something
# ---------------------------------------------------------------------------

def _participation_clarify_adapter(monkeypatch):
    import tools.clarify_gateway as clarify_gateway

    adapter = _participation_adapter()
    _register_guest_chat(adapter)
    adapter._guest_inline_message_ids["42"] = "imi_author"
    adapter._guest_chat_types["42"] = "supergroup"
    session_key = "agent:main:telegram:42:999"
    monkeypatch.setattr(adapter, "_gateway_session_key", lambda event: session_key)
    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()
    return adapter, clarify_gateway, session_key


@pytest.mark.asyncio
async def test_an_unauthorized_tap_raises_the_participation_card_not_expired(monkeypatch):
    """The existing design, finally reachable: the refusal becomes a request the operator can grant."""
    adapter, clarify_gateway, session_key = _participation_clarify_adapter(monkeypatch)
    await _draw_clarify(adapter, clarify_gateway, session_key)
    query = _foreign_tap(data="cl:c1:0", user_id=OTHER)

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False):
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    toast = query.answer.await_args.kwargs["text"]
    assert "asked my owner" in toast and "expired" not in toast.lower()
    card = adapter._bot.send_message.await_args.kwargs["text"]
    # Names the question AND the button pressed. A clarify keyboard labels its buttons with the
    # choice NUMBER (the question text right beside it lists them), so "1" is the label there; an
    # exec approval or a picker carries a word.
    assert "❓ Which one?" in card and "with “1”" in card


@pytest.mark.asyncio
async def test_allow_once_on_a_tap_lets_exactly_that_tap_through(monkeypatch):
    """Today it answered "nothing left to run", so the person tapped, was refused, and a new card
    appeared — forever. One press is what the operator approved, so one press is what it buys."""
    adapter, clarify_gateway, session_key = _participation_clarify_adapter(monkeypatch)
    await _draw_clarify(adapter, clarify_gateway, session_key)
    with patch.object(adapter, "_is_callback_user_authorized", return_value=False):
        await adapter._handle_callback_query(
            MagicMock(callback_query=_foreign_tap(data="cl:c1:0", user_id=OTHER)), MagicMock())
    approve = _foreign_tap(data=f"gp:o:{_request_id(adapter)}", user_id=OPERATOR, imi="imi_op")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_callback_query(MagicMock(callback_query=approve), MagicMock())

    assert "may press that once" in approve.answer.await_args.kwargs["text"]
    # Their next press of that button answers the question — and says so, rather than asking the
    # operator all over again.
    retry = _foreign_tap(data="cl:c1:0", user_id=OTHER)
    with patch.object(adapter, "_is_callback_user_authorized", return_value=False):
        await adapter._handle_callback_query(MagicMock(callback_query=retry), MagicMock())
    assert clarify_gateway._entries["c1"].response == "Left"
    assert "asked my owner" not in retry.answer.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_the_grant_is_one_press_on_one_prompt(monkeypatch):
    adapter, clarify_gateway, session_key = _participation_clarify_adapter(monkeypatch)
    await _draw_clarify(adapter, clarify_gateway, session_key)
    prompt_key = adapter._guest_prompt_tap_context("cl:c1:0")["prompt_key"]
    adapter._grant_guest_tap(OTHER, prompt_key)

    assert adapter._consume_guest_tap_grant(OTHER, prompt_key) is True
    # Spent by that press.
    assert adapter._consume_guest_tap_grant(OTHER, prompt_key) is False
    # Never another prompt, and never another person.
    adapter._grant_guest_tap(OTHER, prompt_key)
    assert adapter._consume_guest_tap_grant(OTHER, "42#999") is False
    assert adapter._consume_guest_tap_grant(AUTHOR, prompt_key) is False


@pytest.mark.asyncio
async def test_a_grant_dies_with_its_prompt(monkeypatch):
    adapter, clarify_gateway, session_key = _participation_clarify_adapter(monkeypatch)
    await _draw_clarify(adapter, clarify_gateway, session_key)
    prompt_key = adapter._guest_prompt_tap_context("cl:c1:0")["prompt_key"]
    adapter._grant_guest_tap(OTHER, prompt_key)

    adapter._release_guest_turn("42")

    assert adapter._consume_guest_tap_grant(OTHER, prompt_key) is False


@pytest.mark.asyncio
async def test_an_expired_grant_does_not_let_a_tap_through(monkeypatch):
    adapter, clarify_gateway, session_key = _participation_clarify_adapter(monkeypatch)
    await _draw_clarify(adapter, clarify_gateway, session_key)
    prompt_key = adapter._guest_prompt_tap_context("cl:c1:0")["prompt_key"]
    adapter._grant_guest_tap(OTHER, prompt_key)
    slot = adapter._guest_tap_grant_key(OTHER, prompt_key)
    adapter._guest_tap_grants[slot] -= (adapter._GUEST_TAP_GRANT_TTL_SECONDS + 1)

    assert adapter._consume_guest_tap_grant(OTHER, prompt_key) is False


# ---------------------------------------------------------------------------
# An Allow-once turn: its author is the stranger, so every prompt it raises
# was tappable by nobody at all before attribution came from the prompt.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_operator_can_approve_an_exec_prompt_in_an_allow_once_turn(clarify_turn):
    """Worst case of the old bug: the turn's author is the stranger, so the operator's own tap on
    its exec approval was unattributable and nobody could decide it."""
    adapter, _clarify_gateway, session_key = clarify_turn
    adapter._guest_allow_once_authors["42"] = OTHER
    await adapter.send_exec_approval(chat_id="42", command="rm -rf /", session_key=session_key)
    data = next(d for d in adapter._guest_prompt_taps if d.startswith("ea:once:"))
    query = _foreign_tap(data=data, user_id=OPERATOR)

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True), \
         patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    resolve.assert_called_once()
    assert "Approved" in query.answer.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_the_allow_once_author_types_the_answer_to_their_own_question(clarify_turn):
    """"✏️ Other (type answer)" asks for a message, and that message meets the guest gate long
    before the continuation branch — so the opening their buttons have is offered there too."""
    adapter, clarify_gateway, session_key = clarify_turn
    adapter._guest_allow_once_authors["42"] = OTHER
    entry = clarify_gateway.register("c1", session_key, "Which day?", ["Mon", "Tue"])
    clarify_gateway.mark_awaiting_text(entry.clarify_id)
    update, msg = _make_guest_update(
        update_id=71, gqid="gq_typed", text="@testbot next Friday", caller_id=OTHER)
    msg.reply_to_message = None
    enqueued = []

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False), \
         patch.object(adapter, "_apply_telegram_group_observe_attribution", side_effect=lambda e: e), \
         patch.object(adapter, "_enqueue_text_event", side_effect=enqueued.append):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert len(enqueued) == 1 and enqueued[0].text == "next Friday"


@pytest.mark.asyncio
async def test_an_unrelated_message_from_that_author_is_still_refused(clarify_turn):
    """The opening is the question their request asked, not a licence to talk to the bot."""
    adapter, clarify_gateway, session_key = clarify_turn
    adapter._guest_allow_once_authors["42"] = OTHER
    clarify_gateway.register("c1", session_key, "Which day?", ["Mon", "Tue"])  # buttons, not text
    update, msg = _make_guest_update(
        update_id=72, gqid="gq_other", text="@testbot and now delete everything", caller_id=OTHER)
    msg.reply_to_message = None
    enqueued = []

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False), \
         patch.object(adapter, "_enqueue_text_event", side_effect=enqueued.append):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert enqueued == []


@pytest.mark.asyncio
async def test_a_typed_answer_from_someone_else_is_still_refused(clarify_turn):
    """Only the person whose request it is: a bystander cannot answer the turn's question."""
    adapter, clarify_gateway, session_key = clarify_turn
    adapter._guest_allow_once_authors["42"] = OTHER
    entry = clarify_gateway.register("c1", session_key, "Which day?", ["Mon", "Tue"])
    clarify_gateway.mark_awaiting_text(entry.clarify_id)
    update, msg = _make_guest_update(
        update_id=73, gqid="gq_bystander", text="@testbot Tuesday", caller_id="7777")
    msg.reply_to_message = None
    enqueued = []

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False), \
         patch.object(adapter, "_enqueue_text_event", side_effect=enqueued.append):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert enqueued == []


@pytest.mark.asyncio
async def test_an_author_cannot_answer_a_question_from_another_live_turn(monkeypatch):
    """Being allowed one request of your own is not being allowed into someone else's.

    Needs ``guest_thread_sessions``: without it a chat has one session, so two live turns are not
    distinguishable conversations and there is no other turn to cross into.
    """
    import tools.clarify_gateway as clarify_gateway

    adapter = _make_adapter()
    adapter.config.extra["guest_thread_sessions"] = True
    adapter._guest_chat_types["42"] = "supergroup"
    # One session key per conversation, as thread sessions give.
    monkeypatch.setattr(
        adapter, "_gateway_session_key",
        lambda event: f"agent:main:telegram:42:{getattr(event.source, 'thread_id', None)}")
    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()
    # Their own approved conversation, waiting on nothing.
    adapter._register_guest_turn("42", "g1aaaau8446220098", "gq_mine")
    adapter._guest_allow_once_authors["g1aaaau8446220098"] = OTHER
    # Somebody else's conversation, which IS waiting on a typed answer.
    adapter._register_guest_turn("42", "g2bbbbu999", "gq_theirs")
    entry = clarify_gateway.register(
        "c1", "agent:main:telegram:42:g2bbbbu999", "Which day?", ["Mon", "Tue"])
    clarify_gateway.mark_awaiting_text(entry.clarify_id)
    update, msg = _make_guest_update(
        update_id=74, gqid="gq_cross", text="@testbot Tuesday", caller_id=OTHER)
    msg.reply_to_message = None
    enqueued = []

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False), \
         patch.object(adapter, "_enqueue_text_event", side_effect=enqueued.append):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert enqueued == []
    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()
