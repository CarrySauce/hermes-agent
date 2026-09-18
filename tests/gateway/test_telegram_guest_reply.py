"""Integration tests for Telegram guest mode reply flow (Bot API 10.0).

Branch 1 — normal query: stub fires, skill runs, OPC edits stub with text or
           media button.
Branch 2 — deliver_<token> with valid token: answerGuestQuery with cached media,
           no stub, no edit cycle.
Branch 3 — deliver_<token> with invalid/expired token: answerGuestQuery with
           "something went wrong", no stub.

Guest messages arrive as a native ``telegram.Message`` on ``update.guest_message``,
carrying its own ``guest_query_id`` and a normal, populated ``from_user`` — that's
how PTB >=22.8 actually models Bot API 10.0 guest bots (there is no api_kwargs
involved). Every fixture below builds updates that shape, not a raw dict.
"""

import logging
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Telegram library mock
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
import plugins.platforms.telegram.adapter as _tg_adapter_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Real stand-ins for PTB's inline-query/keyboard classes
#
# tests/gateway/conftest.py installs a shared generic MagicMock for the whole
# ``telegram`` package before any test file's imports run (see its
# ``_ensure_telegram_mock``, which wins whenever real PTB hasn't already been
# imported in this process). Calling a MagicMock attribute like
# ``InlineQueryResultArticle(id=..., title=...)`` returns a shared
# auto-generated child mock that does NOT reflect the kwargs it was called
# with — so asserting ``result.id == "thinking"`` would just be inspecting
# mock plumbing, not what the adapter actually built. These lightweight real
# classes give every test something genuinely inspectable regardless of
# whether real PTB or the shared mock is active in this process.
# ---------------------------------------------------------------------------

class _FakeInputTextMessageContent:
    def __init__(self, message_text, **_kw):
        self.message_text = message_text


class _FakeInlineQueryResultArticle:
    def __init__(self, *, id, title, input_message_content, **_kw):
        self.id = id
        self.title = title
        self.input_message_content = input_message_content
        # Kept so the control-prompt tests can assert the keyboard rides along with the
        # one-shot answer (real InlineQueryResultArticle takes reply_markup the same way).
        self.reply_markup = _kw.get("reply_markup")


class _FakeInlineQueryResultCachedVideo:
    def __init__(self, *, id, video_file_id, title=None, **_kw):
        self.id, self.video_file_id, self.title = id, video_file_id, title


class _FakeInlineQueryResultCachedPhoto:
    def __init__(self, *, id, photo_file_id, title=None, **_kw):
        self.id, self.photo_file_id, self.title = id, photo_file_id, title


class _FakeInlineQueryResultCachedAudio:
    def __init__(self, *, id, audio_file_id, title=None, **_kw):
        self.id, self.audio_file_id, self.title = id, audio_file_id, title


class _FakeInlineQueryResultCachedDocument:
    def __init__(self, *, id, document_file_id, title=None, **_kw):
        self.id, self.document_file_id, self.title = id, document_file_id, title


class _FakeInlineQueryResultCachedVoice:
    def __init__(self, *, id, voice_file_id, title=None, **_kw):
        self.id, self.voice_file_id, self.title = id, voice_file_id, title


class _FakeInlineKeyboardButton:
    def __init__(self, text, switch_inline_query_current_chat=None, **_kw):
        self.text = text
        self.switch_inline_query_current_chat = switch_inline_query_current_chat


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard, **_kw):
        self.inline_keyboard = inline_keyboard


@pytest.fixture(autouse=True)
def _real_inline_result_classes(monkeypatch):
    monkeypatch.setattr(_tg_adapter_mod, "InputTextMessageContent", _FakeInputTextMessageContent)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultArticle", _FakeInlineQueryResultArticle)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultCachedVideo", _FakeInlineQueryResultCachedVideo)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultCachedPhoto", _FakeInlineQueryResultCachedPhoto)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultCachedAudio", _FakeInlineQueryResultCachedAudio)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultCachedDocument", _FakeInlineQueryResultCachedDocument)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultCachedVoice", _FakeInlineQueryResultCachedVoice)
    monkeypatch.setattr(_tg_adapter_mod, "InlineKeyboardButton", _FakeInlineKeyboardButton)
    monkeypatch.setattr(_tg_adapter_mod, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> TelegramAdapter:
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {"guest_mode": True}
    adapter = TelegramAdapter(cfg)
    adapter._bot = MagicMock()
    adapter._bot.username = "testbot"  # real str so _clean_bot_trigger_text regex works
    adapter._bot.answer_guest_query = AsyncMock(
        return_value=MagicMock(inline_message_id="imi_abc")
    )
    adapter._bot.edit_message_text = AsyncMock()
    return adapter


def _register_guest_chat(adapter: TelegramAdapter, chat_id="42") -> None:
    """Pre-populate state as if branch-1 processing started."""
    adapter._pending_guest_queries[chat_id] = "gqid_test"
    adapter._guest_only_chats.add(chat_id)
    adapter._guest_inline_message_ids[chat_id] = False


def _make_guest_message(
    caller_id="999", caller_username="someone", caller_first_name=None,
    chat_id="42", chat_type="supergroup", text="@bot hi", gqid="gq1",
):
    """A Message-shaped mock matching PTB's actual guest_message modeling:
    guest_query_id and from_user are both real, direct attributes."""
    msg = MagicMock()
    msg.guest_query_id = gqid
    msg.text = text
    msg.caption = None
    msg.message_id = 1
    # Explicit, not a bare MagicMock: _effective_message_thread_id treats any
    # truthy value (including an unconfigured auto-mock attribute) as a real
    # forum-topic thread id, silently corrupting the session key.
    msg.message_thread_id = None
    msg.is_topic_message = False
    msg.chat = MagicMock(id=int(chat_id), type=chat_type, title="Test Group", is_forum=False)
    msg.from_user = (
        MagicMock(id=int(caller_id), username=caller_username, first_name=caller_first_name, is_bot=False)
        if caller_id is not None else None
    )
    return msg


def _make_guest_update(*, update_id=1, **kwargs):
    msg = _make_guest_message(**kwargs)
    update = MagicMock()
    update.update_id = update_id
    update.guest_message = msg
    return update, msg


# ---------------------------------------------------------------------------
# Branch 1 — stub fires unconditionally on send_typing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch1_stub_fires_on_send_typing():
    """Stub fires on send_typing for any query — no content classification."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)

    with patch.object(adapter, "_guest_fire_text_stub", new_callable=AsyncMock) as mock_stub:
        await adapter.send_typing("42")
        mock_stub.assert_awaited_once_with("42")


@pytest.mark.asyncio
async def test_branch1_stub_fires_for_media_keyword_query():
    """No classification suppression — stub fires even for 'download this video'."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)

    with patch.object(adapter, "_guest_fire_text_stub", new_callable=AsyncMock) as mock_stub:
        await adapter.send_typing("42")
        mock_stub.assert_awaited_once_with("42")


@pytest.mark.asyncio
async def test_guest_fire_text_stub_stores_typed_inline_message_id():
    """The stub fire calls the typed Bot.answer_guest_query and stores the
    (guaranteed-present) inline_message_id straight off the typed result —
    no dict/.get() involved."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._bot.answer_guest_query = AsyncMock(return_value=MagicMock(inline_message_id="imi_live"))

    await adapter._guest_fire_text_stub("42")

    adapter._bot.answer_guest_query.assert_awaited_once()
    call = adapter._bot.answer_guest_query.await_args
    assert call.args[0] == "gqid_test"
    result = call.args[1]
    assert result.id == "thinking"
    assert adapter._guest_inline_message_ids["42"] == "imi_live"


@pytest.mark.asyncio
async def test_guest_fire_text_stub_exception_leaves_imi_none():
    """A raised exception (network error, API rejection) — not a malformed
    response, PTB validates SentGuestMessage's inline_message_id as required —
    is the only way the stub can fail; OPC's no-imi fallback picks it up."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._bot.answer_guest_query = AsyncMock(side_effect=RuntimeError("boom"))

    await adapter._guest_fire_text_stub("42")

    assert adapter._guest_inline_message_ids["42"] is None


# ---------------------------------------------------------------------------
# Branch 1 OPC — text result: edit_message_text on imi
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch1_opc_text_result_edits_stub():
    """OPC text result: edit_message_text(inline_message_id=, text=final_text)."""
    from gateway.platforms.base import ProcessingOutcome

    adapter = _make_adapter()
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    adapter._guest_reply_buffer["42"] = "Here is your answer."
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"
    outcome = ProcessingOutcome.SUCCESS

    await adapter.on_processing_complete(event, outcome)

    adapter._bot.edit_message_text.assert_awaited()
    last_call = adapter._bot.edit_message_text.await_args_list[-1]
    assert last_call.kwargs["inline_message_id"] == "imi_abc"
    assert "Here is your answer." in last_call.kwargs["text"]


@pytest.mark.asyncio
async def test_branch1_opc_truncates_by_utf16_units_not_python_length():
    """A reply with many astral-plane emoji is short in Python string length
    but can exceed Telegram's 4,096 *UTF-16 code unit* limit — each emoji is
    a surrogate pair (2 units). A naive text[:4096] slice would pass this
    reply through untruncated and let the real edit_message_text call fail
    after guest state has already been torn down, silently dropping the
    turn. Every edit_message_text call (typewriter frames and the final edit)
    must stay within the UTF-16 cap."""
    from gateway.platforms.base import ProcessingOutcome, utf16_len

    adapter = _make_adapter()
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    # 2,049 emoji: 2,049 Python chars, but 4,098 UTF-16 units — over the cap
    # despite `len(_plain) < 4096` looking safe.
    adapter._guest_reply_buffer["42"] = "😀" * 2049
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    edit_texts = [c.kwargs["text"] for c in adapter._bot.edit_message_text.await_args_list]
    assert edit_texts, "expected at least one edit_message_text call"
    for text in edit_texts:
        assert utf16_len(text) <= adapter.MAX_MESSAGE_LENGTH, (
            f"edit_message_text frame exceeds UTF-16 cap: {utf16_len(text)} units"
        )


@pytest.mark.asyncio
async def test_branch1_opc_no_imi_falls_back_to_answer_guest_query():
    """If the stub call itself raised (imi never obtained), OPC must still
    answer the guest query directly instead of silently dropping the turn."""
    from gateway.platforms.base import ProcessingOutcome

    adapter = _make_adapter()
    adapter._pending_guest_queries["42"] = "gqid_fallback"
    adapter._guest_inline_message_ids["42"] = None  # stub raised
    adapter._guest_reply_buffer["42"] = "Fallback answer."
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.edit_message_text.assert_not_awaited()
    adapter._bot.answer_guest_query.assert_awaited_once()
    call = adapter._bot.answer_guest_query.await_args
    assert call.args[0] == "gqid_fallback"
    assert "Fallback answer." in call.args[1].input_message_content.message_text


# ---------------------------------------------------------------------------
# Caller authorization gate (fail-closed) — _handle_guest_message_update
#
# The caller for an incoming guest message is msg.from_user, same as any
# other Telegram message. Unauthorized callers are denied before any state
# registration or API call; no from_user / unknown caller => deny.
# ---------------------------------------------------------------------------

import plugins.platforms.telegram.adapter as _tg_adapter_mod  # noqa: E402


@pytest.mark.asyncio
async def test_guest_caller_unauthorized_is_denied():
    """A caller not in any allowlist is denied before state/API."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="999")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False) as mock_auth, \
         patch.object(adapter, "_should_process_message") as mock_should:
        await adapter._handle_guest_message_update(update, MagicMock())

    mock_auth.assert_called_once()
    assert mock_auth.call_args.args[0] == "999"  # caller id from msg.from_user
    assert adapter._pending_guest_queries == {}
    adapter._bot.answer_guest_query.assert_not_called()
    mock_should.assert_not_called()


@pytest.mark.asyncio
async def test_guest_caller_authorized_passes_gate():
    """An authorized caller passes the gate and reaches normal processing."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="123304346")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True) as mock_auth, \
         patch.object(adapter, "_should_process_message", return_value=False) as mock_should:
        await adapter._handle_guest_message_update(update, MagicMock())

    mock_auth.assert_called_once()
    mock_should.assert_called_once()  # gate passed → reached processing


@pytest.mark.asyncio
async def test_guest_missing_from_user_denies_fail_closed():
    """No from_user at all (e.g. an anonymous-admin edge case) => empty
    caller id => real _is_callback_user_authorized denies (no runner, no env
    allowlist)."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id=None)

    await adapter._handle_guest_message_update(update, MagicMock())

    assert adapter._pending_guest_queries == {}
    adapter._bot.answer_guest_query.assert_not_called()


@pytest.mark.asyncio
async def test_guest_allowlisted_caller_via_env_passes(monkeypatch):
    """End-to-end through the real gate: env allowlist authorizes the caller."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="999")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

    with patch.object(adapter, "_should_process_message", return_value=False) as mock_should:
        await adapter._handle_guest_message_update(update, MagicMock())

    mock_should.assert_called_once()  # real gate allowed the env-allowlisted caller


@pytest.mark.asyncio
async def test_guest_message_missing_guest_query_id_is_ignored(monkeypatch):
    """update.guest_message present but with an empty guest_query_id must not
    be treated as a normal message — no state registered, no API call."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="999", gqid="")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

    await adapter._handle_guest_message_update(update, MagicMock())

    assert adapter._pending_guest_queries == {}
    adapter._bot.answer_guest_query.assert_not_called()


@pytest.mark.asyncio
async def test_non_guest_update_is_ignored():
    """A regular update (update.guest_message is None, PTB's default for any
    non-guest update) must be a pure no-op — this is the fix's core contract:
    the handler no longer reaches into api_kwargs at all."""
    adapter = _make_adapter()
    update = MagicMock()
    update.guest_message = None

    await adapter._handle_guest_message_update(update, MagicMock())

    assert adapter._pending_guest_queries == {}
    adapter._bot.answer_guest_query.assert_not_called()


# ---------------------------------------------------------------------------
# Session isolation per guest caller — different callers in the same chat must
# not share a session (context bleed). This now falls entirely out of the
# unpatched _build_message_event, which reads user_id/user_name straight off
# msg.from_user — there is no guest-specific stamping code left to test
# separately from that.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guest_session_isolated_per_caller(monkeypatch):
    from gateway.session import SessionSource, build_session_key
    from gateway.config import Platform

    adapter = _make_adapter()
    adapter._bot.username = "testbot"
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

    update, msg = _make_guest_update(caller_id="999", chat_id="42")
    captured = {}

    with patch.object(adapter, "_should_process_message", return_value=True), \
         patch.object(adapter, "_apply_telegram_group_observe_attribution", side_effect=lambda e: e), \
         patch.object(adapter, "_enqueue_text_event", side_effect=lambda e: captured.update(event=e)):
        await adapter._handle_guest_message_update(update, MagicMock())

    # from_user flows straight into the session source via _build_message_event.
    assert captured["event"].source.user_id == "999"
    key = build_session_key(captured["event"].source, group_sessions_per_user=True)
    assert key.endswith(":42:999")

    # Two different callers in the same chat get distinct sessions (no bleed).
    k_a = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="group", user_id="999"),
        group_sessions_per_user=True,
    )
    k_b = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="group", user_id="888"),
        group_sessions_per_user=True,
    )
    assert k_a != k_b


# ---------------------------------------------------------------------------
# Branch 1 OPC — media result: edit_message_text with deliver button
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch1_opc_media_result_edits_stub_with_button():
    """OPC media result: edit_message_text with switch_inline_query_current_chat button."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()

    from gateway.platforms.base import ProcessingOutcome

    adapter = _make_adapter()
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    adapter._guest_reply_buffer["42"] = "Here is your video."
    adapter._guest_turn_media["42"] = {"file_id": "fid_video", "media_kind": "video"}
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.edit_message_text.assert_awaited()
    call = adapter._bot.edit_message_text.await_args_list[-1]
    assert call.kwargs["inline_message_id"] == "imi_abc"
    assert "✅ Ready" in call.kwargs["text"]
    markup = call.kwargs.get("reply_markup")
    assert markup is not None, "reply_markup must be present"
    button = markup.inline_keyboard[0][0]
    assert button.switch_inline_query_current_chat.startswith("deliver_")

    # Token must exist in store
    token = button.switch_inline_query_current_chat[len("deliver_"):]
    assert token in gmt._TOKEN_STORE
    assert gmt._TOKEN_STORE[token]["media_kind"] == "video"

    gmt._TOKEN_STORE.clear()


# ---------------------------------------------------------------------------
# Branch 2 — valid token: answerGuestQuery with cached media, no stub
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch2_valid_token_answers_with_cached_media():
    """Branch 2: deliver_<token> with a valid token dispatches straight to
    answer_guest_query with the cached media — no stub, no LLM pass.

    Invokes the real handler (not a hand-built API call) so token
    resolution, caller authorization, and dispatch are all actually
    exercised, not just that the mock records whatever call the test made
    directly."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_video", "video")

    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="123304346", gqid="gqid_branch2", text=f"deliver_{token}")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_guest_message_update(update, MagicMock())

    adapter._bot.answer_guest_query.assert_awaited_once()
    call = adapter._bot.answer_guest_query.await_args
    assert call.args[0] == "gqid_branch2"
    result = call.args[1]
    assert result.id == "delivery"
    assert result.video_file_id == "fid_video"
    # No stub / no edit cycle — Branch 2 never touches guest turn state.
    assert adapter._pending_guest_queries == {}

    gmt._TOKEN_STORE.clear()


@pytest.mark.asyncio
async def test_deliver_token_denied_for_unauthorized_caller():
    """A valid, unexpired token must still be denied if the caller isn't
    authorized — the caller gate sits in front of the deliver_<token>
    branch too, so a leaked token can't be redeemed by a stranger."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_video", "video")

    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="999", gqid="gqid_branch2", text=f"deliver_{token}")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False):
        await adapter._handle_guest_message_update(update, MagicMock())

    adapter._bot.answer_guest_query.assert_not_called()

    gmt._TOKEN_STORE.clear()


# ---------------------------------------------------------------------------
# Branch 3 — invalid/expired token: "something went wrong" result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch3_expired_token_answers_with_error():
    """Branch 3: deliver_<token> with an expired token answers with
    "something went wrong" instead of the cached media — via the real
    handler, not a hand-built API call."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid", "video")
    gmt._TOKEN_STORE[token]["expires_at"] = time.monotonic() - 1  # force expiry

    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="123304346", gqid="gqid_branch3", text=f"deliver_{token}")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_guest_message_update(update, MagicMock())

    adapter._bot.answer_guest_query.assert_awaited_once()
    call = adapter._bot.answer_guest_query.await_args
    assert call.args[0] == "gqid_branch3"
    result = call.args[1]
    assert "wrong" in result.input_message_content.message_text.lower()

    gmt._TOKEN_STORE.clear()


# ---------------------------------------------------------------------------
# Callback handler — no delivery attempt, only answerCallbackQuery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_callback_on_stub_only_dismisses_loading():
    """Button press on stub produces callback_query — handler must not call answer_guest_query."""
    adapter = _make_adapter()

    cq = MagicMock()
    cq.id = "cq_123"
    cq.inline_message_id = "imi_abc"
    cq.answer = AsyncMock()

    await cq.answer()

    cq.answer.assert_awaited_once()
    adapter._bot.answer_guest_query.assert_not_called()


# ---------------------------------------------------------------------------
# _guest_media_send — hard path containment for the MEDIA: staging flow
#
# Unaffected by the PTB-native-API modernization: this staging path always
# used typed Bot.send_photo/send_video/send_audio/send_document (never
# do_api_request), so nothing here changes.
#
# The delivery-constraint prompt *tells* the LLM to stage under
# HERMES_HOME/cache/<subdir>, but that was only a prompt-level instruction --
# nothing enforced it, so a guest-triggered turn coerced into requesting an
# arbitrary host path (e.g. the credentials store) would previously have had
# it staged and made deliverable to the guest chat. These tests cover the
# hard containment check added to _guest_media_send.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guest_media_send_rejects_path_outside_staging_root(tmp_path, monkeypatch):
    """A path outside HERMES_HOME/cache is rejected before any open()/upload."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100999")

    outside_file = tmp_path / "auth.json"
    outside_file.write_text('{"api_key": "secret"}')

    adapter = _make_adapter()
    adapter._bot.send_document = AsyncMock()

    result = await adapter._guest_media_send("42", "document", str(outside_file))

    assert result.success is False
    assert "outside the allowed staging directory" in result.error
    adapter._bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_guest_media_send_allows_path_inside_staging_root(tmp_path, monkeypatch):
    """A path inside HERMES_HOME/cache proceeds to staging as before."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100999")

    cache_dir = tmp_path / "cache" / "videos"
    cache_dir.mkdir(parents=True)
    video_file = cache_dir / "clip.mp4"
    video_file.write_bytes(b"fake video bytes")

    adapter = _make_adapter()

    sent_video = MagicMock()
    sent_video.video = MagicMock(file_id="fid_ok")
    adapter._bot.send_video = AsyncMock(return_value=sent_video)

    result = await adapter._guest_media_send("42", "video", str(video_file))

    assert result.success is True
    adapter._bot.send_video.assert_awaited_once()
    assert adapter._guest_turn_media["42"]["file_id"] == "fid_ok"


@pytest.mark.asyncio
async def test_guest_media_send_null_byte_path_fails_cleanly(tmp_path, monkeypatch):
    """A path resolve() itself chokes on (embedded null byte) returns a clean
    failure SendResult instead of raising out of the adapter.

    Regression: resolution and containment used to share one try block, so a
    null-byte path entered the ValueError ("outside staging root") handler
    with _abs_resolved unbound — the rejection log line itself then raised
    UnboundLocalError, escaping the SendResult contract entirely.
    """
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100999")

    adapter = _make_adapter()
    adapter._bot.send_document = AsyncMock()

    result = await adapter._guest_media_send(
        "42", "document", str(tmp_path / "cache") + "/evil\x00.pdf"
    )

    assert result.success is False
    assert "path validation failed" in result.error
    adapter._bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_guest_media_send_restages_when_file_content_changes(tmp_path, monkeypatch):
    """Re-generating a file at the same path (new mtime) re-stages and mints a
    fresh file_id instead of serving the stale cached one."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100999")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    photo = cache_dir / "chart.png"
    photo.write_bytes(b"v1")
    os.utime(photo, (1000, 1000))

    adapter = _make_adapter()
    fids = iter(["fid_v1", "fid_v2"])

    def _mint(*a, **kw):
        sent = MagicMock()
        sent.photo = [MagicMock(file_id=next(fids))]
        return sent

    adapter._bot.send_photo = AsyncMock(side_effect=_mint)

    await adapter._guest_media_send("42", "photo", str(photo))
    assert adapter._guest_turn_media["42"]["file_id"] == "fid_v1"

    # Same path, same mtime -> served from cache, no second upload.
    await adapter._guest_media_send("42", "photo", str(photo))
    assert adapter._bot.send_photo.await_count == 1

    # Same path, new content/mtime -> re-staged, fresh file_id.
    photo.write_bytes(b"v2 -- regenerated")
    os.utime(photo, (2000, 2000))
    await adapter._guest_media_send("42", "photo", str(photo))
    assert adapter._bot.send_photo.await_count == 2
    assert adapter._guest_turn_media["42"]["file_id"] == "fid_v2"


# ---------------------------------------------------------------------------
# Inbound attachments — a guest @mention that carries a photo/file must take the
# same caching path a DM takes, not be dropped for having no text.
# ---------------------------------------------------------------------------

def _make_guest_media_message(
    kind="photo", caption="@testbot what is this?", chat_id="42", gqid="gq_media",
    caller_id="999", media_group_id=None,
):
    """A guest Message carrying an attachment.

    Every media attribute is set explicitly: on a bare MagicMock they would all
    auto-create as truthy, and the media pipeline picks the FIRST present one.
    """
    msg = _make_guest_message(caller_id=caller_id, chat_id=chat_id, gqid=gqid, text=None)
    msg.text = None
    msg.caption = caption
    msg.entities = []
    msg.caption_entities = []
    msg.media_group_id = media_group_id
    for attr in ("photo", "video", "audio", "voice", "document", "sticker", "animation", "video_note"):
        setattr(msg, attr, None)
    if kind == "photo":
        msg.photo = [MagicMock(file_id="in_photo", file_size=1024)]
    else:
        setattr(msg, kind, MagicMock(file_id=f"in_{kind}", file_size=1024))
    return msg


def _make_guest_media_update(*, update_id=11, **kwargs):
    msg = _make_guest_media_message(**kwargs)
    update = MagicMock()
    update.update_id = update_id
    update.guest_message = msg
    return update, msg


@pytest.mark.asyncio
async def test_guest_photo_routes_through_the_shared_media_pipeline():
    """A captioned photo becomes a PHOTO-typed event on the same caching path as a DM."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()
    update, msg = _make_guest_media_update(kind="photo")
    seen = {}

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True), \
         patch.object(adapter, "_should_process_message", return_value=True), \
         patch.object(adapter, "_apply_telegram_group_observe_attribution", side_effect=lambda e: e), \
         patch.object(adapter, "_cache_and_route_media",
                      new=AsyncMock(side_effect=lambda m, e: seen.update(msg=m, event=e))):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert seen["msg"] is msg
    assert seen["event"].message_type == MessageType.PHOTO
    # The stub fires BEFORE the download so the caller isn't left staring at silence
    # while the file is fetched and vision runs.
    adapter._bot.answer_guest_query.assert_awaited_once()
    assert adapter._guest_inline_message_ids["42"] == "imi_abc"


@pytest.mark.asyncio
async def test_guest_caption_less_document_is_not_dropped_as_empty():
    """No text and no caption: the attachment itself is the request."""
    adapter = _make_adapter()
    update, msg = _make_guest_media_update(kind="document", caption=None)
    routed = []

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True), \
         patch.object(adapter, "_should_process_message", return_value=True), \
         patch.object(adapter, "_apply_telegram_group_observe_attribution", side_effect=lambda e: e), \
         patch.object(adapter, "_cache_and_route_media", new=AsyncMock(side_effect=lambda m, e: routed.append(e))):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert len(routed) == 1


@pytest.mark.asyncio
async def test_guest_album_continuation_merges_instead_of_busy_reply():
    """An album is one request: its 2nd..Nth guest messages join the in-flight turn."""
    adapter = _make_adapter()
    adapter._pending_guest_queries["42"] = "gq_first"
    adapter._guest_only_chats.add("42")
    adapter._guest_media_group_ids["42"] = "album1"
    update, msg = _make_guest_media_update(
        kind="photo", caption=None, gqid="gq_second", media_group_id="album1", update_id=12)
    routed = []

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True), \
         patch.object(adapter, "_cache_and_route_media", new=AsyncMock(side_effect=lambda m, e: routed.append(e))):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert len(routed) == 1
    # No busy reply, and the first item's query id (the one that owns the stub) is untouched.
    adapter._bot.answer_guest_query.assert_not_called()
    assert adapter._pending_guest_queries["42"] == "gq_first"


@pytest.mark.asyncio
async def test_guest_second_unrelated_message_still_gets_the_busy_reply():
    """The album exemption is narrow: an unrelated second ask is still rejected."""
    adapter = _make_adapter()
    adapter._pending_guest_queries["42"] = "gq_first"
    adapter._guest_only_chats.add("42")
    update, msg = _make_guest_update(update_id=13, gqid="gq_second", text="@testbot another one")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_guest_message_update(update, MagicMock())

    adapter._bot.answer_guest_query.assert_awaited_once()
    assert adapter._bot.answer_guest_query.await_args.args[1].id == "busy"
    assert adapter._pending_guest_queries["42"] == "gq_first"


# ---------------------------------------------------------------------------
# Outbound attachments — the native send paths divert into staging for guest chats
# instead of calling sendDocument/sendPhoto on a chat the bot isn't a member of.
# ---------------------------------------------------------------------------

def _staged_file(tmp_path, monkeypatch, name="report.pdf", data=b"pdf bytes"):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100999")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    path.write_bytes(data)
    return path


@pytest.mark.asyncio
async def test_send_document_in_guest_chat_stages_instead_of_sending(tmp_path, monkeypatch):
    """send_document must not reach the guest chat: it stages and records the file_id."""
    path = _staged_file(tmp_path, monkeypatch)
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    sent = MagicMock()
    sent.document = MagicMock(file_id="fid_doc")
    adapter._bot.send_document = AsyncMock(return_value=sent)

    result = await adapter.send_document("42", str(path))

    assert result.success is True
    assert adapter._guest_turn_media["42"] == {
        "file_id": "fid_doc", "media_kind": "document", "caption": None, "file_name": "report.pdf"}
    # The upload went to the staging channel, never to the guest chat.
    assert str(adapter._bot.send_document.await_args.kwargs["chat_id"]) == "-100999"


@pytest.mark.asyncio
async def test_send_multiple_images_in_guest_chat_stages_each(tmp_path, monkeypatch):
    """The album path stages every local image rather than calling send_media_group."""
    from urllib.parse import quote

    first = _staged_file(tmp_path, monkeypatch, name="a.png", data=b"a")
    second = _staged_file(tmp_path, monkeypatch, name="b.png", data=b"b")
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    fids = iter(["fid_a", "fid_b"])
    adapter._bot.send_photo = AsyncMock(
        side_effect=lambda **kw: MagicMock(photo=[MagicMock(file_id=next(fids))]))
    adapter._bot.send_media_group = AsyncMock()

    result = await adapter.send_multiple_images(
        "42", [(f"file://{quote(str(first))}", "first"), (f"file://{quote(str(second))}", "second")])

    assert result.success is True
    adapter._bot.send_media_group.assert_not_called()
    assert [r["file_id"] for r in adapter._guest_turn_media_all["42"]] == ["fid_a", "fid_b"]


@pytest.mark.asyncio
async def test_guest_media_send_without_staging_chat_fails_without_uploading(tmp_path, monkeypatch):
    """No home channel configured: refuse cleanly rather than uploading somewhere unintended."""
    path = _staged_file(tmp_path, monkeypatch, name="clip.mp4", data=b"v")
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    adapter = _make_adapter()
    adapter.config.home_channel = None
    adapter._bot.send_video = AsyncMock()

    result = await adapter._guest_media_send("42", "video", str(path))

    assert result.success is False
    assert "staging chat" in result.error
    adapter._bot.send_video.assert_not_called()


# ---------------------------------------------------------------------------
# Delivery tokens
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delivery_token_is_single_use():
    """A redeemed token is spent: replaying the same deliver_ message gets the error result."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_photo", "photo")

    adapter = _make_adapter()
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        first, _ = _make_guest_update(update_id=21, gqid="gq_a", text=f"deliver_{token}")
        await adapter._handle_guest_message_update(first, MagicMock())
        second, _ = _make_guest_update(update_id=22, gqid="gq_b", text=f"deliver_{token}")
        await adapter._handle_guest_message_update(second, MagicMock())

    assert adapter._bot.answer_guest_query.await_count == 2
    replay = adapter._bot.answer_guest_query.await_args_list[-1].args[1]
    assert "wrong" in replay.input_message_content.message_text.lower()
    gmt._TOKEN_STORE.clear()


@pytest.mark.asyncio
async def test_opc_without_imi_answers_with_the_media_itself():
    """Stub never landed, so the query slot is still unspent — spend it on the file."""
    from gateway.platforms.base import ProcessingOutcome

    adapter = _make_adapter()
    adapter._pending_guest_queries["42"] = "gq_live"
    adapter._guest_inline_message_ids["42"] = None  # stub fired, Telegram returned no imi
    adapter._guest_reply_buffer["42"] = "Here is the chart."
    adapter._guest_turn_media["42"] = {"file_id": "fid_chart", "media_kind": "photo", "file_name": "chart.png"}
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.answer_guest_query.assert_awaited_once()
    result = adapter._bot.answer_guest_query.await_args.args[1]
    assert result.photo_file_id == "fid_chart"
    adapter._bot.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_delivery_payload_survives_the_prefilled_at_mention():
    """switch_inline_query_current_chat pre-fills "@bot deliver_<token>" — still a redemption."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_doc", "document", file_name="report.pdf")

    adapter = _make_adapter()
    update, _ = _make_guest_update(update_id=31, gqid="gq_mention", text=f"@testbot deliver_{token}")
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_guest_message_update(update, MagicMock())

    result = adapter._bot.answer_guest_query.await_args.args[1]
    assert result.document_file_id == "fid_doc"
    # Redemption is a handover, not a turn: no LLM state was registered.
    assert adapter._pending_guest_queries == {}
    gmt._TOKEN_STORE.clear()


@pytest.mark.asyncio
async def test_prose_starting_with_deliver_is_not_a_redemption():
    """"deliver_..." only counts in the exact button payload shape."""
    adapter = _make_adapter()
    assert adapter._guest_delivery_token("deliver_ the package tomorrow") is None
    assert adapter._guest_delivery_token("please deliver_abcdefghij") is None
    assert adapter._guest_delivery_token("deliver_abcdefghij") == "abcdefghij"


@pytest.mark.asyncio
async def test_inline_query_redeems_a_delivery_token():
    """With inline mode on, the button's payload arrives as an inline query instead."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_photo", "photo")

    adapter = _make_adapter()
    inline_query = MagicMock()
    inline_query.query = f"deliver_{token}"
    inline_query.from_user = MagicMock(id=123, username="someone")
    inline_query.answer = AsyncMock()
    update = MagicMock(inline_query=inline_query)

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_inline_query(update, MagicMock())

    inline_query.answer.assert_awaited_once()
    results = inline_query.answer.await_args.args[0]
    assert len(results) == 1 and results[0].photo_file_id == "fid_photo"
    # Spent: the same token yields an empty answer, not a second copy of the file.
    inline_query.answer.reset_mock()
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_inline_query(update, MagicMock())
    assert inline_query.answer.await_args.args[0] == []
    gmt._TOKEN_STORE.clear()


@pytest.mark.asyncio
async def test_inline_query_delivery_is_denied_for_unauthorized_users():
    """The inline redemption path sits behind the same caller gate as everything else."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_photo", "photo")

    adapter = _make_adapter()
    inline_query = MagicMock()
    inline_query.query = f"deliver_{token}"
    inline_query.from_user = MagicMock(id=999, username="stranger")
    inline_query.answer = AsyncMock()

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False):
        await adapter._handle_inline_query(MagicMock(inline_query=inline_query), MagicMock())

    assert inline_query.answer.await_args.args[0] == []
    # Denied, not consumed: the rightful caller can still redeem it.
    assert token in gmt._TOKEN_STORE
    gmt._TOKEN_STORE.clear()


@pytest.mark.asyncio
async def test_oversized_photo_staging_falls_back_to_document(tmp_path, monkeypatch):
    """sendPhoto caps at ~10 MB; a refused photo is re-staged as a document, not dropped."""
    path = _staged_file(tmp_path, monkeypatch, name="huge.png", data=b"x" * 32)
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._bot.send_photo = AsyncMock(side_effect=RuntimeError("Photo_invalid_dimensions"))
    sent_doc = MagicMock()
    sent_doc.document = MagicMock(file_id="fid_as_doc")
    adapter._bot.send_document = AsyncMock(return_value=sent_doc)

    result = await adapter._guest_media_send("42", "photo", str(path))

    assert result.success is True
    assert adapter._guest_turn_media["42"]["file_id"] == "fid_as_doc"
    assert adapter._guest_turn_media["42"]["media_kind"] == "document"


# ---------------------------------------------------------------------------
# Control prompts (clarify / approval / pickers) in a guest chat
#
# Every one of them funnels through _send_prompt → _send_control_message →
# sendMessage, which a guest chat rejects with "Forbidden: bot is not a member".
# The turn then hung on a prompt that never rendered. A guest prompt is an edit of
# the inline message already on screen.
# ---------------------------------------------------------------------------

def _clarify_kwargs(**over):
    kwargs = dict(chat_id="42", question="Confirm the reminder?", choices=["Yes", "No"],
                  clarify_id="c1", session_key="s1")
    kwargs.update(over)
    return kwargs


@pytest.mark.asyncio
async def test_guest_clarify_edits_stub_instead_of_sending():
    """The prompt and its buttons land by editing the stub — no sendMessage at all."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    adapter._bot.send_message = AsyncMock()

    result = await adapter.send_clarify(**_clarify_kwargs())

    assert result.success is True
    adapter._bot.send_message.assert_not_called()
    call = adapter._bot.edit_message_text.await_args
    assert call.kwargs["inline_message_id"] == "imi_abc"
    assert call.kwargs["reply_markup"] is not None
    assert "Confirm the reminder?" in call.kwargs["text"]
    # on_sent still runs, so the tap can be resolved against the session.
    assert adapter._clarify_state["c1"] == "s1"


@pytest.mark.asyncio
async def test_guest_prompt_without_a_stub_spends_the_query_on_itself():
    """Nothing drawn yet: answer the one-shot query with the prompt rather than a stub."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)  # leaves the sentinel at False = stub not fired
    adapter._bot.answer_guest_query = AsyncMock(return_value=MagicMock(inline_message_id="imi_new"))

    result = await adapter.send_clarify(**_clarify_kwargs())

    assert result.success is True
    adapter._bot.edit_message_text.assert_not_called()
    answered = adapter._bot.answer_guest_query.await_args.args[1]
    assert "Confirm the reminder?" in answered.input_message_content.message_text
    assert answered.reply_markup is not None
    # Later edits (the tap, then OPC's final reply) target that same message.
    assert adapter._guest_inline_message_ids["42"] == "imi_new"


@pytest.mark.asyncio
async def test_guest_prompt_without_any_surface_fails_definitively():
    """No inline message and no usable query: say so, so the caller doesn't wait forever."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._guest_inline_message_ids["42"] = None  # stub fired, Telegram returned no imi
    adapter._bot.answer_guest_query = AsyncMock(side_effect=RuntimeError("query expired"))

    result = await adapter.send_clarify(**_clarify_kwargs())

    assert result.success is False
    assert result.error == "guest_no_inline_message"
    # The dead query id is dropped, so the gateway's plain-text fallback fails too instead of
    # buffering into a void and reporting the prompt delivered.
    assert "42" not in adapter._pending_guest_queries
    fallback = await adapter.send("42", "❓ Confirm the reminder?")
    assert fallback.success is False
    # Names the reason: this is the retry of a prompt that could not be drawn, and the turn is
    # blocked on its answer, so buffering it would park the waiter on an invisible question.
    assert fallback.error == "guest_prompt_undeliverable"
    # The flag is one-shot (one retry), and the no-surface check still refuses what follows.
    assert (await adapter.send("42", "anything else")).error == "guest_no_inline_message"


@pytest.mark.asyncio
async def test_guest_exec_approval_prompt_also_edits_the_stub():
    """The fix sits in the shared prompt shell, so the whole family is covered."""
    from gateway.platforms.base import ExecApprovalPrompt

    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    adapter._bot.send_message = AsyncMock()
    prompt = ExecApprovalPrompt(
        chat_id="42", session_key="s1", text="⚠️ Run <pre>rm -rf /tmp/x</pre>?",
        actions=[("✅ Allow Once", "once", "primary"), ("❌ Deny", "deny", "danger")],
        command="rm -rf /tmp/x", description="cleanup", smart_denied=False, metadata=None)

    result = await adapter._send_exec_approval_prompt(prompt)

    assert result.success is True
    adapter._bot.send_message.assert_not_called()
    assert adapter._bot.edit_message_text.await_args.kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_non_guest_control_prompt_is_unchanged():
    """Regression guard: an ordinary chat still gets a plain sendMessage."""
    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=7))

    result = await adapter.send_clarify(**_clarify_kwargs(chat_id="999"))

    assert result.success is True and result.message_id == "7"
    adapter._bot.send_message.assert_awaited_once()
    adapter._bot.edit_message_text.assert_not_called()


# ---------------------------------------------------------------------------
# Button taps on an inline message (query.message is None)
# ---------------------------------------------------------------------------

def _inline_query_tap(adapter, *, imi="imi_abc", data="cl:c1:other", user_id=999):
    query = MagicMock()
    query.message = None
    query.inline_message_id = imi
    query.data = data
    query.from_user = MagicMock(id=user_id, first_name="Asker", username="asker")
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


@pytest.mark.asyncio
async def test_guest_callback_ctx_recovers_the_chat_from_the_inline_message():
    """A tap carries no chat; the mapping recorded when the prompt was drawn supplies it."""
    adapter = _make_adapter()
    adapter._guest_chat_types["42"] = "supergroup"
    adapter._remember_guest_inline_message("42", "imi_abc", prompt_text="❓ Confirm?")

    ctx = adapter._callback_ctx(_inline_query_tap(adapter))

    assert ctx["chat_id"] == "42"
    assert ctx["chat_type"] == "supergroup"
    assert ctx["thread_id"] is None


@pytest.mark.asyncio
async def test_unattributable_inline_tap_is_refused_fail_closed():
    """No mapping → no auth context → refuse, rather than evaluate it as a DM."""
    adapter = _make_adapter()
    query = _inline_query_tap(adapter, imi="imi_unknown")

    with patch.object(adapter, "_handle_clarify_callback", new=AsyncMock()) as handler:
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    handler.assert_not_called()
    query.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_guest_clarify_other_branch_does_not_raise_without_message():
    """The ✏️ Other branch used to dereference query.message.text and blow up on every guest tap."""
    import tools.clarify_gateway as clarify_gateway

    adapter = _make_adapter()
    adapter._guest_chat_types["42"] = "supergroup"
    adapter._remember_guest_inline_message("42", "imi_abc", prompt_text="❓ Confirm the reminder?")
    adapter._clarify_state["c1"] = "s1"
    query = _inline_query_tap(adapter)

    with patch.object(adapter, "_callback_authorized", new=AsyncMock(return_value=True)), \
         patch.object(clarify_gateway, "mark_awaiting_text", return_value=True):
        await adapter._handle_clarify_callback(query, "cl:c1:other", adapter._callback_ctx(query))

    query.answer.assert_awaited()
    edited = query.edit_message_text.await_args.kwargs["text"]
    assert "Confirm the reminder?" in edited
    assert "Awaiting typed response" in edited


@pytest.mark.asyncio
async def test_guest_picker_tap_reaches_its_handler():
    """The picker dispatch read the chat off query.message, so guest taps did nothing at all."""
    adapter = _make_adapter()
    adapter._guest_chat_types["42"] = "supergroup"
    adapter._remember_guest_inline_message("42", "imi_abc")
    query = _inline_query_tap(adapter, data="cp:0")

    with patch.object(adapter, "_handle_choice_picker_callback", new=AsyncMock()) as handler:
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    handler.assert_awaited_once()
    assert handler.await_args.args[2] == "42"


# ---------------------------------------------------------------------------
# Typed clarify answers ("✏️ Other", and open-ended prompts)
#
# The turn is blocked in wait_for_response holding _pending_guest_queries[chat],
# and the busy guard rejected inbound messages BECAUSE it holds it: the answer the
# turn was waiting for could never reach it. A continuation is not a new request.
# ---------------------------------------------------------------------------

@pytest.fixture
def clarify_session(monkeypatch):
    """A guest turn in flight with a pending clarify registered under its session key."""
    import tools.clarify_gateway as clarify_gateway

    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    adapter._guest_chat_types["42"] = "supergroup"

    session_key = "agent:main:telegram:42:999"
    monkeypatch.setattr(adapter, "_gateway_session_key", lambda event: session_key)
    for cid in list(clarify_gateway._entries):
        clarify_gateway._entries.pop(cid, None)
    clarify_gateway._session_index.clear()
    yield adapter, clarify_gateway, session_key
    clarify_gateway.clear_session(session_key)
    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()


async def _deliver_guest_text(adapter, text, *, update_id, gqid):
    update, _msg = _make_guest_update(update_id=update_id, gqid=gqid, text=text)
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True), \
         patch.object(adapter, "_should_process_message", return_value=True), \
         patch.object(adapter, "_apply_telegram_group_observe_attribution", side_effect=lambda e: e), \
         patch.object(adapter, "_enqueue_text_event") as enqueued:
        await adapter._handle_guest_message_update(update, MagicMock())
    return enqueued


@pytest.mark.asyncio
async def test_typed_answer_after_other_reaches_the_in_flight_turn(clarify_session):
    """The ✏️ Other path: the answer is routed into the waiting turn, not answered "busy"."""
    adapter, clarify_gateway, session_key = clarify_session
    entry = clarify_gateway.register("c1", session_key, "Which day?", ["Mon", "Tue"])
    clarify_gateway.mark_awaiting_text(entry.clarify_id)

    enqueued = await _deliver_guest_text(adapter, "@testbot next Friday", update_id=51, gqid="gq_answer")

    enqueued.assert_called_once()
    assert enqueued.call_args.args[0].text == "next Friday"
    # The caller's own message is acknowledged, and it is NOT the busy reply.
    answered = adapter._bot.answer_guest_query.await_args.args[1]
    assert answered.id == "clarify-ack"
    # The turn keeps the surface it has been writing to all along.
    assert adapter._pending_guest_queries["42"] == "gqid_test"
    assert adapter._guest_inline_message_ids["42"] == "imi_abc"


@pytest.mark.asyncio
async def test_open_ended_clarify_answer_reaches_the_turn(clarify_session):
    """An open-ended clarify registers awaiting_text=True and has no button to fall back on."""
    adapter, clarify_gateway, session_key = clarify_session
    clarify_gateway.register("c2", session_key, "What should I call it?", None)

    enqueued = await _deliver_guest_text(adapter, "@testbot call it Atlas", update_id=52, gqid="gq_open")

    enqueued.assert_called_once()
    assert enqueued.call_args.args[0].text == "call it Atlas"


@pytest.mark.asyncio
async def test_numeric_pick_typed_at_an_open_choice_prompt_is_a_continuation(clarify_session):
    """Typing "2" at a prompt whose buttons are still showing resolves it, per the intercept."""
    adapter, clarify_gateway, session_key = clarify_session
    clarify_gateway.register("c3", session_key, "Which day?", ["Mon", "Tue"])

    enqueued = await _deliver_guest_text(adapter, "@testbot 2", update_id=53, gqid="gq_pick")

    enqueued.assert_called_once()


@pytest.mark.asyncio
async def test_second_unrelated_request_during_a_clarify_still_gets_the_busy_reply(clarify_session):
    """Regression guard for the guard: prose the clarify would decline is not a continuation."""
    adapter, clarify_gateway, session_key = clarify_session
    # Choices, not awaiting_text: unmatched prose is TEXT_REJECTED_PROSE, which the gateway
    # intercept declines — routing it here would leave it with no surface for its own reply.
    clarify_gateway.register("c4", session_key, "Which day?", ["Mon", "Tue"])

    enqueued = await _deliver_guest_text(
        adapter, "@testbot what's the weather in Lisbon?", update_id=54, gqid="gq_other")

    enqueued.assert_not_called()
    assert adapter._bot.answer_guest_query.await_args.args[1].id == "busy"


@pytest.mark.asyncio
async def test_slash_command_at_a_clarify_is_not_swallowed_as_an_answer(clarify_session):
    """The intercept lets slash commands fall through, so they must not bypass the guard."""
    adapter, clarify_gateway, session_key = clarify_session
    entry = clarify_gateway.register("c5", session_key, "Which day?", None)
    assert entry.awaiting_text is True  # would otherwise accept any text

    enqueued = await _deliver_guest_text(adapter, "@testbot /status", update_id=55, gqid="gq_slash")

    enqueued.assert_not_called()
    assert adapter._bot.answer_guest_query.await_args.args[1].id == "busy"
    assert clarify_gateway.has_pending(session_key) is True


@pytest.mark.asyncio
async def test_second_request_without_any_pending_clarify_still_gets_the_busy_reply(clarify_session):
    """Nothing pending: the busy guard keeps protecting the in-flight turn's state."""
    adapter, _clarify_gateway, _session_key = clarify_session

    enqueued = await _deliver_guest_text(adapter, "@testbot and another thing", update_id=56, gqid="gq_new")

    enqueued.assert_not_called()
    assert adapter._bot.answer_guest_query.await_args.args[1].id == "busy"


@pytest.mark.asyncio
async def test_answer_from_another_caller_does_not_reach_the_clarify(clarify_session, monkeypatch):
    """Per-caller session keys: someone else's text is not the answer this turn is waiting for."""
    adapter, clarify_gateway, session_key = clarify_session
    clarify_gateway.register("c6", session_key, "Which day?", None)
    # The other caller's own session key — which has no pending clarify.
    monkeypatch.setattr(adapter, "_gateway_session_key", lambda event: "agent:main:telegram:42:1234")

    enqueued = await _deliver_guest_text(adapter, "@testbot Tuesday", update_id=57, gqid="gq_stranger")

    enqueued.assert_not_called()
    assert adapter._bot.answer_guest_query.await_args.args[1].id == "busy"
    assert clarify_gateway.has_pending(session_key) is True


@pytest.mark.asyncio
async def test_guest_session_key_matches_the_key_the_clarify_is_registered_under():
    """Pins the §6.2 assumption: the branch derives the key the gateway routes under."""
    adapter = _make_adapter()
    runner = MagicMock()
    runner._session_key_for_source = MagicMock(return_value="agent:main:telegram:42:999")
    adapter._message_handler = MagicMock(__self__=runner)

    event = MagicMock()
    event.source = MagicMock(chat_id="42", user_id="999")

    assert adapter._gateway_session_key(event) == "agent:main:telegram:42:999"
    runner._session_key_for_source.assert_called_once_with(event.source)


@pytest.mark.asyncio
async def test_attachment_during_a_clarify_is_not_treated_as_a_typed_answer(clarify_session):
    """A file is not an answer, and routing it as TEXT would drop it — busy reply instead."""
    adapter, clarify_gateway, session_key = clarify_session
    clarify_gateway.register("c7", session_key, "Which day?", None)
    update, _msg = _make_guest_media_update(kind="photo", caption="@testbot Tuesday", gqid="gq_media")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True), \
         patch.object(adapter, "_cache_and_route_media", new=AsyncMock()) as routed:
        await adapter._handle_guest_message_update(update, MagicMock())

    routed.assert_not_called()
    assert adapter._bot.answer_guest_query.await_args.args[1].id == "busy"


# ---------------------------------------------------------------------------
# Sends to a guest-only chat between turns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_between_turns_to_a_guest_only_chat_is_refused_not_attempted():
    """Advisory traffic after the turn ends used to spend a request on a certain Forbidden."""
    adapter = _make_adapter()
    adapter._known_guest_chats["42"] = 0.0
    adapter._bot.send_message = AsyncMock()

    result = await adapter.send("42", "⏳ Another process is using this session.")

    assert result.success is False
    assert result.error == "guest_chat_no_surface"
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_a_real_member_message_clears_the_guest_only_marking():
    """The bot was added to the group: stop refusing sends there."""
    adapter = _make_adapter()
    adapter._known_guest_chats["42"] = 0.0
    update = MagicMock()
    update.message = MagicMock(chat=MagicMock(id=42))

    adapter._note_member_chat(update)

    assert "42" not in adapter._known_guest_chats


# ---------------------------------------------------------------------------
# Clarify batches on one surface
#
# An ordinary chat gives every question its own message. A guest chat has one,
# and the batch path has two writers racing for it: the loop echoing "you
# answered X", and the agent thread that the same resolve() just unblocked,
# drawing the next question. The echo is two awaits behind, so it lands last and
# replaces a live question — keyboard and all — and the turn waits forever on a
# card nobody can see.
# ---------------------------------------------------------------------------

def _guest_prompt_adapter(imi="imi_abc"):
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._guest_inline_message_ids["42"] = imi
    adapter._guest_chat_types["42"] = "supergroup"
    adapter._remember_guest_inline_message("42", imi)
    return adapter


def _tap(adapter, *, data, imi="imi_abc"):
    query = MagicMock()
    query.message = None
    query.inline_message_id = imi
    query.data = data
    query.from_user = MagicMock(id=999, first_name="Alex", username="alex")
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


@pytest.mark.asyncio
async def test_batch_next_question_survives_the_answer_echo(clarify_session):
    """The reported hang: q1 is drawn, then the echo lands and wipes it. It must not."""
    adapter, clarify_gateway, session_key = clarify_session
    adapter._remember_guest_inline_message("42", "imi_abc")
    clarify_gateway.register("q0", session_key, "Which hand?", ["Left (A-117)", "Right (B-902)"])
    adapter._clarify_state["q0"] = session_key
    query = _tap(adapter, data="cl:q0:0")
    cb = adapter._callback_ctx(query)

    async def _draw_next_question(*_a, **_kw):
        """What the unblocked agent thread does the moment the tap resolves q0."""
        await adapter.send_clarify(
            chat_id="42", question="Pick a word", choices=["Hello"], clarify_id="q1",
            session_key=session_key)

    with patch.object(adapter, "_callback_authorized", new=AsyncMock(return_value=True)), \
         patch("tools.clarify_gateway.resolve_gateway_clarify", side_effect=lambda *a: True), \
         patch.object(adapter, "_record_guest_answer", wraps=adapter._record_guest_answer):
        await _draw_next_question()          # q1 lands on the surface (generation bumps)
        await adapter._handle_clarify_callback(query, "cl:q0:0", cb)

    # The echo captured the older generation, so it is dropped rather than replacing q1.
    query.edit_message_text.assert_not_called()
    last = adapter._bot.edit_message_text.await_args
    assert "Pick a word" in last.kwargs["text"]
    assert last.kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_batch_answer_stays_on_screen_above_the_next_question(clarify_session):
    """One surface: without carrying answers forward, q0's answer vanishes when q1 is drawn."""
    adapter, clarify_gateway, session_key = clarify_session
    adapter._remember_guest_inline_message("42", "imi_abc")
    clarify_gateway.register("q0", session_key, "Which hand?", ["Left (A-117)", "Right (B-902)"])
    adapter._clarify_state["q0"] = session_key
    query = _tap(adapter, data="cl:q0:0")
    cb = adapter._callback_ctx(query)

    with patch.object(adapter, "_callback_authorized", new=AsyncMock(return_value=True)), \
         patch("tools.clarify_gateway.resolve_gateway_clarify", side_effect=lambda *a: True):
        await adapter._handle_clarify_callback(query, "cl:q0:0", cb)
    # Recorded before the resolve, so the draw that the resolve releases already sees it.
    assert adapter._guest_answered_lines["42"] == [("Which hand?", "Left (A-117)")]

    await adapter.send_clarify(
        chat_id="42", question="Pick a word", choices=["Hello"], clarify_id="q1",
        session_key=session_key)

    drawn = adapter._bot.edit_message_text.await_args.kwargs["text"]
    assert "Which hand?" in drawn and "Left (A-117)" in drawn  # the answer is still on screen
    assert "Pick a word" in drawn


@pytest.mark.asyncio
async def test_single_question_echo_is_still_the_final_state(clarify_session):
    """Regression guard for §4: with nothing drawn after it, the echo must land."""
    adapter, clarify_gateway, session_key = clarify_session
    adapter._remember_guest_inline_message("42", "imi_abc")
    clarify_gateway.register("only", session_key, "Which hand?", ["Left", "Right"])
    adapter._clarify_state["only"] = session_key
    query = _tap(adapter, data="cl:only:1")
    cb = adapter._callback_ctx(query)

    with patch.object(adapter, "_callback_authorized", new=AsyncMock(return_value=True)), \
         patch("tools.clarify_gateway.resolve_gateway_clarify", side_effect=lambda *a: True):
        await adapter._handle_clarify_callback(query, "cl:only:1", cb)

    query.edit_message_text.assert_awaited_once()
    assert "Right" in query.edit_message_text.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_three_question_batch_walks_through_every_question(clarify_session):
    """Each question replaces the last, keeping the answers so far above it."""
    adapter, clarify_gateway, session_key = clarify_session
    adapter._remember_guest_inline_message("42", "imi_abc")

    for idx, question in enumerate(("First?", "Second?", "Third?")):
        result = await adapter.send_clarify(
            chat_id="42", question=question, choices=["a", "b"], clarify_id=f"q{idx}",
            session_key=session_key)
        assert result.success is True
        adapter._record_guest_answer("42", question, f"answer {idx}")

    drawn = [c.kwargs["text"] for c in adapter._bot.edit_message_text.await_args_list]
    assert "First?" in drawn[0] and "Second?" in drawn[1] and "Third?" in drawn[2]
    assert "answer 0" in drawn[1]              # q0's answer rides above q1
    assert "answer 0" in drawn[2] and "answer 1" in drawn[2]
    # Every question bumped the surface, so any write decided before them is stale.
    assert adapter._guest_surface_generation("42") == 3


@pytest.mark.asyncio
async def test_unauthorized_tap_during_a_batch_neither_resolves_nor_advances(clarify_session):
    """Fail-closed stays fail-closed: no resolution, no recorded answer, no surface write."""
    adapter, clarify_gateway, session_key = clarify_session
    adapter._remember_guest_inline_message("42", "imi_abc")
    clarify_gateway.register("q0", session_key, "Which hand?", ["Left", "Right"])
    adapter._clarify_state["q0"] = session_key
    query = _tap(adapter, data="cl:q0:0")
    cb = adapter._callback_ctx(query)

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False), \
         patch("tools.clarify_gateway.resolve_gateway_clarify") as resolve:
        await adapter._handle_clarify_callback(query, "cl:q0:0", cb)

    resolve.assert_not_called()
    assert "42" not in adapter._guest_answered_lines
    query.edit_message_text.assert_not_called()
    assert clarify_gateway.has_pending(session_key) is True


@pytest.mark.asyncio
async def test_opc_does_not_overwrite_a_question_drawn_while_it_flushed(clarify_session):
    """The guard is a property of the surface, not of one call order: OPC obeys it too."""
    from gateway.platforms.base import ProcessingOutcome

    adapter, _clarify_gateway, session_key = clarify_session
    adapter._guest_reply_buffer["42"] = "Here is the answer."
    # A question landed on the surface after the flush's generation was captured.
    adapter._guest_surface_generations["42"] = 5
    event = MagicMock()
    event.source.chat_id = "42"

    # The flush reads the generation twice: once to capture what it is writing against, once at
    # the gate. A question drawn in between is exactly the 5 → 6 the side effect models.
    with patch.object(adapter, "_guest_surface_generation", side_effect=[5, 6]):
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_undeliverable_next_question_fails_instead_of_parking_the_turn(clarify_session):
    """§6: q1 that cannot be drawn must report failure, so the batch reports timed_out."""
    adapter, _clarify_gateway, session_key = clarify_session
    adapter._bot.edit_message_text = AsyncMock(side_effect=RuntimeError("Bad Request: message can't be edited"))

    result = await adapter.send_clarify(
        chat_id="42", question="Pick a word", choices=["Hello"], clarify_id="q1",
        session_key=session_key)

    assert result.success is False
    # The gateway retries the card once as plain text; that retry must fail too, or the turn
    # blocks in wait_for_response on a question that was never rendered.
    retry = await adapter.send("42", "❓ Pick a word\n\n  1. Hello")
    assert retry.success is False
    assert retry.error == "guest_prompt_undeliverable"


# ---------------------------------------------------------------------------
# Surface migration on a typed answer
#
# A tap creates no message, so a resolved choice keeps the card it was tapped on.
# Typed text creates one, and it is the newest thing in the chat — answering it
# with a dead "got it" left the real reply in a message above it, which is why
# users ended up replying to the ack.
# ---------------------------------------------------------------------------

def _migrating_adapter(clarify_session, *, new_imi="imi_new"):
    adapter, clarify_gateway, session_key = clarify_session
    adapter._remember_guest_inline_message("42", "imi_abc")
    adapter._bot.answer_guest_query = AsyncMock(return_value=MagicMock(inline_message_id=new_imi))
    return adapter, clarify_gateway, session_key


@pytest.mark.asyncio
async def test_typed_answer_moves_the_surface_to_its_own_message(clarify_session):
    """The reply belongs under the text it answers, not in a card further up."""
    from gateway.platforms.base import ProcessingOutcome

    adapter, clarify_gateway, session_key = _migrating_adapter(clarify_session)
    clarify_gateway.register("c1", session_key, "What should I call it?", None)

    await _deliver_guest_text(adapter, "@testbot call it Atlas", update_id=61, gqid="gq_typed")

    assert adapter._guest_inline_message_ids["42"] == "imi_new"
    # The turn's reply then flushes into that new message, not the one above.
    adapter._guest_reply_buffer["42"] = "Done — it's Atlas now."
    event = MagicMock()
    event.source.chat_id = "42"
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    edits = adapter._bot.edit_message_text.await_args_list
    assert edits and all(c.kwargs["inline_message_id"] == "imi_new" for c in edits)


@pytest.mark.asyncio
async def test_migration_leaves_the_old_message_untouched(clarify_session):
    """After ✏️ Other the old card still reads "Awaiting typed response…" — nothing rewrites it."""
    adapter, clarify_gateway, session_key = _migrating_adapter(clarify_session)
    clarify_gateway.register("c2", session_key, "Which day?", ["Mon", "Tue"])
    clarify_gateway.mark_awaiting_text("c2")

    await _deliver_guest_text(adapter, "@testbot next Friday", update_id=62, gqid="gq_other_typed")

    # The migration's only write is the stub on the NEW message; the old imi gets nothing.
    for call in adapter._bot.edit_message_text.await_args_list:
        assert call.kwargs.get("inline_message_id") != "imi_abc"


@pytest.mark.asyncio
async def test_button_tap_does_not_move_the_surface(clarify_session):
    """Regression guard: a tap creates no message, so there is nothing to migrate to."""
    adapter, clarify_gateway, session_key = _migrating_adapter(clarify_session)
    clarify_gateway.register("c3", session_key, "Which day?", ["Mon", "Tue"])
    adapter._clarify_state["c3"] = session_key
    query = _tap(adapter, data="cl:c3:0")
    cb = adapter._callback_ctx(query)

    with patch.object(adapter, "_callback_authorized", new=AsyncMock(return_value=True)), \
         patch("tools.clarify_gateway.resolve_gateway_clarify", side_effect=lambda *a: True):
        await adapter._handle_clarify_callback(query, "cl:c3:0", cb)

    assert adapter._guest_inline_message_ids["42"] == "imi_abc"
    adapter._bot.answer_guest_query.assert_not_called()
    query.edit_message_text.assert_awaited_once()  # the echo, in place, as before


@pytest.mark.asyncio
async def test_batch_next_question_is_drawn_on_the_migrated_surface(clarify_session):
    """q1 must follow the conversation down, carrying the answers so far."""
    adapter, clarify_gateway, session_key = _migrating_adapter(clarify_session)
    clarify_gateway.register("q0", session_key, "What should I call it?", None)

    await _deliver_guest_text(adapter, "@testbot call it Atlas", update_id=63, gqid="gq_batch_typed")
    await adapter.send_clarify(
        chat_id="42", question="Which colour?", choices=["Blue"], clarify_id="q1",
        session_key=session_key)

    drawn = adapter._bot.edit_message_text.await_args_list[-1]
    assert drawn.kwargs["inline_message_id"] == "imi_new"
    assert "Which colour?" in drawn.kwargs["text"]
    assert "call it Atlas" in drawn.kwargs["text"]  # the recorded answer rides above it


@pytest.mark.asyncio
async def test_migration_without_an_inline_message_id_keeps_the_old_surface(clarify_session):
    """A failed migration must never leave the reply with nowhere to go."""
    from gateway.platforms.base import ProcessingOutcome

    adapter, clarify_gateway, session_key = clarify_session
    adapter._remember_guest_inline_message("42", "imi_abc")
    adapter._bot.answer_guest_query = AsyncMock(return_value=MagicMock(inline_message_id=None))
    clarify_gateway.register("c4", session_key, "What should I call it?", None)

    await _deliver_guest_text(adapter, "@testbot call it Atlas", update_id=64, gqid="gq_no_imi")

    assert adapter._guest_inline_message_ids["42"] == "imi_abc"
    adapter._guest_reply_buffer["42"] = "Done."
    event = MagicMock()
    event.source.chat_id = "42"
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert adapter._bot.edit_message_text.await_args.kwargs["inline_message_id"] == "imi_abc"


@pytest.mark.asyncio
async def test_migration_bumps_the_generation_and_the_later_flush_survives_it(clarify_session):
    """The 3d8b1c69 invariant holds across a migration, in both directions."""
    from gateway.platforms.base import ProcessingOutcome

    adapter, clarify_gateway, session_key = _migrating_adapter(clarify_session)
    clarify_gateway.register("c5", session_key, "What should I call it?", None)
    before = adapter._guest_surface_generation("42")
    stale_query = _tap(adapter, data="cl:c5:0")
    stale_cb = adapter._callback_ctx(stale_query)     # captured against the OLD surface

    await _deliver_guest_text(adapter, "@testbot call it Atlas", update_id=65, gqid="gq_gen")

    assert adapter._guest_surface_generation("42") > before
    # A write decided before the migration is dropped…
    await adapter._edit_html_quiet(stale_query, "stale echo", generation=stale_cb["guest_generation"])
    stale_query.edit_message_text.assert_not_called()
    # …while the flush that comes after it is not.
    adapter._guest_reply_buffer["42"] = "Done — it's Atlas now."
    event = MagicMock()
    event.source.chat_id = "42"
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert adapter._bot.edit_message_text.await_args.kwargs["inline_message_id"] == "imi_new"


@pytest.mark.asyncio
async def test_unauthorized_typed_answer_neither_resolves_nor_migrates(clarify_session):
    """Fail-closed: the gate sits above the continuation branch, so nothing moves."""
    adapter, clarify_gateway, session_key = _migrating_adapter(clarify_session)
    clarify_gateway.register("c6", session_key, "What should I call it?", None)
    update, _msg = _make_guest_update(update_id=66, gqid="gq_stranger", text="@testbot call it Atlas")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False), \
         patch.object(adapter, "_enqueue_text_event") as enqueued:
        await adapter._handle_guest_message_update(update, MagicMock())

    enqueued.assert_not_called()
    adapter._bot.answer_guest_query.assert_not_called()
    assert adapter._guest_inline_message_ids["42"] == "imi_abc"
    assert clarify_gateway.has_pending(session_key) is True


@pytest.mark.asyncio
async def test_migration_preserves_an_undeliverable_prompt_mark(clarify_session):
    """A question that could not be drawn is still the one being waited on."""
    adapter, clarify_gateway, session_key = _migrating_adapter(clarify_session)
    clarify_gateway.register("c7", session_key, "What should I call it?", None)
    adapter._guest_prompt_undeliverable.add("42")

    await _deliver_guest_text(adapter, "@testbot call it Atlas", update_id=67, gqid="gq_undeliv")

    assert "42" in adapter._guest_prompt_undeliverable


# ---------------------------------------------------------------------------
# Media on the message a guest mention replies to
#
# The member text path caches it inside _build_triggered_event; the guest path
# open-codes the event build and skipped that hop, so "@bot what do you hear?"
# sent as a reply to someone's video arrived as a bare quoted line.
# ---------------------------------------------------------------------------

class _CachedStub:
    """What cache_media_bytes_async hands back, reduced to what _attach_cached reads."""

    def __init__(self, kind="video", path="/tmp/cache/clip.mp4", media_type="video/mp4"):
        self.kind, self.path, self.media_type = kind, path, media_type
        self.display_name = path.rsplit("/", 1)[-1]

    def context_note(self):
        return f"[{self.kind}]"


def _make_reply_target(kind=None):
    """A replied-to Message: every media field explicit, since a bare mock is all-truthy."""
    reply = MagicMock()
    reply.text = "Только я одна это слышу..? X))"
    reply.caption = None
    for attr in ("photo", "video", "audio", "voice", "document", "sticker"):
        setattr(reply, attr, None)
    if kind == "photo":
        reply.photo = [MagicMock(file_id="replied_photo", file_size=2048)]
    elif kind:
        setattr(reply, kind, MagicMock(file_id=f"replied_{kind}", file_size=2048, file_name=f"a.{kind}"))
    return reply


async def _deliver_guest_reply(adapter, *, reply_target, text="@testbot что ты там слышишь?", update_id=81):
    update, msg = _make_guest_update(update_id=update_id, gqid="gq_reply", text=text)
    msg.reply_to_message = reply_target
    captured = {}
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True), \
         patch.object(adapter, "_should_process_message", return_value=True), \
         patch.object(adapter, "_apply_telegram_group_observe_attribution", side_effect=lambda e: e), \
         patch.object(adapter, "_enqueue_text_event", side_effect=lambda e: captured.update(event=e)):
        await adapter._handle_guest_message_update(update, MagicMock())
    return captured


@pytest.mark.asyncio
async def test_guest_reply_to_a_video_attaches_the_replied_media():
    """The reported case: the turn must see the file, not just the quoted caption."""
    adapter = _make_adapter()
    cached = _CachedStub()

    with patch.object(adapter, "_download_observed_media", new=AsyncMock(return_value=("ok", cached))):
        captured = await _deliver_guest_reply(adapter, reply_target=_make_reply_target("video"))

    event = captured["event"]
    assert event.media_urls == ["/tmp/cache/clip.mp4"]
    assert event.media_types == ["video/mp4"]
    assert "Replied-to video" in event.text
    # The download takes seconds, so the caller gets the ⏳ before it starts.
    adapter._bot.answer_guest_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_guest_reply_to_plain_text_keeps_its_current_timing():
    """No media to fetch: nothing is downloaded and the stub is not fired early."""
    adapter = _make_adapter()

    with patch.object(adapter, "_download_observed_media", new=AsyncMock()) as download:
        captured = await _deliver_guest_reply(adapter, reply_target=_make_reply_target(None), update_id=82)

    download.assert_not_called()
    assert captured["event"].media_urls == []
    adapter._bot.answer_guest_query.assert_not_called()


@pytest.mark.asyncio
async def test_guest_reply_media_download_failure_still_delivers_the_turn():
    """A failed download degrades to the text turn rather than dropping it."""
    adapter = _make_adapter()

    with patch.object(adapter, "_download_observed_media", new=AsyncMock(return_value=("failed", None))):
        captured = await _deliver_guest_reply(adapter, reply_target=_make_reply_target("voice"), update_id=83)

    assert captured["event"].media_urls == []
    assert captured["event"].text  # the turn still runs, on the text alone


# ---------------------------------------------------------------------------
# Delivering staged audio
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audio_staging_sends_a_non_empty_title(tmp_path, monkeypatch):
    """Cached-audio results render the file's stored title; empty means Audio_title_empty."""
    path = _staged_file(tmp_path, monkeypatch, name="audio_b2c94.mp3", data=b"id3-less bytes")
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    sent = MagicMock()
    sent.audio = MagicMock(file_id="fid_audio")
    adapter._bot.send_audio = AsyncMock(return_value=sent)

    result = await adapter.send_document("42", str(path))

    assert result.success is True
    kwargs = adapter._bot.send_audio.await_args.kwargs
    assert kwargs["title"] == "audio_b2c94.mp3"
    assert kwargs["filename"] == "audio_b2c94.mp3"


@pytest.mark.asyncio
async def test_voice_and_audio_build_their_own_result_types():
    """A voice file_id is not an audio file_id — Telegram rejects the swap."""
    adapter = _make_adapter()

    voice = adapter._guest_cached_media_result(
        {"file_id": "fid_voice", "media_kind": "voice", "file_name": "note.ogg"})
    assert voice.voice_file_id == "fid_voice"
    assert voice.title  # required on this result type

    audio = adapter._guest_cached_media_result(
        {"file_id": "fid_audio", "media_kind": "audio", "file_name": "track.mp3"})
    assert audio.audio_file_id == "fid_audio"


@pytest.mark.asyncio
async def test_a_refused_delivery_keeps_the_token_and_says_so(caplog):
    """The reported dead end: nothing sent, "delivered" logged, and the token burnt."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_audio", "audio", file_name="audio_b2c94.mp3")

    adapter = _make_adapter()
    adapter._bot.answer_guest_query = AsyncMock(side_effect=RuntimeError("Audio_title_empty"))
    update, _msg = _make_guest_update(update_id=91, gqid="gq_fail", text=f"deliver_{token}")

    with caplog.at_level(logging.INFO), \
         patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert "Delivered guest attachment" not in caplog.text
    # The failure is put on screen, in wording distinct from the expired-token case…
    assert adapter._bot.answer_guest_query.await_count == 2
    assert adapter._bot.answer_guest_query.await_args.args[1].id == "delivery_failed"
    # …and the token is still redeemable, so tapping again is a real retry.
    assert token in gmt._TOKEN_STORE

    adapter._bot.answer_guest_query = AsyncMock(return_value=MagicMock(inline_message_id="imi"))
    retry, _msg2 = _make_guest_update(update_id=92, gqid="gq_retry", text=f"deliver_{token}")
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_guest_message_update(retry, MagicMock())
    assert adapter._bot.answer_guest_query.await_args.args[1].audio_file_id == "fid_audio"
    assert token not in gmt._TOKEN_STORE  # spent only now that it actually arrived
    gmt._TOKEN_STORE.clear()


@pytest.mark.asyncio
async def test_inline_redemption_keeps_the_token_when_the_answer_fails():
    """Same consume-on-success rule on the inline path (§B.4)."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_photo", "photo")

    adapter = _make_adapter()
    inline_query = MagicMock()
    inline_query.query = f"deliver_{token}"
    inline_query.from_user = MagicMock(id=123, username="someone")
    inline_query.answer = AsyncMock(side_effect=[RuntimeError("boom"), None])

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_inline_query(MagicMock(inline_query=inline_query), MagicMock())

    assert token in gmt._TOKEN_STORE
    assert inline_query.answer.await_args.args[0][0].id == "delivery_failed"
    gmt._TOKEN_STORE.clear()
