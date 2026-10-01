"""The live iteration counter rides along with every tool-progress send.

Surfaces that render progress as ONE message (Telegram guest mode: a single inline message,
with no second bubble to put the "⏳ Working — N min — iteration X/Y" heartbeat in) need the
counter on the same card as the tool lines. The heartbeat that carries it everywhere else runs
on a 3-minute timer, so ``TurnRunner._stamp_progress_activity`` refreshes it on the progress
metadata instead, where every progress line already passes.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


def _runner(ctx, activity):
    from gateway.run_turn_runner import TurnRunner

    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return None

        @staticmethod
        def _agent_activity_summary(agent):
            return activity

    return TurnRunner(_StubGatewayRunner(), ctx)


def _ctx(metadata):
    return TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42"),
        _progress_metadata=metadata,
        agent_holder=[object()],
    )


@pytest.mark.asyncio
async def test_progress_send_carries_the_live_iteration_counter():
    ctx = _ctx({"thread_id": "g1"})
    runner = _runner(ctx, {"api_call_count": 7, "max_iterations": 150})
    adapter = SimpleNamespace(
        name="tg", send=AsyncMock(return_value=SimpleNamespace(success=True, message_id=None)))

    await runner._send_progress_text(SimpleNamespace(adapter=adapter), "💻 terminal")

    sent = adapter.send.await_args.kwargs["metadata"]
    assert sent["agent_iteration"] == 7 and sent["agent_max_iterations"] == 150
    # The thread routing every other platform depends on is untouched.
    assert sent["thread_id"] == "g1"


@pytest.mark.asyncio
async def test_the_counter_is_refreshed_on_every_progress_send():
    """A card that showed the iteration it had at the first tool call would be a stopped clock."""
    ctx = _ctx({})
    activity = {"api_call_count": 1, "max_iterations": 150}
    runner = _runner(ctx, activity)
    adapter = SimpleNamespace(
        name="tg", send=AsyncMock(return_value=SimpleNamespace(success=True, message_id=None)))

    await runner._send_progress_text(SimpleNamespace(adapter=adapter), "💻 terminal")
    activity["api_call_count"] = 9
    await runner._send_progress_text(SimpleNamespace(adapter=adapter), "🔍 searching")

    assert ctx._progress_metadata["agent_iteration"] == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("activity", [{}, {"api_call_count": None}])
async def test_an_unavailable_summary_leaves_the_metadata_alone(activity):
    """No agent yet (or a provider that reports nothing): the card simply shows no counter."""
    ctx = _ctx({"thread_id": "g1"})
    runner = _runner(ctx, activity)
    adapter = SimpleNamespace(
        name="tg", send=AsyncMock(return_value=SimpleNamespace(success=True, message_id=None)))

    await runner._send_progress_text(SimpleNamespace(adapter=adapter), "💻 terminal")

    assert ctx._progress_metadata == {"thread_id": "g1"}


@pytest.mark.asyncio
async def test_a_chat_with_no_progress_thread_still_gets_the_counter():
    """A guest chat without thread sessions carries no progress metadata at all — and it is
    exactly the chat that needs the counter, since it has nowhere else to show it."""
    ctx = _ctx(None)
    runner = _runner(ctx, {"api_call_count": 3, "max_iterations": 150})
    adapter = SimpleNamespace(
        name="tg", send=AsyncMock(return_value=SimpleNamespace(success=True, message_id=None)))

    await runner._send_progress_text(SimpleNamespace(adapter=adapter), "💻 terminal")

    assert adapter.send.await_args.kwargs["metadata"]["agent_iteration"] == 3


@pytest.mark.asyncio
async def test_nothing_to_report_leaves_the_metadata_absent():
    """No counter, no dict: a platform that gets None today keeps getting None."""
    ctx = _ctx(None)
    runner = _runner(ctx, {})
    adapter = SimpleNamespace(
        name="tg", send=AsyncMock(return_value=SimpleNamespace(success=True, message_id=None)))

    await runner._send_progress_text(SimpleNamespace(adapter=adapter), "💻 terminal")

    assert adapter.send.await_args.kwargs["metadata"] is None
