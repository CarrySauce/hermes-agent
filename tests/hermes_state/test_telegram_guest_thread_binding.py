"""Durable guest thread bindings: which guest conversation a bot message belongs to.

The session behind a guest thread already survives a restart in ``state.db``; only the
binding from a message to its thread was volatile, which is what made a reply to an older
message start a new conversation after a gateway restart.
"""

import time

from hermes_state import SessionDB


def _db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_guest_thread_migration()
    return db


def test_a_surface_resolves_to_its_thread(tmp_path):
    db = _db(tmp_path)
    db.record_telegram_guest_thread(
        chat_id="-100", surface_key="imi_a", thread_token="g1u7", text_key="About A")

    assert db.lookup_telegram_guest_thread(chat_id="-100", surface_key="imi_a") == "g1u7"
    assert db.lookup_telegram_guest_thread(chat_id="-100", text_key="About A") == "g1u7"
    assert db.lookup_telegram_guest_thread(chat_id="-100", surface_key="imi_unknown") is None


def test_the_surface_wins_over_the_text(tmp_path):
    """An exact message is a stronger claim than text two messages could share."""
    db = _db(tmp_path)
    db.record_telegram_guest_thread(
        chat_id="-100", surface_key="imi_a", thread_token="g1u7", text_key="same text")
    db.record_telegram_guest_thread(
        chat_id="-100", surface_key="imi_b", thread_token="g2u7", text_key="same text")

    assert db.lookup_telegram_guest_thread(
        chat_id="-100", surface_key="imi_a", text_key="same text") == "g1u7"


def test_a_memo_write_does_not_erase_the_text_it_was_found_by(tmp_path):
    db = _db(tmp_path)
    db.record_telegram_guest_thread(
        chat_id="-100", surface_key="imi_a", thread_token="g1u7", text_key="About A")
    db.record_telegram_guest_thread(chat_id="-100", surface_key="imi_a", thread_token="g1u7")

    assert db.lookup_telegram_guest_thread(chat_id="-100", text_key="About A") == "g1u7"


def test_latest_is_per_chat_and_per_profile(tmp_path):
    db = _db(tmp_path)
    db.record_telegram_guest_thread(chat_id="-100", surface_key="imi_a", thread_token="g1u7")
    db.record_telegram_guest_thread(chat_id="-100", surface_key="imi_b", thread_token="g2u7")
    db.record_telegram_guest_thread(chat_id="-200", surface_key="imi_c", thread_token="g3u7")
    db.record_telegram_guest_thread(
        chat_id="-100", surface_key="imi_d", thread_token="g4u7", profile_name="work")

    assert db.latest_telegram_guest_thread(chat_id="-100") == "g2u7"
    assert db.latest_telegram_guest_thread(chat_id="-200") == "g3u7"
    assert db.latest_telegram_guest_thread(chat_id="-100", profile_name="work") == "g4u7"
    assert db.latest_telegram_guest_thread(chat_id="-999") is None
    # A profile's rows are invisible to another's lookup, as for topic bindings.
    assert db.lookup_telegram_guest_thread(chat_id="-100", surface_key="imi_d") is None


def test_old_rows_age_out_and_the_chat_is_capped(tmp_path):
    db = _db(tmp_path)
    db.record_telegram_guest_thread(chat_id="-100", surface_key="ancient", thread_token="g0u7")
    # Backdate it past the retention window, then write again to trigger the prune.
    db._execute_write(lambda conn: conn.execute(
        f"UPDATE {db._GUEST_THREAD_TABLE} SET updated_at = ? WHERE surface_key = 'ancient'",
        (time.time() - db._GUEST_THREAD_RETAIN_SECONDS - 60,)))
    db.record_telegram_guest_thread(chat_id="-100", surface_key="fresh", thread_token="g1u7")

    assert db.lookup_telegram_guest_thread(chat_id="-100", surface_key="ancient") is None
    assert db.lookup_telegram_guest_thread(chat_id="-100", surface_key="fresh") == "g1u7"


def test_lookups_are_quiet_before_the_migration_runs(tmp_path):
    """A gateway that never enables the feature never grows the table; reads must not raise."""
    db = SessionDB(db_path=tmp_path / "state.db")

    assert db.lookup_telegram_guest_thread(chat_id="-100", surface_key="imi_a") is None
    assert db.latest_telegram_guest_thread(chat_id="-100") is None
