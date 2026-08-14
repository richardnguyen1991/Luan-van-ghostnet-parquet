from scripts.kaggle_orchestrator import decide_next_session, progress


CONFIG = {
    "recent_push_guard_minutes": 45,
    "running_heartbeat_stale_hours": 6,
    "maximum_session_attempts": 250,
    "maximum_stagnant_restarts": 3,
}


def test_paused_partial_epoch_is_resumed():
    active = {"status": "paused", "completed_epoch": 0, "active_epoch": 1,
              "progress_cursor": 40, "consumed_train_sequences": 12000}
    decision = decide_next_session(active, {}, "complete", CONFIG, 100000)
    assert decision.should_push is True
    assert progress(active) == (0, 1, 12000)


def test_running_kernel_and_completed_run_are_not_restarted():
    assert not decide_next_session({"status": "paused"}, {}, "running", CONFIG, 100000).should_push
    assert not decide_next_session({"status": "completed", "completed_epoch": 100}, {}, "complete", CONFIG, 100000).should_push
