"""
Tests for the core environment: reset, step, state, action dispatch.
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from server.environments.software_dev_env import SoftwareDevEnvironment
from server.models import SoftwareDevAction, ActionType


@pytest.fixture
def env():
    e = SoftwareDevEnvironment()
    e.reset(task_id="bug_fix_off_by_one")
    return e


class TestReset:
    def test_returns_observation(self):
        env = SoftwareDevEnvironment()
        obs = env.reset(task_id="bug_fix_off_by_one")
        assert obs.task_id == "bug_fix_off_by_one"
        assert obs.step == 0
        assert "solution.py" in obs.files

    def test_all_task_ids_valid(self):
        from server.tasks.catalog import TASK_MAP
        for tid in TASK_MAP:
            env = SoftwareDevEnvironment()
            obs = env.reset(task_id=tid)
            assert obs.task_id == tid

    def test_state_resets(self):
        env = SoftwareDevEnvironment()
        env.reset(task_id="bug_fix_off_by_one")
        env.step(SoftwareDevAction(action_type=ActionType.NO_OP))
        obs2 = env.reset(task_id="bug_fix_off_by_one")
        assert obs2.step == 0

    def test_invalid_task_raises(self):
        env = SoftwareDevEnvironment()
        with pytest.raises(KeyError):
            env.reset(task_id="nonexistent_task_xyz")


class TestStep:
    def test_no_op_increments_step(self, env):
        result = env.step(SoftwareDevAction(action_type=ActionType.NO_OP))
        assert result.observation.step == 1

    def test_read_file_returns_content(self, env):
        result = env.step(SoftwareDevAction(
            action_type=ActionType.READ_FILE, target_file="solution.py"))
        assert result.observation.last_action_result.get("status") == "ok"
        assert "binary_search" in result.observation.last_action_result.get("content", "")

    def test_write_file_updates_fs(self, env):
        env.step(SoftwareDevAction(
            action_type=ActionType.WRITE_FILE,
            target_file="solution.py",
            text_input="def foo():\n    '''foo'''\n    pass\n"))
        result = env.step(SoftwareDevAction(
            action_type=ActionType.READ_FILE, target_file="solution.py"))
        assert "foo" in result.observation.last_action_result.get("content", "")

    def test_run_tests_before_fix_fails(self, env):
        result = env.step(SoftwareDevAction(action_type=ActionType.RUN_TESTS))
        info = result.info.get("action_result", {})
        assert info.get("all_passed") is False or info.get("failed", 0) > 0

    def test_submit_terminates_episode(self, env):
        result = env.step(SoftwareDevAction(action_type=ActionType.SUBMIT))
        assert result.done is True

    def test_step_after_done_raises(self, env):
        env.step(SoftwareDevAction(action_type=ActionType.SUBMIT))
        with pytest.raises(RuntimeError):
            env.step(SoftwareDevAction(action_type=ActionType.NO_OP))

    def test_reward_is_float_and_bounded(self, env):
        # step_penalty=-0.02 always fires; grading bonus may outweigh it early on
        result = env.step(SoftwareDevAction(action_type=ActionType.NO_OP))
        assert isinstance(result.reward, float)
        assert -5.0 < result.reward < 10.0  # sane bounds

    def test_state_tracks_steps(self, env):
        env.step(SoftwareDevAction(action_type=ActionType.NO_OP))
        env.step(SoftwareDevAction(action_type=ActionType.NO_OP))
        s = env.state()
        assert s.step_count == 2
