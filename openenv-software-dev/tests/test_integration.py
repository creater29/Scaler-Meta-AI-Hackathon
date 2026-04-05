"""
Integration tests: full episode solve paths for each task.
These prove the environment is solvable — addressing reviewer feedback.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from server.environments.software_dev_env import SoftwareDevEnvironment
from server.models import SoftwareDevAction, ActionType

CORRECT_BINARY_SEARCH = '''\
def binary_search(arr, target):
    """Binary search - returns index or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''

CORRECT_LRU = '''\
from collections import OrderedDict

class LRUCache:
    """LRU cache with O(1) get/put."""
    def __init__(self, capacity):
        """Init."""
        self.capacity = capacity
        self._cache = OrderedDict()
    def get(self, key):
        """Get value."""
        if key not in self._cache: return -1
        self._cache.move_to_end(key)
        return self._cache[key]
    def put(self, key, value):
        """Put value."""
        if key in self._cache: self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity: self._cache.popitem(last=False)
'''

CORRECT_REVIEW = '''\
# Security Code Review

## Issue 1: Unsafe Pickle Deserialisation
pickle.load() is unsafe and can execute malicious code during deserialisation.
Suggest: replace with json.load() or a validated schema parser.

## Issue 2: Shell Injection via os.system
os.system(cmd) allows shell injection if cmd is user-controlled.
Suggest: use subprocess.run() with a fixed argument list.

## Issue 3: Path Traversal
Concatenating user_id into path allows traversal attacks.
Suggest: sanitise with os.path.basename() and validate against allowlist.

## Issue 4: Hardcoded Secret / Credential
The hardcoded secret should be loaded from an environment variable.
Suggest: os.getenv("SECRET_KEY") with a vault-backed default.
'''


class TestBugFixEpisode:
    def test_solve_off_by_one(self):
        env = SoftwareDevEnvironment()
        env.reset(task_id="bug_fix_off_by_one")
        # Read → write fix → run tests → submit
        env.step(SoftwareDevAction(action_type=ActionType.READ_FILE, target_file="solution.py"))
        env.step(SoftwareDevAction(action_type=ActionType.WRITE_FILE,
                                   target_file="solution.py", text_input=CORRECT_BINARY_SEARCH))
        tr = env.step(SoftwareDevAction(action_type=ActionType.RUN_TESTS))
        assert tr.info["action_result"].get("all_passed"), \
            f"Tests should pass after fix. Got: {tr.info['action_result']}"
        result = env.step(SoftwareDevAction(action_type=ActionType.SUBMIT))
        assert result.done
        assert result.info["grading"]["score"] >= 0.7

    def test_unfixed_scores_low(self):
        env = SoftwareDevEnvironment()
        env.reset(task_id="bug_fix_off_by_one")
        result = env.step(SoftwareDevAction(action_type=ActionType.SUBMIT))
        assert result.info["grading"]["score"] < 0.75


class TestFeatureImplEpisode:
    def test_implement_lru_cache(self):
        env = SoftwareDevEnvironment()
        env.reset(task_id="feature_impl_lru_cache")
        env.step(SoftwareDevAction(action_type=ActionType.WRITE_FILE,
                                   target_file="solution.py", text_input=CORRECT_LRU))
        tr = env.step(SoftwareDevAction(action_type=ActionType.RUN_TESTS))
        assert tr.info["action_result"].get("all_passed"), \
            f"LRU tests should pass. Got: {tr.info['action_result']}"
        result = env.step(SoftwareDevAction(action_type=ActionType.SUBMIT))
        assert result.done
        assert result.info["grading"]["score"] >= 0.7


class TestCodeReviewEpisode:
    def test_solve_security_review(self):
        env = SoftwareDevEnvironment()
        env.reset(task_id="code_review_security")
        env.step(SoftwareDevAction(action_type=ActionType.WRITE_FILE,
                                   target_file="review.md", text_input=CORRECT_REVIEW))
        tr = env.step(SoftwareDevAction(action_type=ActionType.RUN_TESTS))
        assert tr.info["action_result"].get("all_passed"), \
            f"Review tests should pass. Got: {tr.info['action_result']}"
        result = env.step(SoftwareDevAction(action_type=ActionType.SUBMIT))
        assert result.done
        assert result.info["grading"]["score"] >= 0.7


class TestMultiEpisodeConsistency:
    def test_multiple_resets_independent(self):
        env = SoftwareDevEnvironment()
        for _ in range(3):
            obs = env.reset(task_id="bug_fix_off_by_one")
            assert obs.step == 0
            assert "len(arr)" in obs.files.get("solution.py", "")

    def test_cumulative_reward_resets(self):
        env = SoftwareDevEnvironment()
        env.reset(task_id="bug_fix_off_by_one")
        for _ in range(3):
            env.step(SoftwareDevAction(action_type=ActionType.NO_OP))
        env.reset(task_id="bug_fix_off_by_one")
        assert env.state().cumulative_reward == 0.0
