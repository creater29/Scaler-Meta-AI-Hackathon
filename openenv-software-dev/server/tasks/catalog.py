"""
Three concrete task families with rich starter code.
These tasks are solvable and produce clear reward signals.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from server.tasks.base import Task, TaskMetrics

# ═══════════════════════════════════════════════════════════════
#  TASK 1 – Off-by-one bug (easy)
# ═══════════════════════════════════════════════════════════════
_BUGGY_SORT = '''\
def binary_search(arr, target):
    """Binary search - returns index or -1."""
    lo, hi = 0, len(arr)          # BUG: should be len(arr)-1
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

_TEST_SORT = '''\
from solution import binary_search

def test_found_middle():
    assert binary_search([1,3,5,7,9], 5) == 2

def test_found_first():
    assert binary_search([1,3,5,7,9], 1) == 0

def test_found_last():
    assert binary_search([1,3,5,7,9], 9) == 4

def test_not_found():
    assert binary_search([1,3,5,7,9], 4) == -1

def test_empty():
    assert binary_search([], 1) == -1
'''

_CORRECT_SORT = '''\
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


@dataclass
class OffByOneTask(Task):
    task_id: str = "bug_fix_off_by_one"
    title: str = "Fix Binary Search Off-By-One"
    description: str = (
        "The binary_search function in solution.py has an off-by-one error "
        "in the initial 'hi' bound. Fix it so all 5 tests pass."
    )
    difficulty: str = "easy"
    category: str = "bug_fix"
    starter_files: dict[str, str] = field(default_factory=lambda: {"solution.py": _BUGGY_SORT})
    test_files: dict[str, str] = field(default_factory=lambda: {"test_solution.py": _TEST_SORT})
    metrics: TaskMetrics = field(default_factory=lambda: TaskMetrics(
        required_files=["solution.py"],
        required_functions=["binary_search"],
        expected_test_pass_count=5,
        reference_solution=_CORRECT_SORT,
        forbidden_patterns=["import"],
    ))
    _hints: list[str] = field(default_factory=lambda: [
        "Run the tests to see which cases fail.",
        "Focus on the initial value of 'hi' in the while loop setup.",
        "hi should be len(arr) - 1, not len(arr).",
    ])

    def get_hint(self, step: int) -> str:
        idx = min(step // 5, len(self._hints) - 1)
        return self._hints[idx]

    def acceptance_check(self, snapshot: dict[str, str]) -> dict[str, Any]:
        sol = snapshot.get("solution.py", "")
        accepted = "len(arr) - 1" in sol and "len(arr)" in sol
        return {"accepted": accepted, "checks": [{"name": "off_by_one_fixed", "passed": accepted}]}


# ═══════════════════════════════════════════════════════════════
#  TASK 2 – Feature implementation: LRU Cache (medium)
# ═══════════════════════════════════════════════════════════════
_CACHE_STUB = '''\
class LRUCache:
    """Least-Recently-Used cache with O(1) get/put."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # TODO: implement internal data structures

    def get(self, key: int) -> int:
        """Return value if key exists, else -1."""
        raise NotImplementedError

    def put(self, key: int, value: int) -> None:
        """Insert/update key. Evict LRU entry when at capacity."""
        raise NotImplementedError
'''

_TEST_CACHE = '''\
from solution import LRUCache

def test_basic_get_put():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1

def test_eviction():
    c = LRUCache(2)
    c.put(1, 1); c.put(2, 2); c.put(3, 3)
    assert c.get(1) == -1  # evicted

def test_update_existing():
    c = LRUCache(2)
    c.put(1, 1); c.put(1, 99)
    assert c.get(1) == 99

def test_lru_order():
    c = LRUCache(2)
    c.put(1,1); c.put(2,2); c.get(1); c.put(3,3)
    assert c.get(2) == -1  # 2 was LRU

def test_capacity_one():
    c = LRUCache(1)
    c.put(1,1); c.put(2,2)
    assert c.get(1) == -1
    assert c.get(2) == 2
'''

_CORRECT_CACHE = '''\
from collections import OrderedDict

class LRUCache:
    """Least-Recently-Used cache with O(1) get/put."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        """Return value if key exists, else -1."""
        if key not in self._cache:
            return -1
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: int, value: int) -> None:
        """Insert/update key. Evict LRU entry when at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
'''


@dataclass
class LRUCacheTask(Task):
    task_id: str = "feature_impl_lru_cache"
    title: str = "Implement LRU Cache"
    description: str = (
        "Implement LRUCache in solution.py. "
        "get(key) returns -1 if absent; put(key,value) inserts and evicts LRU on overflow. "
        "Both ops must be O(1). 5 tests must pass."
    )
    difficulty: str = "medium"
    category: str = "feature_impl"
    starter_files: dict[str, str] = field(default_factory=lambda: {"solution.py": _CACHE_STUB})
    test_files: dict[str, str] = field(default_factory=lambda: {"test_solution.py": _TEST_CACHE})
    metrics: TaskMetrics = field(default_factory=lambda: TaskMetrics(
        required_files=["solution.py"],
        required_functions=["get", "put"],
        expected_test_pass_count=5,
        reference_solution=_CORRECT_CACHE,
    ))
    _hints: list[str] = field(default_factory=lambda: [
        "Run tests to see which scenarios fail.",
        "Use a dictionary for O(1) lookup and a doubly-linked list for order.",
        "Python's OrderedDict supports move_to_end() — perfect for LRU.",
    ])

    def get_hint(self, step: int) -> str:
        idx = min(step // 5, len(self._hints) - 1)
        return self._hints[idx]

    def acceptance_check(self, snapshot: dict[str, str]) -> dict[str, Any]:
        sol = snapshot.get("solution.py", "")
        has_get = "def get" in sol
        has_put = "def put" in sol
        not_stub = "raise NotImplementedError" not in sol
        accepted = has_get and has_put and not_stub
        return {
            "accepted": accepted,
            "checks": [
                {"name": "has_get", "passed": has_get},
                {"name": "has_put", "passed": has_put},
                {"name": "not_stub", "passed": not_stub},
            ],
        }


# ═══════════════════════════════════════════════════════════════
#  TASK 3 – Code review (hard)
# ═══════════════════════════════════════════════════════════════
_CODE_TO_REVIEW = '''\
import os, sys, pickle, subprocess

def load_user_data(user_id):
    path = "/tmp/users/" + user_id + ".pkl"   # ISSUE: path traversal
    with open(path, "rb") as f:
        return pickle.load(f)                  # ISSUE: unsafe deserialisation

def run_report(cmd):
    os.system(cmd)                             # ISSUE: shell injection

def get_secret():
    return "hardcoded_secret_123"              # ISSUE: hardcoded secret

class DataProcessor:
    def process(self, items):
        result = []
        for i in range(len(items)):            # ISSUE: non-Pythonic
            result.append(items[i] * 2)
        return result
'''

_TEST_REVIEW = '''\
def test_review_mentions_pickle():
    review = open("review.md").read().lower()
    assert "pickle" in review or "deserializ" in review or "unsafe" in review

def test_review_mentions_injection():
    review = open("review.md").read().lower()
    assert "inject" in review or "os.system" in review or "shell" in review

def test_review_mentions_traversal():
    review = open("review.md").read().lower()
    assert "traversal" in review or "path" in review or "sanitiz" in review

def test_review_mentions_hardcoded():
    review = open("review.md").read().lower()
    assert "hardcoded" in review or "secret" in review or "credential" in review

def test_review_has_suggestions():
    review = open("review.md").read().lower()
    assert "suggest" in review or "recommend" in review or "use" in review or "replace" in review
'''


@dataclass
class CodeReviewTask(Task):
    task_id: str = "code_review_security"
    title: str = "Security Code Review"
    description: str = (
        "Review the code in code_to_review.py and write your findings to review.md. "
        "You must identify: pickle unsafe deserialisation, os.system shell injection, "
        "path traversal, and hardcoded secret. For each issue provide a recommendation."
    )
    difficulty: str = "hard"
    category: str = "code_review"
    starter_files: dict[str, str] = field(default_factory=lambda: {
        "code_to_review.py": _CODE_TO_REVIEW,
        "review.md": "# Code Review\n\n<!-- Write your findings here -->\n",
    })
    test_files: dict[str, str] = field(default_factory=lambda: {"test_review.py": _TEST_REVIEW})
    metrics: TaskMetrics = field(default_factory=lambda: TaskMetrics(
        required_files=["review.md"],
        expected_test_pass_count=5,
        required_patterns=["pickle", "inject", "traversal", "secret"],
    ))
    _hints: list[str] = field(default_factory=lambda: [
        "Read code_to_review.py carefully line by line.",
        "Look for: unsafe deserialisation, shell injection, path traversal, hardcoded credentials.",
        "Write your findings in review.md with a heading per issue and a recommendation.",
    ])

    def get_hint(self, step: int) -> str:
        idx = min(step // 5, len(self._hints) - 1)
        return self._hints[idx]

    def acceptance_check(self, snapshot: dict[str, str]) -> dict[str, Any]:
        review = snapshot.get("review.md", "").lower()
        checks = [
            {"name": "mentions_pickle", "passed": "pickle" in review or "deserializ" in review},
            {"name": "mentions_injection", "passed": "inject" in review or "os.system" in review},
            {"name": "mentions_traversal", "passed": "traversal" in review or "path sanitiz" in review},
            {"name": "mentions_secret", "passed": "hardcoded" in review or "secret" in review},
        ]
        accepted = all(c["passed"] for c in checks)
        return {"accepted": accepted, "checks": checks}


# ═══════════════════════════════════════════════════════════════
#  Registry helper
# ═══════════════════════════════════════════════════════════════
ALL_TASKS: list[Task] = [OffByOneTask(), LRUCacheTask(), CodeReviewTask()]
TASK_MAP: dict[str, Task] = {t.task_id: t for t in ALL_TASKS}
