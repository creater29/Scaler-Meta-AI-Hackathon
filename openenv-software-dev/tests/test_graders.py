"""Tests for graders, filesystem, and sandbox executor."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from server.sandbox.filesystem import VirtualFilesystem
from server.sandbox.executor import SandboxedExecutor
from server.graders.graders import ProgrammaticGrader, CompositeGrader
from server.tasks.catalog import TASK_MAP


@pytest.fixture
def fs():
    return VirtualFilesystem()

@pytest.fixture
def executor():
    return SandboxedExecutor()

@pytest.fixture
def bug_task():
    return TASK_MAP["bug_fix_off_by_one"]


class TestFilesystem:
    def test_write_read_roundtrip(self, fs):
        fs.write("foo.py", "x = 1\n")
        assert fs.read("foo.py") == "x = 1\n"

    def test_read_missing_returns_none(self, fs):
        assert fs.read("nope.py") is None

    def test_delete(self, fs):
        fs.write("a.py", "x"); fs.delete("a.py")
        assert not fs.exists("a.py")

    def test_patch_replaces_text(self, fs):
        fs.write("a.py", "hello world")
        assert fs.patch("a.py", "world", "python")
        assert fs.read("a.py") == "hello python"

    def test_snapshot_restore(self, fs):
        fs.write("a.py", "v1")
        snap = fs.snapshot()
        fs.write("a.py", "v2")
        fs.restore(snap)
        assert fs.read("a.py") == "v1"

    def test_glob_list(self, fs):
        fs.write("test_a.py", ""); fs.write("solution.py", ""); fs.write("README.md", "")
        assert set(fs.list_files("test_*.py")) == {"test_a.py"}


class TestExecutor:
    def test_passing_tests(self, executor):
        fs = VirtualFilesystem()
        fs.write("solution.py", "def add(a,b): return a+b\n")
        fs.write("test_add.py", "from solution import add\ndef test_add(): assert add(1,2)==3\n")
        r = executor.run_tests(fs)
        assert r["all_passed"]; assert r["passed"] == 1

    def test_failing_tests(self, executor):
        fs = VirtualFilesystem()
        fs.write("solution.py", "def add(a,b): return 0\n")
        fs.write("test_add.py", "from solution import add\ndef test_add(): assert add(1,2)==3\n")
        r = executor.run_tests(fs)
        assert not r["all_passed"]; assert r["failed"] == 1

    def test_linter_catches_long_line(self, executor):
        fs = VirtualFilesystem()
        fs.write("x.py", "x = " + "1" * 130 + "\n")
        r = executor.run_linter(fs)
        assert not r["clean"]

    def test_build_detects_syntax_error(self, executor):
        fs = VirtualFilesystem()
        fs.write("bad.py", "def foo(\n")
        r = executor.build(fs)
        assert not r["success"]


class TestGraders:
    def test_buggy_code_scores_low(self, bug_task, executor):
        fs = VirtualFilesystem()
        for p, c in bug_task.starter_files.items():
            fs.write(p, c)
        g = ProgrammaticGrader()
        r = g.grade(bug_task, fs, executor)
        assert r["score"] < 0.75  # partial credit from file/pattern checks

    def test_fixed_code_scores_high(self, bug_task, executor):
        fixed = bug_task.metrics.reference_solution or ""
        fs = VirtualFilesystem()
        fs.write("solution.py", fixed)
        g = ProgrammaticGrader()
        r = g.grade(bug_task, fs, executor)
        assert r["score"] >= 0.6

    def test_composite_grader_no_llm(self, bug_task, executor):
        fs = VirtualFilesystem()
        for p, c in bug_task.starter_files.items():
            fs.write(p, c)
        g = CompositeGrader(enable_llm=False)
        r = g.grade(bug_task, fs, executor, action_history=[], is_final=True)
        assert 0.0 <= r["score"] <= 1.0
        assert "programmatic" in r
