"""
Virtual filesystem for isolated code manipulation.
All file ops happen against this in-memory store.
"""
from __future__ import annotations
import difflib
import fnmatch
from typing import Optional


class VirtualFilesystem:
    MAX_HISTORY = 50

    def __init__(self) -> None:
        self._files: dict[str, str] = {}
        self._history: list[dict[str, str]] = []

    # ── CRUD ────────────────────────────────────────────────────────────────
    def read(self, path: str) -> Optional[str]:
        return self._files.get(path)

    def write(self, path: str, content: str) -> None:
        self._checkpoint()
        self._files[path] = content

    def patch(self, path: str, old_text: str, new_text: str) -> bool:
        content = self._files.get(path)
        if content is None or old_text not in content:
            return False
        self._checkpoint()
        self._files[path] = content.replace(old_text, new_text, 1)
        return True

    def delete(self, path: str) -> bool:
        if path not in self._files:
            return False
        self._checkpoint()
        del self._files[path]
        return True

    # ── Queries ─────────────────────────────────────────────────────────────
    def list_files(self, pattern: str = "*") -> list[str]:
        return [p for p in self._files if fnmatch.fnmatch(p, pattern)]

    def exists(self, path: str) -> bool:
        return path in self._files

    def snapshot(self) -> dict[str, str]:
        return dict(self._files)

    def restore(self, snapshot: dict[str, str]) -> None:
        self._files = dict(snapshot)

    # ── Diff ────────────────────────────────────────────────────────────────
    def diff(self, path: str, new_content: str) -> str:
        old = self._files.get(path, "").splitlines(keepends=True)
        new = new_content.splitlines(keepends=True)
        return "".join(difflib.unified_diff(old, new, fromfile=path, tofile=path))

    # ── Internal ────────────────────────────────────────────────────────────
    def _checkpoint(self) -> None:
        self._history.append(dict(self._files))
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)
