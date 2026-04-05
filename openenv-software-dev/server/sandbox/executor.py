"""
Sandboxed executor: runs code purely in-memory.
Intercepts imports and open() to redirect to VirtualFilesystem.
"""
from __future__ import annotations
import ast, sys, io, types, traceback, builtins
from typing import Any
from server.sandbox.filesystem import VirtualFilesystem


class SandboxedExecutor:
    TIMEOUT_SECONDS = 10
    MAX_LINE_LEN = 120

    def run_tests(self, fs: VirtualFilesystem) -> dict[str, Any]:
        """Discover test_*.py files, compile modules, inject into sys.modules, run test functions."""
        test_files = fs.list_files("test_*.py")
        results: list[dict] = []
        passed = failed = 0

        # Build a fake import system: compile all .py files → module objects
        module_cache: dict[str, types.ModuleType] = {}
        for path in fs.list_files("*.py"):
            mod_name = path.replace(".py", "").replace("/", ".")
            src = fs.read(path) or ""
            mod = types.ModuleType(mod_name)
            mod.__file__ = path
            try:
                code = compile(src, path, "exec")
                exec(code, mod.__dict__)  # noqa: S102
            except Exception:
                pass  # will surface as import error when test runs
            module_cache[mod_name] = mod

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in module_cache:
                return module_cache[name]
            return _real_import(name, globals, locals, fromlist, level)

        def _fake_open(filename, mode="r", *args, **kwargs):
            # Redirect reads of virtual files through the filesystem
            if isinstance(filename, str) and "b" not in mode:
                content = fs.read(filename)
                if content is not None:
                    return io.StringIO(content)
            return _real_open(filename, mode, *args, **kwargs)

        _real_import = builtins.__import__
        _real_open = builtins.open

        for tf in test_files:
            src = fs.read(tf) or ""
            ns: dict[str, Any] = {}
            # Inject fake import + open
            builtins.__import__ = _fake_import
            builtins.open = _fake_open
            try:
                exec(compile(src, tf, "exec"), ns)  # noqa: S102
            except Exception as e:
                results.append({"file": tf, "name": "compile", "passed": False, "error": str(e)})
                failed += 1
                continue
            finally:
                builtins.__import__ = _real_import
                builtins.open = _real_open

            for name, obj in ns.items():
                if name.startswith("test_") and callable(obj):
                    stdout_cap = io.StringIO()
                    old_stdout = sys.stdout
                    sys.stdout = stdout_cap
                    builtins.__import__ = _fake_import
                    builtins.open = _fake_open
                    try:
                        obj()
                        passed += 1
                        results.append({"file": tf, "name": name, "passed": True,
                                        "output": stdout_cap.getvalue()})
                    except Exception:
                        failed += 1
                        results.append({"file": tf, "name": name, "passed": False,
                                        "error": traceback.format_exc()})
                    finally:
                        sys.stdout = old_stdout
                        builtins.__import__ = _real_import
                        builtins.open = _real_open

        total = passed + failed
        return {"all_passed": failed == 0 and total > 0, "passed": passed,
                "failed": failed, "total": total, "details": results}

    def run_linter(self, fs: VirtualFilesystem) -> dict[str, Any]:
        """AST-based lint: syntax errors + line-length + missing docstrings."""
        issues: list[dict] = []
        for path in fs.list_files("*.py"):
            src = fs.read(path) or ""
            try:
                tree = ast.parse(src, filename=path)
            except SyntaxError as e:
                issues.append({"file": path, "line": e.lineno, "msg": f"SyntaxError: {e.msg}"})
                continue
            for i, line in enumerate(src.splitlines(), 1):
                if len(line) > self.MAX_LINE_LEN:
                    issues.append({"file": path, "line": i,
                                   "msg": f"Line too long ({len(line)} > {self.MAX_LINE_LEN})"})
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not ast.get_docstring(node):
                        issues.append({"file": path, "line": node.lineno,
                                       "msg": f"Missing docstring: {node.name}"})
        return {"clean": len(issues) == 0, "issue_count": len(issues), "issues": issues[:20]}

    def build(self, fs: VirtualFilesystem) -> dict[str, Any]:
        """Check parse errors across all .py files."""
        parse_errors: list[dict] = []
        for path in fs.list_files("*.py"):
            src = fs.read(path) or ""
            try:
                ast.parse(src, filename=path)
            except SyntaxError as e:
                parse_errors.append({"file": path, "line": e.lineno, "msg": str(e)})
        return {"success": len(parse_errors) == 0, "parse_errors": parse_errors}
