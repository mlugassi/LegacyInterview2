import ast
import os
from typing import Optional

from architect.state import ArchitectState

# Files/dirs to skip
SKIP_DIRS = {"__pycache__", ".git", ".tox", "venv", "env", "node_modules", "migrations"}
SKIP_FILES = {"setup.py", "conftest.py"}


def _score_file(path: str) -> int:
    """Return a logic-density score for a Python file. Higher = better target."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except SyntaxError:
        return -1

    score = 0
    has_external_imports = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ("os", "sys", "math", "re", "collections", "itertools",
                               "functools", "typing", "string", "random", "copy"):
                    has_external_imports = True
        if isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in ("os", "sys", "math", "re", "collections", "itertools",
                           "functools", "typing", "string", "random", "copy", ""):
                has_external_imports = True

        # Reward logic-heavy constructs
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            score += 5
        if isinstance(node, ast.Return):
            score += 2
        if isinstance(node, (ast.For, ast.While)):
            score += 3
        if isinstance(node, ast.If):
            score += 2
        if isinstance(node, ast.BinOp):
            score += 1
        if isinstance(node, ast.Compare):
            score += 1

    # Pure utility files are preferred (no heavy external deps)
    if has_external_imports:
        score = score // 2

    # Penalise very short files
    if len(source.splitlines()) < 20:
        score = 0

    return score


def map_files(state: ArchitectState) -> ArchitectState:
    clone_path = state["clone_path"]
    best_path: Optional[str] = None
    best_score = -1

    for root, dirs, files in os.walk(clone_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if not fname.endswith(".py") or fname in SKIP_FILES:
                continue
            full = os.path.join(root, fname)
            score = _score_file(full)
            if score > best_score:
                best_score = score
                best_path = full

    if best_path is None:
        raise RuntimeError("No suitable Python file found in repository.")

    with open(best_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    print(f"[mapper] Selected target file: {best_path} (score={best_score})")
    state["target_file"] = best_path
    state["original_code"] = content
    return state
