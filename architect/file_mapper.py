import ast
import os
from typing import Optional

from architect.state import ArchitectState

SKIP_DIRS = {"__pycache__", ".git", ".tox", "venv", "env", "node_modules", "migrations"}
SKIP_FILES = {"setup.py", "conftest.py"}

# File names that strongly suggest pure utility/math/string logic
_UTILITY_HINTS = ("util", "math", "string", "str", "num", "convert", "parse",
                  "calc", "format", "helper", "algo", "numeric", "text", "encode")


def _count_module_level_primitive_functions(tree: ast.Module) -> int:
    """Count module-level functions whose body uses numeric/string/list ops (not class instances)."""
    count = 0
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        if node.end_lineno - node.lineno < 4:
            continue
        # Check body for numeric literals, string ops, or list/tuple ops
        has_number = any(isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                         for n in ast.walk(node))
        has_binop = any(isinstance(n, ast.BinOp) for n in ast.walk(node))
        has_loop = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node))
        has_return = any(isinstance(n, ast.Return) and n.value is not None
                         for n in ast.walk(node))
        if has_return and (has_number or has_binop or has_loop):
            count += 1
    return count


def _score_file(path: str) -> int:
    """Return a score favouring files with pure utility/math module-level functions."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except SyntaxError:
        return -1

    lines = source.splitlines()
    if len(lines) < 20:
        return 0

    score = 0

    # Bonus for utility-sounding file names
    basename = os.path.basename(path).lower()
    if any(hint in basename for hint in _UTILITY_HINTS):
        score += 30

    # Count module-level class definitions — class-heavy files are bad targets
    class_count = sum(1 for n in tree.body if isinstance(n, ast.ClassDef))
    score -= class_count * 20

    # Reward module-level functions with primitive-looking logic
    primitive_funcs = _count_module_level_primitive_functions(tree)
    score += primitive_funcs * 15

    # Penalise files with heavy external imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ("os", "sys", "math", "re", "collections", "itertools",
                               "functools", "typing", "string", "random", "copy"):
                    score -= 10
        if isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in ("os", "sys", "math", "re", "collections", "itertools",
                           "functools", "typing", "string", "random", "copy", ""):
                score -= 5

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
