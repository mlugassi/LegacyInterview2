import ast
import os
import random

from architect.state import ArchitectState

SKIP_DIRS = {"__pycache__", ".git", ".tox", "venv", "env", "node_modules", "migrations",
             "tests", "test", "spec", "specs"}
SKIP_FILES = {"setup.py", "conftest.py"}

_UTILITY_HINTS = ("util", "math", "string", "str", "num", "convert", "parse",
                  "calc", "format", "helper", "algo", "numeric", "text", "encode")

# How many top-scoring files to randomly sample from
_TOP_N_FILES = 3


def _count_module_level_primitive_functions(tree: ast.Module) -> int:
    count = 0
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        if node.end_lineno - node.lineno < 4:
            continue
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
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except SyntaxError:
        return -1

    if len(source.splitlines()) < 20:
        return 0

    score = 0
    basename = os.path.basename(path).lower()
    if any(hint in basename for hint in _UTILITY_HINTS):
        score += 30

    class_count = sum(1 for n in tree.body if isinstance(n, ast.ClassDef))
    score -= class_count * 20

    primitive_funcs = _count_module_level_primitive_functions(tree)
    score += primitive_funcs * 15

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
    scored: list[tuple[int, str]] = []

    for root, dirs, files in os.walk(clone_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if not fname.endswith(".py") or fname in SKIP_FILES:
                continue
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            full = os.path.join(root, fname)
            s = _score_file(full)
            if s > 0:
                scored.append((s, full))

    if not scored:
        raise RuntimeError("No suitable Python file found in repository.")

    # Sort descending, take top N, then pick one at random
    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = scored[:_TOP_N_FILES]
    chosen_score, chosen_path = random.choice(top_n)

    with open(chosen_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    print(f"[mapper] Top candidates: {[os.path.basename(p) for _, p in top_n]}")
    print(f"[mapper] Randomly selected: {chosen_path} (score={chosen_score})")
    state["target_file"] = chosen_path
    state["original_code"] = content
    return state
