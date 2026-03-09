import ast
import os
import random
from typing import Dict, List, Set, Tuple

from architect.state import ArchitectState

SKIP_DIRS = {"__pycache__", ".git", ".tox", "venv", "env", "node_modules", "migrations",
             "tests", "test", "spec", "specs"}
SKIP_FILES = {"setup.py", "conftest.py"}

_UTILITY_HINTS = ("util", "math", "string", "str", "num", "convert", "parse",
                  "calc", "format", "helper", "algo", "numeric", "text", "encode")

# How many top-scoring files to randomly sample from
_TOP_N_FILES = 3


def _analyze_call_graph(tree: ast.Module) -> Dict[str, Set[str]]:
    """
    Build a call graph: for each function, which other functions does it call?
    Returns: {func_name: set_of_called_functions}
    """
    call_graph: Dict[str, Set[str]] = {}
    
    # First pass: collect all function names
    all_functions: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):  # Skip private functions
                all_functions.add(node.name)
    
    # Second pass: for each function, find what it calls
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        
        called_funcs: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Direct function call
                if isinstance(child.func, ast.Name) and child.func.id in all_functions:
                    called_funcs.add(child.func.id)
                # Attribute call (obj.method) - check if method name is a known function
                elif isinstance(child.func, ast.Attribute) and child.func.attr in all_functions:
                    called_funcs.add(child.func.attr)
        
        call_graph[node.name] = called_funcs
    
    return call_graph


def _compute_max_depth(func: str, call_graph: Dict[str, Set[str]], 
                       visited: Set[str] = None) -> int:
    """
    Compute the maximum call-chain depth from this function.
    Depth 1 = leaf function (calls no other functions in this file)
    Depth 2 = calls a leaf function
    Etc.
    """
    if visited is None:
        visited = set()
    
    if func in visited:  # Circular dependency
        return 1
    
    if func not in call_graph:
        return 1
    
    called = call_graph[func]
    if not called:  # Leaf function
        return 1
    
    visited.add(func)
    max_child_depth = max(
        (_compute_max_depth(child, call_graph, visited.copy()) for child in called),
        default=0
    )
    return max_child_depth + 1


def _get_functions_by_depth(source: str) -> List[Tuple[str, int, List[str]]]:
    """
    Analyze all functions and return: [(func_name, depth, call_chain), ...]
    Sorted by depth descending.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    call_graph = _analyze_call_graph(tree)
    
    results: List[Tuple[str, int, List[str]]] = []
    for func_name in call_graph:
        depth = _compute_max_depth(func_name, call_graph)
        # Build a sample call chain (simplified - just one path)
        chain = [func_name]
        current = func_name
        for _ in range(depth - 1):
            called = call_graph.get(current, set())
            if not called:
                break
            # Pick one called function for the chain
            next_func = next(iter(called))
            chain.append(next_func)
            current = next_func
        
        results.append((func_name, depth, chain))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results


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


def _score_file(path: str, target_nesting: int) -> Tuple[int, int]:
    """
    Score a file for sabotage suitability.
    Returns: (score, max_nesting_depth)
    """
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except SyntaxError:
        return -1, 0

    if len(source.splitlines()) < 20:
        return -1000, 0  # too short

    # Pure data/config files with no functions can't be sabotaged
    func_count = sum(1 for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    if func_count == 0:
        return -1000, 0

    # Analyze nesting depth
    functions_by_depth = _get_functions_by_depth(source)
    max_depth = max((depth for _, depth, _ in functions_by_depth), default=0) if functions_by_depth else 0
    
    score = 0
    basename = os.path.basename(path).lower()
    if any(hint in basename for hint in _UTILITY_HINTS):
        score += 30

    class_count = sum(1 for n in tree.body if isinstance(n, ast.ClassDef))
    score -= class_count * 20

    primitive_funcs = _count_module_level_primitive_functions(tree)
    score += primitive_funcs * 15
    
    # Bonus for files with good nesting depth
    if max_depth >= target_nesting:
        score += 50  # Perfect match or better
    elif max_depth >= target_nesting - 1:
        score += 30  # Close enough
    elif max_depth >= 2:
        score += 10  # Has some nesting

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ("os", "sys", "math", "re", "collections", "itertools",
                               "functools", "typing", "string", "random", "copy"):
                    score -= 10
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative import (from .something import X) — file depends on package
                # structure; functions can't be exec-tested in isolation.
                score -= 25
            else:
                top = (node.module or "").split(".")[0]
                if top not in ("os", "sys", "math", "re", "collections", "itertools",
                               "functools", "typing", "string", "random", "copy", ""):
                    score -= 5

    return score, max_depth


def map_files(state: ArchitectState) -> ArchitectState:
    clone_path = state["clone_path"]
    target_nesting = state["nesting_level"]
    scored: list[tuple[int, int, str]] = []  # (score, max_depth, path)
    fallback: list[tuple[int, int, str]] = []

    for root, dirs, files in os.walk(clone_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if not fname.endswith(".py") or fname in SKIP_FILES:
                continue
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            full = os.path.join(root, fname)
            s, max_depth = _score_file(full, target_nesting)
            if s > 0:
                scored.append((s, max_depth, full))
            elif s > -999:  # parseable but penalised — keep as fallback
                fallback.append((s, max_depth, full))

    # If no file scored positively, use the best-scoring fallback candidates
    if not scored:
        if not fallback:
            raise RuntimeError("No suitable Python file found in repository.")
        print("[mapper] No files scored > 0; using best-available fallback candidates.")
        scored = fallback

    # Sort descending by score, take top N, then pick one with preference for matching depth
    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = scored[:_TOP_N_FILES]
    
    # Weighted selection: prefer files with max_depth >= target_nesting
    weights = []
    for score, max_depth, path in top_n:
        if max_depth >= target_nesting:
            weight = score * 3  # Strong preference for sufficient depth
        elif max_depth >= target_nesting - 1:
            weight = score * 1.5  # Moderate preference for close depth
        else:
            weight = score * 0.5  # Lower weight for shallow files
        weights.append(weight)
    
    # Weighted random choice
    total_weight = sum(weights)
    if total_weight > 0:
        r = random.uniform(0, total_weight)
        cumulative = 0
        chosen_idx = 0
        for idx, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                chosen_idx = idx
                break
        chosen_score, chosen_depth, chosen_path = top_n[chosen_idx]
    else:
        chosen_score, chosen_depth, chosen_path = top_n[0]

    # Build ordered fallback list: chosen first, then remaining candidates by score
    all_sorted_paths = [p for _, _, p in scored]
    remaining = [p for p in all_sorted_paths if p != chosen_path]
    candidate_files = [chosen_path] + remaining

    with open(chosen_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    debug_mode = state.get("debug_mode", False)
    if debug_mode:
        print(f"[mapper] Target nesting level: {target_nesting}")
        print(f"[mapper] Top candidates: {[(os.path.basename(p), d) for _, d, p in top_n]}")
        print(f"[mapper] Randomly selected: {chosen_path} (score={chosen_score}, max_depth={chosen_depth})")
        print(f"[mapper] {len(candidate_files)} total candidates available as fallback")
    
    state["target_file"] = chosen_path
    state["original_code"] = content
    state["candidate_files"] = candidate_files
    return state
