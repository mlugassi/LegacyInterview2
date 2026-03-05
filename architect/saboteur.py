import ast
import difflib
import json
import random
import textwrap

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from architect.state import ArchitectState

# How many helpers to inject bugs into per level
_BUGS_PER_LEVEL = {1: 1, 2: 2, 3: 2}
# Maximum call-chain depth to descend when picking bug targets
# Level 1 → depth 2 (surface → helper)
# Level 2 → depth 3 (surface → mid → helper)
# Level 3 → depth 4 (surface → mid → mid2 → helper)
_MAX_DEPTH_LEVEL = {1: 4, 2: 4, 3: 4}

_LEVEL_INSTRUCTIONS = {
    1: """
You are performing a Level 1 sabotage on a SINGLE Python function.

STEP 1 — Understand the function contract:
  Read the function carefully. Identify exactly what it is SUPPOSED to do:
  its inputs, its expected outputs, and its core logic.

STEP 2 — Invent and inject ONE FUNCTIONAL bug of your own choosing:
  You decide the bug type — be creative and pick something that fits THIS specific function.
  Requirements:
    * The code must still run without exceptions on valid inputs
    * The function must return a WRONG RESULT for at least some inputs
    * The bug must be non-obvious — a student needs to read the logic carefully to spot it
  Make 1–3 small coordinated changes to produce the behavioral failure.
  Do NOT rename any variables, do NOT add comments.
  Return the function with ONLY the bug injected — keep everything else identical.
""",
    2: """
You are performing a Level 2 sabotage on a SINGLE Python function.
This builds on Level 1: inject a bug AND obfuscate variable names.

STEP 1 — Understand the function contract:
  Read the function carefully. Identify exactly what it is SUPPOSED to do.

STEP 2 — Invent and inject ONE FUNCTIONAL bug of your own choosing:
  You decide the bug type — be creative and pick something that fits THIS specific function.
  Requirements:
    * The code must still run without exceptions on valid inputs
    * The function must return a WRONG RESULT for at least some inputs
    * The bug must be non-obvious — require tracing logic to find
  Make 1–3 coordinated changes.

STEP 3 — Rename every internal local variable to a meaningless name (var1, temp_x, ptr_b, etc.).
  Do NOT rename the function itself, its parameters, or any imported names.
  Do NOT add any comments.
""",
    3: """
You are performing a Level 3 sabotage on a SINGLE Python function.
This builds on Level 2: inject a bug, obfuscate variable names, AND add misleading comments.

STEP 1 — Understand the function contract:
  Read the function carefully. Identify exactly what it is SUPPOSED to do.

STEP 2 — Invent and inject ONE FUNCTIONAL bug of your own choosing:
  You decide the bug type — be creative and pick something that fits THIS specific function.
  Requirements:
    * The code must still run without exceptions on valid inputs
    * The function must return a WRONG RESULT for at least some inputs
    * The bug must be non-obvious — require understanding the algorithm to diagnose
  Make 1–3 coordinated changes.

STEP 3 — Rename every internal local variable to a meaningless name (coeff_a, magic_val, var1, etc.).
  Do NOT rename the function itself, its parameters, or any imported names.

STEP 4 — Add 1–2 misleading inline comments near the bug site.
  Each comment must describe what the code APPEARS to be doing correctly — as if it is right.
  NEVER use words like "incorrect", "wrong", "bug", "error", "broken", "corrupted",
  "intentional", "mistake", or any synonym that hints at a problem.
  Example of BAD comment: `# Incorrectly appending closing brace`  ← reveals the bug
  Example of GOOD comment: `# Append the closing brace to complete the field`  ← looks correct
""",
}

_SYSTEM_PROMPT = """
You are the Legacy Challenge Architect. You will receive either ONE or TWO Python functions.

SCENARIO A — labeled "FUNCTION TO SABOTAGE":
  Sabotage that function directly. test_args are for calling that function.

SCENARIO B — labeled "SURFACE FUNCTION" and "HELPER FUNCTION TO SABOTAGE":
  Sabotage ONLY the HELPER FUNCTION (inject the bug there).
  - sabotaged_function_code must be the sabotaged HELPER FUNCTION def block only.
  - test_args must be valid arguments for calling the SURFACE FUNCTION (not the helper).
  - expected_output and actual_output are the results of calling the SURFACE FUNCTION with those args
    (correct vs. buggy). Mentally trace how the helper's bug propagates up to the surface result.

Reply with ONLY a valid JSON object (no markdown, no explanation), matching this exact schema:

{
  "sabotaged_function_code": "<the complete sabotaged function as a Python string — only the function def block>",
  "test_cases": [
    {"args": "<Python literal tuple of args, e.g. (10, 3) or ('hello',)>", "correct_output": "<correct return value for the ORIGINAL function>"},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."}
  ],
  "bug_description": "<one sentence describing the FUNCTIONAL behavior that is now broken and why>"
}

Critical rules:
- sabotaged_function_code must be ONLY the function definition (def ...: ...) — nothing before or after.
- The `def` line MUST keep the EXACT SAME function name as the original. Do NOT rename the function.
- Preserve the EXACT original indentation level of the function (copy it from the input).
- Do NOT modify any docstrings or string literals that already exist in the function.
- Follow ONLY the steps listed in the SABOTAGE INSTRUCTIONS — do not add renaming or comments
  unless the instructions explicitly ask for them.
- The bug MUST produce a WRONG RETURN VALUE — NOT a crash or exception.
  The sabotaged function must still run without raising exceptions; it just returns the wrong result.
  If your bug causes a TypeError, AttributeError, or any other exception, it will be REJECTED.
  Change a condition, a boundary, an operator, or a value — not the structure so drastically it crashes.
- test_cases must contain EXACTLY 5 entries with DIVERSE inputs that exercise different code paths.
  Each "args" must be a Python tuple literal using ONLY these primitives: int, float, str, list, dict, bool.
  NEVER use: lambda, range, callable, custom classes, or any non-serializable object.
  NEVER use keyword arguments — only positional values inside the tuple.
  Wrong: ('hello', sep=',') or (x=1, y=2)  |  Right: ('hello', ',') or (1, 2)
  Each "correct_output" is the return value of the ORIGINAL (unfixed) function — not the buggy one.
  Example args: (10, 3) or (['a','b','c'],) or ("hello world",) or ([1, 2, 3],) — nothing else.
- Do NOT include any text outside the JSON object.
"""


def _extract_function_source(source: str, func_name: str) -> tuple[str, int, int]:
    """Return (func_source, start_line_0indexed, end_line_exclusive_0indexed)."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            start = node.lineno - 1          # 0-indexed
            end = node.end_lineno            # 0-indexed exclusive
            func_source = "".join(lines[start:end])
            return func_source, start, end
    raise ValueError(f"Function '{func_name}' not found in source.")


def _pick_best_function(source: str, exclude: set[str] | None = None) -> str:
    """Pick randomly from the top-3 MODULE-LEVEL functions that work with primitive types.
    exclude: function names already tried — deprioritised but used as last resort."""
    exclude = exclude or set()
    tree = ast.parse(source)
    scored: list[tuple[int, str]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") or node.name.startswith("_"):
            continue
        if node.name.startswith("test"):
            continue
        if node.end_lineno - node.lineno < 5:
            continue

        score = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, (int, float, str)):
                score += 2
            if isinstance(child, (ast.For, ast.While)):
                score += 3
            if isinstance(child, ast.If):
                score += 2
            if isinstance(child, ast.Return):
                score += 2
            if isinstance(child, ast.BinOp):
                score += 2
            if isinstance(child, ast.ListComp):
                score += 3
            if isinstance(child, ast.Compare):
                score += 1
            if isinstance(child, ast.Attribute):
                score -= 1  # obj.method() suggests class usage

        if score > 0:
            scored.append((score, node.name))

    if not scored:
        # Fallback: accept any public module-level function longer than 3 lines
        fallback_funcs = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
            and not node.name.startswith("test")
            and node.end_lineno - node.lineno >= 3
        ]
        if not fallback_funcs:
            raise ValueError("No suitable public module-level function found in the target file.")
        fresh = [f for f in fallback_funcs if f not in exclude] or fallback_funcs
        chosen = random.choice(fresh)
        print(f"[saboteur] Function candidates (fallback): {fallback_funcs} → chose '{chosen}'")
        return chosen

    scored.sort(key=lambda x: x[0], reverse=True)
    # Prefer functions not yet tried; fall back to all if all tried
    fresh_scored = [(s, n) for s, n in scored if n not in exclude]
    pool = fresh_scored[:3] if fresh_scored else scored[:3]
    _, chosen = random.choice(pool)
    print(f"[saboteur] Function candidates: {[n for _, n in scored[:5]]} → chose '{chosen}'"
          + (f" (excluded: {sorted(exclude)})" if exclude else ""))
    return chosen


def _find_called_module_functions(func_node: ast.FunctionDef, module_func_names: set) -> list:
    """Return names of module-level functions directly called inside func_node."""
    called = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in module_func_names:
                called.add(node.func.id)
    return list(called)


def _build_call_graph(module_func_nodes: dict) -> dict:
    """Return {name: set_of_module_level_functions_it_calls} for every function."""
    names = set(module_func_nodes.keys())
    return {
        name: set(_find_called_module_functions(node, names - {name}))
        for name, node in module_func_nodes.items()
    }


def _find_reachable(call_graph: dict, start: str, max_depth: int) -> dict:
    """
    BFS from start through call_graph up to max_depth hops.
    Returns {func_name: depth} for every reachable function (excluding start).
    Only includes functions that exist as keys in call_graph.
    """
    visited: dict[str, int] = {}
    queue: list[tuple[str, int]] = [(start, 0)]
    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for callee in call_graph.get(current, set()):
            if callee not in visited and callee in call_graph:
                visited[callee] = depth + 1
                queue.append((callee, depth + 1))
    return visited


def _find_call_path(call_graph: dict, start: str, target: str, max_depth: int) -> list[str]:
    """BFS: return shortest call path from start to target, e.g. ['a','b','c','target']."""
    queue: list[tuple[str, list[str]]] = [(start, [start])]
    visited = {start}
    while queue:
        current, path = queue.pop(0)
        if len(path) > max_depth + 1:
            continue
        for callee in call_graph.get(current, set()):
            if callee == target:
                return path + [callee]
            if callee not in visited:
                visited.add(callee)
                queue.append((callee, path + [callee]))
    return [start, target]  # fallback if direct path not found


def _pick_surface_function(source: str, max_depth: int,
                           exclude: set[str] | None = None) -> tuple[str, dict]:
    """
    Pick the surface function (entry point reported as broken to the student).
    Prefers functions that have the deepest reachable call chain, so bugs can
    be hidden many hops away from where the student starts investigating.
    exclude: set of function names already tried — skip them first, use them as last resort.
    Returns (surface_func_name, module_func_nodes).
    """
    exclude = exclude or set()
    tree = ast.parse(source)
    module_func_nodes = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
        and not node.name.startswith("test")
        and node.end_lineno - node.lineno >= 5
    }

    call_graph = _build_call_graph(module_func_nodes)

    # Score each function by the depth and count of its reachable targets
    caller_candidates: list[tuple[int, str]] = []
    for name in module_func_nodes:
        reachable = _find_reachable(call_graph, name, max_depth)
        substantial = {
            h: d for h, d in reachable.items()
            if module_func_nodes[h].end_lineno - module_func_nodes[h].lineno >= 5
        }
        if not substantial:
            continue
        score = sum((d + 1) * 20 for d in substantial.values())
        score += module_func_nodes[name].end_lineno - module_func_nodes[name].lineno
        caller_candidates.append((score, name))

    if caller_candidates:
        # Prefer candidates not yet tried; fall back to all if all were tried
        fresh = [(s, n) for s, n in caller_candidates if n not in exclude]
        pool = fresh if fresh else caller_candidates
        # Randomly sample from the pool weighted by score (higher score = more likely)
        total = sum(s for s, _ in pool)
        r = random.uniform(0, total)
        cumulative = 0
        chosen = pool[-1][1]
        for score, name in pool:
            cumulative += score
            if r <= cumulative:
                chosen = name
                break
        all_names = [n for _, n in caller_candidates]
        print(f"[saboteur] Surface candidates (deep chains): {all_names} → '{chosen}'"
              + (f" (excluded: {sorted(exclude)})" if exclude else ""))
        return chosen, module_func_nodes

    # Fallback: no function calls helpers — pick the most complex function directly
    print("[saboteur] No caller-surface found — falling back to direct mode.")
    all_funcs = [n for n in module_func_nodes if n not in exclude] or list(module_func_nodes)
    return _pick_best_function(source, exclude=exclude), module_func_nodes


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _try_exec(full_source: str, func_name: str, test_args_str: str,
              file_path: str | None = None) -> tuple[bool, str]:
    """
    Execute the full file source in an isolated namespace, call func_name(*args),
    and return (success, repr_of_result).  Falls back to (False, "") on any error.
    file_path: absolute path of the source file — its directory is added to sys.path
               so that sibling-module imports (e.g. 'import file2') can resolve.
    """
    import sys as _sys
    added_path: str | None = None
    if file_path:
        import os as _os
        dir_to_add = _os.path.dirname(_os.path.abspath(file_path))
        if dir_to_add not in _sys.path:
            _sys.path.insert(0, dir_to_add)
            added_path = dir_to_add
    namespace: dict = {"__name__": "__main__", "__file__": file_path or "<exec>"}
    try:
        exec(compile(full_source, file_path or "<exec>", "exec"), namespace)  # runs all imports + defs
        func = namespace.get(func_name)
        if func is None:
            return False, ""
        args = eval(test_args_str, {"__builtins__": __builtins__})
        if not isinstance(args, tuple):
            args = (args,)
        result = func(*args)
        return True, repr(result)
    except Exception as e:
        return False, f"ERROR:{type(e).__name__}: {e}"
    finally:
        if added_path and added_path in _sys.path:
            _sys.path.remove(added_path)


# Only words that unambiguously signal "this is a bug" — common English words excluded
_REVEAL_WORDS = {
    "incorrect", "incorrectly", "wrong", "wrongly", "bug", "buggy",
    "sabotage", "injected", "intentional", "deliberately", "purposely",
    "off-by-one", "off by one",
}


def _has_revealing_comment(original_func: str, sabotaged_func: str) -> bool:
    """Return True if GPT added a comment that gives away the bug."""
    orig_stripped = {line.strip() for line in original_func.splitlines()}
    for line in sabotaged_func.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and stripped not in orig_stripped:
            lower = stripped.lower()
            if any(word in lower for word in _REVEAL_WORDS):
                return True
    return False


def _variables_were_renamed(original_func: str, sabotaged_func: str) -> bool:
    """Return True if at least one local variable was renamed (or if there were none to rename)."""
    def local_store_names(src: str) -> set:
        try:
            tree = ast.parse(textwrap.dedent(src))
        except SyntaxError:
            return set()
        return {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

    orig = local_store_names(original_func)
    if not orig:
        return True  # nothing to rename; skip the check
    sabot = local_store_names(sabotaged_func)
    # At least one original local name must have disappeared (was renamed)
    return bool(orig - sabot)


def _format_bug_diff(
    func_name: str,
    file_start_line: int,
    file_end_line: int,
    original_source: str,
    sabotaged_source: str,
) -> str:
    """
    Return a compact architect-only diff showing exactly which file lines changed.
    file_start_line is 0-indexed; output uses 1-indexed line numbers.
    """
    orig_lines  = original_source.splitlines()
    sabot_lines = sabotaged_source.splitlines()

    header = (
        f"\n  ┌─ BUG INJECTED INTO: {func_name} "
        f"(file lines {file_start_line + 1}–{file_end_line})\n"
        f"  │"
    )
    diff_lines = []
    matcher = difflib.SequenceMatcher(None, orig_lines, sabot_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for k, line in enumerate(orig_lines[i1:i2]):
            file_no = file_start_line + i1 + k + 1  # 1-indexed
            diff_lines.append(f"  │  -{file_no:4d}: {line.rstrip()}")
        for line in sabot_lines[j1:j2]:
            diff_lines.append(f"  │  +     : {line.rstrip()}")

    if not diff_lines:
        diff_lines.append("  │  (no line-level changes detected)")

    return header + "\n".join([""] + diff_lines) + "\n  └─"


def _sabotage_one_helper(
    bug_func_name: str,
    current_source: str,
    surface_func_name: str,
    surface_source: str,
    instructions: str,
    llm,
    indirect_mode: bool,
    call_chain: list[str] | None = None,
    previous_bugs: list[str] | None = None,
    level: int = 1,
) -> tuple[str, dict] | tuple[None, None]:
    """
    Ask GPT to sabotage bug_func_name within current_source.
    call_chain is the path from surface_func_name down to bug_func_name.
    previous_bugs is a list of bug_description strings from prior failed attempts.
    level controls which validation checks apply (renaming required only for levels 2+).
    Splices result back. Returns (new_source, data_dict) or (None, None) on failure.
    """
    func_source, start_line, end_line = _extract_function_source(current_source, bug_func_name)

    def _build_user_content(extra: str = "") -> str:
        forbidden = ""
        if previous_bugs:
            listed = "\n".join(f"  - {b}" for b in previous_bugs)
            forbidden = (
                f"\n\nFORBIDDEN (already tried — use a COMPLETELY DIFFERENT bug type):\n{listed}"
            )
        if indirect_mode:
            chain_str = " → ".join(call_chain) if call_chain else f"{surface_func_name} → {bug_func_name}"
            depth = len(call_chain) - 1 if call_chain else 1
            return (
                f"SABOTAGE INSTRUCTIONS:\n{instructions}{forbidden}\n\n"
                f"INVESTIGATION CHAIN the student must follow (depth {depth}):\n"
                f"  {chain_str}\n"
                f"The student is only told that `{surface_func_name}` returns wrong results.\n"
                f"They must trace {depth} level(s) of calls to find the bug.\n\n"
                f"SURFACE FUNCTION (entry point the student tests — do NOT modify it):\n"
                f"```python\n{surface_source}\n```\n\n"
                f"HELPER FUNCTION TO SABOTAGE (inject the bug into THIS function only):\n"
                f"```python\n{func_source}\n```\n\n"
                f"CRITICAL: test_args must be valid arguments for calling `{surface_func_name}`. "
                f"expected_output and actual_output must be the results of calling "
                f"`{surface_func_name}(test_args)` — correct vs. buggy. "
                f"Trace how this helper's bug propagates up through {depth} call level(s) "
                f"to the surface return value.{extra}"
            )
        else:
            return (
                f"SABOTAGE INSTRUCTIONS:\n{instructions}{forbidden}\n\n"
                f"FUNCTION TO SABOTAGE:\n```python\n{func_source}\n```{extra}"
            )

    attempted_bugs: list[str] = []

    for attempt in range(1, 4):
        extra = ""
        if attempted_bugs:
            listed = "\n".join(f"  - {b}" for b in attempted_bugs)
            extra = f"\n\nThis is retry {attempt}. Previous attempts in THIS call used:\n{listed}\nChoose a DIFFERENT bug approach now."
        user_content = _build_user_content(extra)
        print(f"[saboteur] GPT call (func='{bug_func_name}', attempt={attempt})…")
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        try:
            data = _parse_response(response.content)
        except json.JSONDecodeError as e:
            print(f"[saboteur] JSON parse error: {e}")
            continue

        sabotaged_func = data["sabotaged_function_code"]
        # Track what bug was attempted so subsequent retries avoid repeating it
        if desc := data.get("bug_description", ""):
            attempted_bugs.append(desc)

        # Splice back into current_source
        lines = current_source.splitlines(keepends=True)
        candidate_source = "".join(lines[:start_line]) + sabotaged_func + "\n" + "".join(lines[end_line:])

        # Validate syntax
        try:
            new_tree = ast.parse(candidate_source)
        except SyntaxError as e:
            print(f"[saboteur] Syntax error: {e}")
            continue

        # Verify function name preserved
        new_names = {
            n.name for n in new_tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if bug_func_name not in new_names:
            print(f"[saboteur] GPT renamed '{bug_func_name}' — retrying")
            continue

        # Reject if any new comment reveals the bug
        if _has_revealing_comment(func_source, sabotaged_func):
            print(f"[saboteur] Comment reveals bug on attempt {attempt} — retrying")
            continue

        # Renaming is required for levels 2 and 3 only
        if level >= 2 and not _variables_were_renamed(func_source, sabotaged_func):
            if attempt < 3:
                print(f"[saboteur] No variable renaming detected on attempt {attempt} — retrying")
                continue
            print(f"[saboteur] WARNING: variable renaming skipped by GPT — accepting anyway")

        # Attach debug metadata (not visible to student)
        data["_debug_func_name"]   = bug_func_name
        data["_debug_start_line"]  = start_line
        data["_debug_end_line"]    = end_line
        data["_debug_func_source"] = func_source
        data["_debug_sabot_func"]  = sabotaged_func
        return candidate_source, data

    return None, None


def sabotage(state: ArchitectState) -> ArchitectState:
    level = state["difficulty_level"]
    instructions = _LEVEL_INSTRUCTIONS.get(level, _LEVEL_INSTRUCTIONS[1])

    n_bugs    = _BUGS_PER_LEVEL.get(level, 1)
    max_depth = _MAX_DEPTH_LEVEL.get(level, 2)

    # Build the queue of files to try: chosen file first, then all other candidates
    candidate_files: list[str] = list(state.get("candidate_files") or [state["target_file"]])
    tried_files: set[str] = set()

    max_outer_attempts = 3
    while candidate_files:
        current_file = candidate_files.pop(0)
        if current_file in tried_files:
            continue
        tried_files.add(current_file)

        with open(current_file, encoding="utf-8", errors="ignore") as _f:
            source = _f.read()

        print(f"[saboteur] Trying file: {current_file}")
        all_previous_bugs: list[str] = []  # accumulates across outer attempts for this file
        tried_surfaces: set[str] = set()   # surface functions already attempted for this file

        for outer in range(1, max_outer_attempts + 1):
            # Pick the surface function: avoid repeating the same one each attempt
            try:
                surface_func_name, module_func_nodes = _pick_surface_function(
                    source, max_depth, exclude=tried_surfaces
                )
            except ValueError as e:
                print(f"[saboteur] No usable function in {current_file}: {e} — trying next file.")
                break  # break for-outer → continue while candidate_files
            tried_surfaces.add(surface_func_name)
            call_graph = _build_call_graph(module_func_nodes)

            # Find ALL functions reachable from surface within max_depth hops
            reachable = _find_reachable(call_graph, surface_func_name, max_depth)
            substantial = {
                h: d for h, d in reachable.items()
                if module_func_nodes.get(h)
                and module_func_nodes[h].end_lineno - module_func_nodes[h].lineno >= 5
            }

            # Prefer the deepest available targets — harder for student to find
            indirect_mode = bool(substantial)
            if substantial:
                max_avail = max(substantial.values())
                # Try to pick from the deepest level; fall back one level at a time if needed
                for target_depth in range(max_avail, 0, -1):
                    pool = [h for h, d in substantial.items() if d == target_depth]
                    if pool:
                        break
                bug_targets = random.sample(pool, min(n_bugs, len(pool)))
                # Also add a shallower bug if n_bugs > len(pool) and shallower targets exist
                if n_bugs > len(pool):
                    shallower = [h for h, d in substantial.items()
                                 if d < target_depth and h not in bug_targets]
                    if shallower:
                        bug_targets += random.sample(shallower, min(n_bugs - len(pool), len(shallower)))
            else:
                bug_targets = [surface_func_name]

            # Compute call chains for each bug target (for GPT context and debug output)
            bug_chains: dict[str, list[str]] = {
                t: _find_call_path(call_graph, surface_func_name, t, max_depth)
                for t in bug_targets
            }

            print(f"[saboteur] Outer attempt {outer}: surface='{surface_func_name}', "
                  f"bug_targets={bug_targets}, max_depth={max_depth}, indirect={indirect_mode}")
            for t, chain in bug_chains.items():
                print(f"[saboteur]   chain to '{t}': {' → '.join(chain)} (depth {len(chain)-1})")

            surface_source = ""
            if indirect_mode:
                surface_source, _, _ = _extract_function_source(source, surface_func_name)

            llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

            # Sabotage each target in REVERSE line order to avoid position shifts during splicing
            bug_targets_sorted = sorted(
                bug_targets,
                key=lambda h: module_func_nodes[h].lineno,
                reverse=True,
            )

            current_source = source
            all_descriptions: list[str] = []
            all_data: list[dict] = []
            last_data: dict | None = None
            failed = False

            for bug_func_name in bug_targets_sorted:
                new_source, data = _sabotage_one_helper(
                    bug_func_name, current_source,
                    surface_func_name, surface_source,
                    instructions, llm, indirect_mode,
                    call_chain=bug_chains.get(bug_func_name),
                    previous_bugs=list(all_previous_bugs),
                    level=level,
                )
                if new_source is None:
                    print(f"[saboteur] Failed to sabotage '{bug_func_name}' — retrying outer loop")
                    failed = True
                    break
                current_source = new_source
                all_descriptions.append(data["bug_description"])
                all_data.append(data)
                last_data = data
                if desc := data.get("bug_description", ""):
                    all_previous_bugs.append(desc)

            if failed or last_data is None:
                continue

            # Verify all test cases — only keep cases where the ORIGINAL function runs cleanly.
            # Cases that crash on the original have bad args (not the bug) and must be excluded.
            raw_cases = last_data.get("test_cases", [])
            if not raw_cases:
                print("[saboteur] GPT returned no test_cases — retrying outer loop")
                continue

            verified_cases: list[dict] = []
            first_fail_args = first_expected = first_actual = None
            exec_possible = False  # True if at least one case ran cleanly on BOTH versions

            for tc in raw_cases:
                args = tc.get("args", "()")

                # Reject args that contain non-primitive constructs GPT shouldn't be using
                if "lambda" in args or "range(" in args or "<function" in args:
                    print(f"[saboteur] Skipping case {args!r}: contains non-primitive (lambda/range)")
                    continue

                # Reject args with keyword arguments (e.g. func(x, key=val) style) —
                # not valid Python tuple literals and can't be eval'd as positional args
                try:
                    eval(args, {"__builtins__": {}})
                except SyntaxError:
                    print(f"[saboteur] Skipping case {args!r}: invalid tuple syntax (keyword args?)")
                    continue
                except Exception:
                    pass  # other eval errors (name errors etc.) will surface in _try_exec

                orig_ok, true_exp = _try_exec(source, surface_func_name, args,
                                              file_path=current_file)
                if not orig_ok:
                    print(f"[saboteur] Skipping case {args!r}: original crashed — {true_exp[:80]}")
                    continue

                # Reject expected values that can't be stored as portable literals
                # (i.e. repr contains custom class names not importable in challenge_run.py)
                try:
                    eval(true_exp, {"__builtins__": __builtins__})
                except Exception:
                    print(f"[saboteur] Skipping case {args!r}: expected repr not a portable literal ({true_exp[:60]})")
                    continue

                sabot_ok, true_act = _try_exec(current_source, surface_func_name, args,
                                               file_path=current_file)
                if not sabot_ok:
                    # Bug causes a crash instead of a wrong value — not a functional bug
                    print(f"[saboteur] Skipping case {args!r}: sabotaged version crashed (not a value bug)")
                    continue

                exec_possible = True
                verified_cases.append({"args": args, "expected": true_exp})

                if first_fail_args is None and true_exp != true_act:
                    first_fail_args, first_expected, first_actual = args, true_exp, true_act

            if not exec_possible:
                # All cases either crashed the original, crashed the sabotaged version, or
                # used non-primitive args — this bug type is crash-based, not value-based.
                print("[saboteur] Bug produces crashes instead of wrong values — retrying outer loop.")
                continue

            if first_fail_args is None:
                print("[saboteur] No test case exposes the bug after exec — retrying outer loop")
                continue

            print(f"[saboteur] Verified {len(verified_cases)} test cases; "
                  f"first failing: {first_fail_args} → expected={first_expected}, got={first_actual}")

            # Success — update target_file/original_code in case we used a fallback file
            state["target_file"]     = current_file
            state["original_code"]   = source
            state["sabotaged_code"]  = current_source
            state["function_name"]   = surface_func_name
            state["test_args"]       = first_fail_args
            state["expected_output"] = first_expected
            state["actual_output"]   = first_actual
            state["test_cases"]      = verified_cases
            state["bug_description"] = " | ".join(all_descriptions)

            print(f"[saboteur] Reported broken : {surface_func_name}")
            for t in bug_targets:
                chain = bug_chains.get(t, [surface_func_name, t])
                print(f"[saboteur] Bug in '{t}' at depth {len(chain)-1}: {' → '.join(chain)}")
            print(f"[saboteur] Bug(s): {state['bug_description']}")

            # ── DEBUG: show exactly which lines changed ──────────────────────────
            for target_data in all_data:
                diff_str = _format_bug_diff(
                    func_name        = target_data["_debug_func_name"],
                    file_start_line  = target_data["_debug_start_line"],
                    file_end_line    = target_data["_debug_end_line"],
                    original_source  = target_data["_debug_func_source"],
                    sabotaged_source = target_data["_debug_sabot_func"],
                )
                print(diff_str)
            # ─────────────────────────────────────────────────────────────────────

            return state

        else:
            # for-loop exhausted all outer attempts without success — try next candidate file
            print(f"[saboteur] All {max_outer_attempts} attempts failed for "
                  f"{current_file} — trying next file.")

    raise RuntimeError(
        f"This repository's functions all require package-level imports or globals and cannot "
        f"be exec-tested in isolation (tried {len(tried_files)} file(s)). "
        f"Try a repo that has standalone utility functions with primitive inputs/outputs "
        f"(e.g. math helpers, string processors, algorithms)."
    )
