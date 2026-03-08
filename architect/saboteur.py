import ast
import difflib
import json
import os
import random
import textwrap

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from architect.state import ArchitectState

# Maximum call-chain depth to descend when picking bug targets
# Level 1 → depth 2 (surface → helper)
# Level 2 → depth 3 (surface → mid → helper)
# Level 3 → depth 4 (surface → mid → mid2 → helper)
_MAX_DEPTH_LEVEL = {1: 4, 2: 4, 3: 4}

# All levels share the same bug-injection instruction.
# Structural transformations (obfuscation, spaghettification) are applied as
# separate post-passes in sabotage() based on the difficulty level.
_BUG_INJECTION_INSTRUCTION = """
You are sabotaging a SINGLE Python function with an AI-RESISTANT bug.

STEP 1 — Understand the function contract:
  Read the function carefully. Identify exactly what it is SUPPOSED to do:
  its inputs, its expected outputs, and its core logic.

STEP 2 — Invent and inject ONE AI-RESISTANT FUNCTIONAL bug of your own choosing:
  The bug MUST be subtle, context-dependent, and extremely hard for an LLM to identify
  by reading the function in isolation. It should require deep understanding of the
  surrounding logic or call context to detect.

  PREFERRED bug types (pick whichever fits this specific function best):
    * Off-by-one in a specific edge case — wrong loop bound or index that only
      matters for a particular pattern of inputs (e.g. empty sequences, odd-length lists)
    * Subtle operator swap that is only wrong in certain conditions — e.g. < vs <=,
      + vs -, | vs &, that look correct at first glance but fail on boundary values
    * Wrong variable used in a rare branch — copies a visually similar variable name
      in the wrong context so casual readers assume it is correct
    * Precedence error — missing parentheses that change evaluation order only for
      specific operand combinations, not for the common case
    * State-dependent side effect — a mutation or accumulation that only causes a
      visible result mismatch after specific sequences of operations
    * Subtle type / comparison issue — comparing by identity instead of value, or an
      implicit type coercion that produces the wrong result for certain input types
    * Wrong default assumption — a constant, threshold, or sentinel that is slightly
      off (e.g. 0 instead of 1, -1 instead of None) but only matters in edge cases

  Requirements:
    * The code must still run without exceptions on valid inputs
    * The function must return a WRONG RESULT for at least some inputs
    * The bug must be non-obvious — a student must read the logic carefully to spot it
    * The bug must NOT be trivially fixable by an LLM reading the function in isolation;
      it should look plausible and correct at a quick glance
  Make 1–3 small coordinated changes to produce the behavioral failure.
  Do NOT rename any variables, do NOT add comments.
  Return the function with ONLY the bug injected — keep everything else identical.
"""

_LEVEL_INSTRUCTIONS = {1: _BUG_INJECTION_INSTRUCTION,
                       2: _BUG_INJECTION_INSTRUCTION,
                       3: _BUG_INJECTION_INSTRUCTION}

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
- test_cases must contain EXACTLY 8 entries with DIVERSE inputs that exercise different code paths.
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
        dir_to_add = os.path.dirname(os.path.abspath(file_path))
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
    """Return True if at least one parameter or local variable was renamed."""
    def all_user_names(src: str) -> set:
        try:
            tree = ast.parse(textwrap.dedent(src))
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            # local variables
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            # parameters
            if isinstance(node, ast.arguments):
                for arg in node.args + node.posonlyargs + node.kwonlyargs:
                    names.add(arg.arg)
                if node.vararg:
                    names.add(node.vararg.arg)
                if node.kwarg:
                    names.add(node.kwarg.arg)
        return names

    orig = all_user_names(original_func)
    if not orig:
        return True  # nothing to rename; skip the check
    sabot = all_user_names(sabotaged_func)
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

        # Attach debug metadata (not visible to student)
        data["_debug_func_name"]   = bug_func_name
        data["_debug_start_line"]  = start_line
        data["_debug_end_line"]    = end_line
        data["_debug_func_source"] = func_source
        data["_debug_sabot_func"]  = sabotaged_func
        return candidate_source, data

    return None, None


def _strip_markdown_code(text: str) -> str:
    """Strip ```python ... ``` or ``` ... ``` fences from a GPT response."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("python"):
            text = text[6:]
        text = text.strip()
    return text


def _chain_for_obfuscation(source: str, surface_func: str, max_depth: int = 4) -> set:
    """BFS call-chain discovery from surface_func through all module-level functions."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {surface_func}
    defined = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    chain: set = set()
    frontier = {surface_func}
    for _ in range(max_depth):
        next_f: set = set()
        for fname in frontier:
            if fname in chain or fname not in defined:
                continue
            chain.add(fname)
            for n in ast.walk(defined[fname]):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    if n.func.id in defined and n.func.id not in chain:
                        next_f.add(n.func.id)
        frontier = next_f
        if not frontier:
            break
    return chain


def _extract_chain_snippet(source: str, chain: set) -> tuple:
    """
    Build a mini source: header imports + only the chain functions.
    Returns (mini_source, last_end_lineno_in_original).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, len(source.splitlines())
    lines = source.splitlines(keepends=True)
    header: list = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)) and node.lineno <= 80:
            header.extend(lines[node.lineno - 1:node.end_lineno])
    blocks: list = []
    last_end = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in chain:
            blocks.append("".join(lines[node.lineno - 1:node.end_lineno]))
            last_end = max(last_end, node.end_lineno)
    mini = "".join(header) + "\n\n" + "\n\n".join(blocks)
    return mini, last_end


def _splice_transforms_back(original: str, transformed_snippet: str, orig_chain: set) -> str:
    """
    Splice transformed functions back into the original file.
    - Functions in orig_chain (same name) -> replace in-place
    - New functions (wrappers/ghosts) -> insert after the last replaced block
    Returns merged source, or original on any parse error.
    """
    try:
        orig_tree = ast.parse(original)
        new_tree  = ast.parse(transformed_snippet)
    except SyntaxError:
        return original

    orig_lines = original.splitlines(keepends=True)
    new_lines  = transformed_snippet.splitlines(keepends=True)

    new_funcs: dict = {}
    for node in new_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new_funcs[node.name] = "".join(new_lines[node.lineno - 1:node.end_lineno])

    orig_pos: dict = {}
    for node in orig_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            orig_pos[node.name] = (node.lineno - 1, node.end_lineno)

    replacements: list = []
    insertions:   list = []
    last_replace_end = -1

    for name, src in new_funcs.items():
        if name in orig_pos:
            start, end = orig_pos[name]
            replacements.append((start, end, src))
            last_replace_end = max(last_replace_end, end)
        else:
            insertions.append(src)

    if not replacements:
        return original

    replacements.sort(key=lambda x: x[0], reverse=True)
    result = list(orig_lines)
    size_delta = 0
    for start, end, new_src in replacements:
        new_src_lines = (new_src.rstrip("\n") + "\n").splitlines(keepends=True)
        old_size = end - start
        result[start:end] = new_src_lines
        size_delta += len(new_src_lines) - old_size

    if insertions and last_replace_end >= 0:
        adjusted = last_replace_end + size_delta
        insert_block = "\n\n" + "\n\n".join(s.rstrip("\n") for s in insertions) + "\n\n"
        result.insert(adjusted, insert_block)

    return "".join(result)


# Horizontal & Vertical Expansion rules applied to EVERY function created or modified.
_EXPANSION_RULES = """
MANDATORY EXPANSION RULES — apply to EVERY function you write or modify:

  !! ZERO TOLERANCE ANTI-PATTERN — NEVER produce this: !!

    def _process_segment_core(data_ref, sep=',', prefix=''):
        \"\"\"Core processing layer.\"\"\"
        return _process_segment_impl(data_ref, sep, prefix)    # <-- VIOLATION

  A function whose body is ONLY a return-delegation call (1-3 lines total) is a
  HARD FAILURE.  It will be automatically rejected.  Every wrapper you write —
  even a "thin" pass-through — MUST look like the CORRECT PATTERN below.

  !! CORRECT PATTERN — every wrapper MUST look like this (50+ lines minimum): !!

    def _process_segment_core(data_ref, sep=',', prefix='', flag=False):
        \"\"\"Core processing layer.\"\"\"
        import sys as _sys_ref
        global _state_flux
        _buffer_offset = len(str(data_ref)) * 1
        _tmp_flag = not False
        _dead_counter = 0
        _entropy_val = _buffer_offset - _buffer_offset
        _noise_a = sum(i for i in range(min(_buffer_offset, 5)))
        _noise_b = _noise_a - _noise_a
        _checksum_pre = (_buffer_offset % 7) - (_buffer_offset % 7)
        _ref_copy = data_ref
        _discarded_a = str(_ref_copy)[::-1][::-1]
        if _tmp_flag and _buffer_offset >= 0:
            _dead_list = [x * 0 for x in range(6)]
            _dead_result = sum(_dead_list)
            if not (flag is None and False):
                _checksum = _dead_result % 3
                if _checksum >= 0:                     # always evaluates true
                    _tmp_marker = bool(data_ref is not None)
                    if _tmp_marker == True:             # noqa
                        _state_flux = _dead_result
                        _entropy_b = len(_discarded_a) - len(_discarded_a)
                        # do not touch
                        if _entropy_b == 0:
                            _noise_c = [i - i for i in range(4)]
                            _ = sum(_noise_c)
                            # logic starts here
                            _sep_copy = sep if sep is not None else sep
                            _pre_copy = prefix if not (prefix is None and False) else prefix
                            _dead_counter += 0
                            try:
                                _intermediate = _process_segment_impl(
                                    _ref_copy, _sep_copy, _pre_copy, flag
                                )
                                _post_check = _intermediate is not None
                                if _post_check == True:  # noqa
                                    _dead_counter += 0
                                    _noise_d = _buffer_offset * 0
                                    return _intermediate
                            except Exception as _e_flux:
                                raise
        return _process_segment_impl(data_ref, sep, prefix, flag)

  EVERY function you write MUST follow this pattern: dense busy-work BEFORE and
  AFTER the real call, nested conditionals that always evaluate true, dead variables,
  global declarations, and the actual delegation buried in the middle.

  EXPANSION 1 — EXTREME FUNCTION BLOATING (MINIMUM 50 LINES, NO EXCEPTIONS):
    Every single function — wrapper, helper, ghost, or original — MUST be at least
    50 lines long. If you are about to write a function shorter than 50 lines, STOP
    and add more padding until it reaches 50+.
    Required padding techniques (use ALL of them):
    - At least 8-10 dummy local variable assignments at the top: _buf, _flag, _counter, _entropy
    - At least 2 loop-based dummy computations: `_n = [x*0 for x in range(5)]`
    - At least 3 levels of always-true nested if/else wrapping the real call
    - `global _state_flux` declaration inside the function
    - `import <module>` statement inside the function body
    - At least 3 trash comments: `# do not touch`, `# logic starts here`, `# ??`
    - A try/except block wrapping the actual delegation call

  EXPANSION 2 — CALL-STACK OVERLOAD:
    Inside each function, inject multiple calls to other newly created helper functions.
    Build a web of dependencies: funcA calls funcB, funcC, funcD; each of those calls
    3 more. The call graph must be deep and wide so tracing execution is exhausting.
    Every helper in the web must also follow EXPANSION 1 (bloated to 50+ lines).

  EXPANSION 3 — "WHERE IS THE LOGIC?" CHALLENGE:
    Surround every real/meaningful call with 5-10 fake "busy-work" calls whose names
    suggest they are critical infrastructure but actually do nothing:
        audit_buffer_state()   verify_integrity_checksum()   sync_internal_stack()
        flush_pipeline_cache()  reindex_lookup_table()   hydrate_context_frame()
    A debugger stepping through must wade past all these calls to find the one that matters.

  EXPANSION 4 — DEEP NESTING & INDENTATION:
    Push the actual core logic 5-7 levels deep using nested if/else and try/except blocks.
    Each nesting level should have a plausible-looking (but ultimately irrelevant) condition.
    The real work must be buried at the deepest indentation level, far to the right of screen.

  EXPANSION 5 — BUG BURIAL:
    The injected bug (or the core logic that contains it) MUST NOT appear at the top or
    bottom of any long function. Bury it around line 40-60 of the function body, inside
    a deeply nested block, preceded and followed by dense "busy-work" padding code.
    This ensures the bug is invisible at first glance and exhausts token budgets.
"""


def _obfuscate_full_file(source: str, surface_func_name: str, llm, level: int = 1,
                         verified_cases: list | None = None, file_path: str | None = None,
                         bug_func_name: str | None = None, buggy_func_source: str | None = None) -> str:
    """
    Level 1 post-pass: legacy obfuscation — cryptic names, deprecated patterns, global vars,
    mixed styling, removed comments.  When level==3, also injects red-herring decoys.
    Only the file where the bug was injected is transformed; no other files are touched.
    After transformation, verifies that surface_func_name still exists and the bug still
    manifests on at least one verified_case. Falls back to original source on any failure.
    """
    # Extract only the chain functions — avoids sending the whole file to GPT
    chain = _chain_for_obfuscation(source, surface_func_name)
    mini_source, _ = _extract_chain_snippet(source, chain)

    red_herring_rule = ""
    if level == 3:
        red_herring_rule = (
            "  RULE 8 \u2014 RED HERRINGS (Level 3 only):\n"
            "    Inject 2-3 convincing decoy bugs that look real but have NO effect on output.\n"
            "    Examples:\n"
            "    - A dead if-branch (condition always False) with suspicious-looking code\n"
            "    - A local variable overwritten immediately before use (looks like wrong value)\n"
            "    - A commented-out line hinting at a past fix: `# old fix: x -= 1`\n"
            "    - Suspicious math `result = result * 1.0000001` that looks wrong but is harmless\n"
            "    These MUST NOT change the observable output \u2014 pure misdirection only.\n\n"
        )

    bug_preservation_rule = ""
    if bug_func_name and buggy_func_source:
        bug_preservation_rule = (
            f"  !!! CRITICAL BUG PRESERVATION — read this before doing ANYTHING !!!\n"
            f"  The function `{bug_func_name}` contains an intentional, hidden bug.\n"
            f"  When you wrap or restructure this function, you MUST place its logic\n"
            f"  EXACTLY as shown below into the innermost wrapper — not paraphrased,\n"
            f"  not rewritten, not 'cleaned up'. Copy every operator, every condition,\n"
            f"  every loop bound VERBATIM. You may rename parameters/variables for\n"
            f"  obfuscation, but the LOGICAL STRUCTURE must be byte-for-byte identical.\n"
            f"  The original buggy function body:\n\n"
            f"{buggy_func_source}\n\n"
            f"  Preserve this logic EXACTLY inside the deepest wrapper.\n\n"
        )

    prompt = (
        f"Refactor ONLY the following Python file into LEVEL 1 — Obfuscated Legacy code.\n"
        f"SCOPE: Modify ONLY this file. Do NOT touch any other file in the project.\n"
        f"All logic and any existing bugs MUST be preserved EXACTLY — same inputs, same outputs.\n\n"
        f"=== LEVEL 1 RULES — apply ALL of them aggressively ===\n\n"
        f"  RULE 1 — MANDATORY SHADOW WRAPPING (at least 4 calls deep):\n"
        f"    Every functional logic path must pass through at least 4 function calls.\n"
        f"    If the original logic is short, create 'Shadow Wrapper' functions around it that\n"
        f"    perform redundant checks or dummy transformations before delegating:\n"
        f"      e.g.  x = (x + 0) * 1   or   val = val if val is not None else val\n"
        f"    Structure every function so its real work is 4 calls deep from the entry point.\n\n"
        f"  RULE 1b — NAMING OF NEW WRAPPER & GHOST FUNCTIONS:\n"
        f"    Every new function you create (shadow wrappers, ghost helpers, etc.)\n"
        f"    MUST have a name that:\n"
        f"    a) Sounds domain-relevant and meaningful — not random noise. It should look\n"
        f"       like a real, important function someone wrote intentionally.\n"
        f"    b) Is easily confused with the REAL functions in the file. If the real\n"
        f"       function is `{surface_func_name}`, create wrappers named things like:\n"
        f"       `_{surface_func_name}_core`, `_{surface_func_name}_impl`,\n"
        f"       `_{surface_func_name}_internal`, `_{surface_func_name}_dispatch`\n"
        f"       Or use near-identical names with a subtle difference: one extra underscore,\n"
        f"       a digit suffix, a transposed letter (e.g. `slpit_format_str` vs `split`).\n"
        f"    c) Makes it IMPOSSIBLE to tell at a glance which function does the real work\n"
        f"       and which ones are just wrappers or dead weight.\n"
        f"    Goal: a reader scanning function names should find 5-10 candidates that ALL\n"
        f"    look like the entry point, with no obvious clue which is authoritative.\n\n"
        f"  RULE 2 — CRYPTIC & MISLEADING VARIABLE/PARAM NAMES:\n"
        f"    Rename ALL parameters and ALL local variables inside every function to\n"
        f"    visually confusing or completely misleading names.\n"
        f"    Mandatory mix of these styles:\n"
        f"    - Look-alike names: l1I, O0O, lI1, Il1I, oO0, temp_val_01, x1l, lx1\n"
        f"    - Irrelevant nouns: pigeon, banana, turnip, flux, zeta, gloop, rutabaga\n"
        f"    - Misleading antonyms: a True flag named `isEmpty`, a counter named `totalIgnored`\n"
        f"    - Mix camelCase / snake_case / PascalCase / ALL_CAPS with zero consistency\n"
        f"    - Update ALL call sites to use the renamed identifiers consistently.\n\n"
        f"  RULE 3 — NESTED IF-ELSE MAZE & BOOLEAN OBFUSCATION:\n"
        f"    Inside EVERY function, inject complex nested if/else chains with redundant,\n"
        f"    convoluted boolean logic to obscure the actual execution path:\n"
        f"    - Use `&`, `|`, `^`, `not not`, `bool()`, `== True`, `!= False`,\n"
        f"      double negations `not (not x)`, pointless `x * 1 + 0 - 0`\n"
        f"    - Conditions that look uncertain but always evaluate to the same path\n"
        f"    - Goal: an AI must struggle to map variables to their actual purpose.\n\n"
        f"  RULE 4 — GLOBAL DECLARATIONS & DEPRECATED SYNTAX:\n"
        f"    Add unnecessary `global _state_flux` declarations inside functions.\n"
        f"    Use old-style string formatting (`%s`, `%d`) instead of f-strings.\n"
        f"    Add pointless `import` statements inside function bodies.\n\n"
        f"  RULE 5 — TRASH COMMENTS (remove all useful ones):\n"
        f"    Delete every helpful comment. Replace with maximally useless noise:\n"
        f"    `# logic starts here`, `# do not touch`, `# entropy increases`, `# ??`\n"
        f"    Add comments that describe the OPPOSITE of what the next line does.\n\n"
        f"  RULE 6 — DEAD WEIGHT:\n"
        f"    Add ghost variables assigned but never read: `_dead = x * 3`\n"
        f"    Add ghost functions defined but never called.\n"
        f"    Sprinkle `if True: pass` blocks and useless intermediate assignments.\n\n"
        f"{red_herring_rule}"
        f"{bug_preservation_rule}"
        f"HARD CONSTRAINTS (NEVER violate):\n"
        f"  - Keep `{surface_func_name}` with its EXACT original name and signature.\n"
        f"  - Do NOT fix any bugs — all buggy behaviour MUST be preserved exactly.\n"
        f"  - The file MUST remain valid, runnable Python (no syntax errors).\n"
        f"  - Do NOT mix tabs and spaces for indentation (Python forbids it).\n"
        f"  - Keep all module-level import statements unchanged.\n"
        f"  - Return ONLY the complete modified Python file — no explanation, no markdown.\n\n"
        f"FUNCTIONS TO TRANSFORM (extracted from a larger file):\n{mini_source}"
        f"\n\n{_EXPANSION_RULES}"
        f"\n\nCRITICAL OUTPUT FORMAT: Return ONLY Python function definitions "
        f"(def ...: ...) — no imports, no module-level code, no markdown fences. "
        f"Include EVERY new wrapper/helper function you create."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        snippet = _strip_markdown_code(response.content)
        snip_tree = ast.parse(snippet)
    except Exception as e:
        print(f"[saboteur] Obfuscation failed (parse/call): {e} — keeping original")
        return source

    # Verify the surface function was not renamed by GPT
    all_func_names = {
        n.name for n in ast.walk(snip_tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if surface_func_name not in all_func_names:
        print(f"[saboteur] Obfuscation renamed `{surface_func_name}` — reverting")
        return source

    # Splice the transformed snippet back into the full original file
    result = _splice_transforms_back(source, snippet, chain)

    # Verify the bug still manifests on at least one test case in the spliced result
    if verified_cases:
        bug_still_present = False
        for tc in verified_cases:
            ok, act = _try_exec(result, surface_func_name, tc["args"], file_path=file_path)
            if ok and act != tc["expected"]:
                bug_still_present = True
                break
        if not bug_still_present:
            print("[saboteur] Obfuscation removed the bug — reverting")
            return source

    print(f"[saboteur] Level {level} obfuscation applied ({len(result.splitlines())} lines)")
    return result


def _spaghettify_file(source: str, surface_func_name: str, llm,
                      file_path: str, verified_cases: list,
                      bug_func_name: str | None = None, buggy_func_source: str | None = None) -> tuple[str, list]:
    """
    Level 2 post-pass: Black Box Wrapping — buries the core logic of every function
    inside 4-5 layers of complex wrapper functions filled with conditional mazes,
    obfuscated data passing, and busy-work logic. Only the target file is transformed.
    Returns (new_source, re_verified_cases). Falls back to (source, verified_cases) if the
    transformation breaks syntax or removes the bug from all test cases.
    """
    # Extract only the chain functions — avoids sending the whole file to GPT
    chain = _chain_for_obfuscation(source, surface_func_name)
    mini_source, _ = _extract_chain_snippet(source, chain)

    bug_preservation_rule = ""
    if bug_func_name and buggy_func_source:
        bug_preservation_rule = (
            f"  !!! CRITICAL BUG PRESERVATION — read this before doing ANYTHING !!!\n"
            f"  The function `{bug_func_name}` contains an intentional, hidden bug.\n"
            f"  When you wrap or move this function's logic, you MUST place it EXACTLY\n"
            f"  as shown below into the innermost helper — not paraphrased, not rewritten.\n"
            f"  Copy every operator, every condition, every loop bound VERBATIM.\n"
            f"  You may rename parameters/variables for the Level 2 naming scheme,\n"
            f"  but the LOGICAL STRUCTURE must be byte-for-byte identical.\n"
            f"  The original buggy function body:\n\n"
            f"{buggy_func_source}\n\n"
            f"  Preserve this logic EXACTLY inside the deepest wrapper.\n\n"
        )

    prompt = (
        f"Refactor ONLY the following Python file into LEVEL 2 — Extreme Spaghetti (The Maze).\n"
        f"SCOPE: Modify ONLY this file. Do NOT touch any other file in the project.\n"
        f"All existing logic and any bugs MUST be preserved EXACTLY — same inputs, same outputs.\n\n"
        f"=== LEVEL 2 RULES — apply ALL aggressively ===\n\n"
        f"  IMPORTANT — NAMING CONVENTION FOR LEVEL 2:\n"
        f"    Use NORMAL, STANDARD, READABLE names for all functions and variables.\n"
        f"    The chaos comes from STRUCTURE and VOLUME, NOT from cryptic names.\n"
        f"    New helpers must have plausible descriptive names such as:\n"
        f"    process_segment, validate_entry, resolve_token, compute_result, apply_filter,\n"
        f"    check_boundary, normalize_value, handle_edge_case, prepare_context, finalize_output\n\n"
        f"  RULE 1 — EXTREME NESTING (minimum 10 function calls deep):\n"
        f"    Shatter every original function into a NON-LINEAR chain of at least 10\n"
        f"    interconnected helpers. The original function body MUST be MOVED (not copied)\n"
        f"    into the innermost helper. The entry-point function keeps its exact name and\n"
        f"    signature but becomes a thin dispatcher only.\n"
        f"    Example chain for `my_func(x, y)` — 10+ layers:\n"
        f"      my_func -> dispatch_entry -> resolve_pipeline -> evaluate_context ->\n"
        f"      prepare_execution -> validate_inputs -> check_preconditions ->\n"
        f"      normalize_arguments -> apply_transformations -> execute_core -> finalize_output\n"
        f"    The real original logic (with the bug) lives in execute_core or finalize_output.\n\n"
        f"  RULE 2 — FUNCTION OVERLOAD (each helper calls 3-5 others):\n"
        f"    Every wrapper in the chain must call 3-5 additional busy-work helper functions\n"
        f"    before delegating to the next layer. This creates a wide AND deep call graph.\n"
        f"    CRITICAL: Every busy-work helper must contain REAL code, NOT bare `pass`.\n"
        f"    Example of a valid busy-work helper:\n"
        f"      def verify_integrity(data):\n"
        f"          buffer = [ord(c) for c in str(data)[:8]]\n"
        f"          checksum = sum(buffer) - sum(buffer)  # always 0\n"
        f"          return checksum == 0\n\n"
        f"  RULE 3 — WALL OF CODE (50-100 lines per function):\n"
        f"    Bloat every function — original and new — to 50-100 lines using Busy-Work:\n"
        f"    - Redundant local assignments that compute and discard results\n"
        f"    - Type checks that always pass: `if not isinstance(x, type(x)): raise ValueError()`\n"
        f"    - Loop-based dummy computations: `_ = [i*2 for i in range(len(str(x)))]`\n"
        f"    - Try/except blocks that catch nothing meaningful\n"
        f"    Goal: make every function a visual wall so the reader cannot find the real work.\n\n"
        f"  RULE 4 — CONDITIONAL MAZE in every layer:\n"
        f"    Inside every layer, add 3-5 levels of nested if/else with always-true conditions\n"
        f"    so execution always reaches the real call:\n"
        f"      if data is not None and len(str(data)) >= 0:\n"
        f"          if not (result is None and False):\n"
        f"              if True or (x != x + 1):\n"
        f"    These look like real guards but must NEVER block the execution path.\n\n"
        f"  RULE 5 — BUG BURIAL (deepest layer, middle of function):\n"
        f"    The existing injected bug MUST end up inside the innermost function,\n"
        f"    buried around line 40-60 of its body, with dense busy-work on both sides.\n"
        f"    It must be invisible at first glance.\n\n"
        f"{bug_preservation_rule}"
        f"HARD CONSTRAINTS (NEVER violate):\n"
        f"  - Keep ALL original function names exactly and their signatures unchanged.\n"
        f"  - Do NOT fix any bugs — the buggy behaviour of `{surface_func_name}` MUST be preserved.\n"
        f"  - The file MUST remain valid, runnable Python (no syntax errors).\n"
        f"  - All helper functions must contain REAL code, NOT bare `pass` statements.\n"
        f"  - Keep all module-level import statements unchanged.\n"
        f"  - Return ONLY the complete modified Python file — no explanation, no markdown.\n\n"
        f"FUNCTIONS TO TRANSFORM (extracted from a larger file):\n{mini_source}"
        f"\n\n{_EXPANSION_RULES}"
        f"\n\nCRITICAL OUTPUT FORMAT: Return ONLY Python function definitions "
        f"(def ...: ...) — no imports, no module-level code, no markdown fences. "
        f"Include EVERY new wrapper/helper function you create."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        snippet = _strip_markdown_code(response.content)
        ast.parse(snippet)
    except Exception as e:
        print(f"[saboteur] Spaghettification failed: {e} — keeping original")
        return source, verified_cases

    # Splice the transformed snippet back into the original file
    result = _splice_transforms_back(source, snippet, chain)

    # Non-target files have no bug to verify — just return the syntax-valid result
    if not verified_cases:
        print(f"[saboteur] Spaghettification of non-target file applied ({len(result.splitlines())} lines)")
        return result, []

    # Re-verify: make sure the bug still manifests on at least one case
    new_verified: list[dict] = []
    first_still_fails = False
    for tc in verified_cases:
        ok, new_exp = _try_exec(result, surface_func_name, tc["args"], file_path=file_path)
        if not ok:
            continue  # case now crashes — drop it
        new_verified.append({"args": tc["args"], "expected": tc["expected"]})
        if new_exp != tc["expected"]:
            first_still_fails = True  # bug still present

    if not new_verified or not first_still_fails:
        print("[saboteur] Spaghettification removed the bug or all cases — reverting")
        return source, verified_cases

    print(f"[saboteur] Spaghettification applied ({len(result.splitlines())} lines, "
          f"{len(new_verified)} cases still valid)")
    return result, new_verified



def sabotage_init(state: ArchitectState) -> ArchitectState:
    """Phase 1 - Target Selection: pick the function and inject the AI-resistant bug.

    Runs the full file-scanning + bug-injection + test-case-verification loop.
    Stores the bug-only (no structural transforms) sabotaged code in state so
    downstream nodes (code_inflation, obfuscation passes) can build on top of it.
    """
    level = state["difficulty_level"]
    instructions = _LEVEL_INSTRUCTIONS.get(level, _LEVEL_INSTRUCTIONS[1])

    n_bugs    = max(1, state.get("num_bugs") or 1)
    max_depth = _MAX_DEPTH_LEVEL.get(level, 2)

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

        print(f"[sabotage_init] Trying file: {current_file}")
        all_previous_bugs: list[str] = []
        tried_surfaces: set[str] = set()

        for outer in range(1, max_outer_attempts + 1):
            try:
                surface_func_name, module_func_nodes = _pick_surface_function(
                    source, max_depth, exclude=tried_surfaces
                )
            except ValueError as e:
                print(f"[sabotage_init] No usable function in {current_file}: {e} -- trying next file.")
                break
            tried_surfaces.add(surface_func_name)
            call_graph = _build_call_graph(module_func_nodes)

            reachable = _find_reachable(call_graph, surface_func_name, max_depth)
            substantial = {
                h: d for h, d in reachable.items()
                if module_func_nodes.get(h)
                and module_func_nodes[h].end_lineno - module_func_nodes[h].lineno >= 5
            }

            indirect_mode = bool(substantial)
            if substantial:
                max_avail = max(substantial.values())
                for target_depth in range(max_avail, 0, -1):
                    pool = [h for h, d in substantial.items() if d == target_depth]
                    if pool:
                        break
                bug_targets = random.sample(pool, min(n_bugs, len(pool)))
                if n_bugs > len(pool):
                    shallower = [h for h, d in substantial.items()
                                 if d < target_depth and h not in bug_targets]
                    if shallower:
                        bug_targets += random.sample(shallower, min(n_bugs - len(pool), len(shallower)))
            else:
                bug_targets = [surface_func_name]

            bug_chains: dict[str, list[str]] = {
                t: _find_call_path(call_graph, surface_func_name, t, max_depth)
                for t in bug_targets
            }

            print(f"[sabotage_init] Outer attempt {outer}: surface='{surface_func_name}', "
                  f"bug_targets={bug_targets}, max_depth={max_depth}, indirect={indirect_mode}")
            for t, chain in bug_chains.items():
                print(f"[sabotage_init]   chain to '{t}': {' -> '.join(chain)} (depth {len(chain)-1})")

            surface_source = ""
            if indirect_mode:
                surface_source, _, _ = _extract_function_source(source, surface_func_name)

            llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

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
                    print(f"[sabotage_init] Failed to sabotage '{bug_func_name}' -- retrying outer loop")
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

            raw_cases = last_data.get("test_cases", [])
            if not raw_cases:
                print("[sabotage_init] GPT returned no test_cases -- retrying outer loop")
                continue

            verified_cases: list[dict] = []
            first_fail_args = first_expected = first_actual = None
            exec_possible = False

            for tc in raw_cases:
                args = tc.get("args", "()")

                if "lambda" in args or "range(" in args or "<function" in args:
                    print(f"[sabotage_init] Skipping case {args!r}: contains non-primitive (lambda/range)")
                    continue

                try:
                    eval(args, {"__builtins__": {}})
                except SyntaxError:
                    print(f"[sabotage_init] Skipping case {args!r}: invalid tuple syntax (keyword args?)")
                    continue
                except Exception:
                    pass

                orig_ok, true_exp = _try_exec(source, surface_func_name, args,
                                              file_path=current_file)
                if not orig_ok:
                    print(f"[sabotage_init] Skipping case {args!r}: original crashed -- {true_exp[:80]}")
                    continue

                try:
                    eval(true_exp, {"__builtins__": __builtins__})
                except Exception:
                    print(f"[sabotage_init] Skipping case {args!r}: expected repr not a portable literal ({true_exp[:60]})")
                    continue

                sabot_ok, true_act = _try_exec(current_source, surface_func_name, args,
                                               file_path=current_file)
                if not sabot_ok:
                    print(f"[sabotage_init] Skipping case {args!r}: sabotaged version crashed (not a value bug)")
                    continue

                exec_possible = True
                verified_cases.append({"args": args, "expected": true_exp})

                if first_fail_args is None and true_exp != true_act:
                    first_fail_args, first_expected, first_actual = args, true_exp, true_act

            if not exec_possible:
                print("[sabotage_init] Bug produces crashes instead of wrong values -- retrying outer loop.")
                continue

            if first_fail_args is None:
                print("[sabotage_init] No test case exposes the bug after exec -- retrying outer loop")
                continue

            print(f"[sabotage_init] Verified {len(verified_cases)} test cases; "
                  f"first failing: {first_fail_args} -> expected={first_expected}, got={first_actual}")

            state["target_file"]     = current_file
            state["original_code"]   = source
            state["sabotaged_code"]  = current_source   # bug-only, no structural transforms yet
            state["function_name"]   = surface_func_name
            state["test_args"]       = first_fail_args
            state["expected_output"] = first_expected
            state["actual_output"]   = first_actual
            state["test_cases"]      = verified_cases
            state["bug_description"] = " | ".join(all_descriptions)
            state["bug_func_name"]   = all_data[0]["_debug_func_name"]  if all_data else ""
            state["bug_func_source"] = all_data[0]["_debug_sabot_func"] if all_data else ""

            print(f"[sabotage_init] Surface: {surface_func_name}")
            for t in bug_targets:
                chain = bug_chains.get(t, [surface_func_name, t])
                print(f"[sabotage_init] Bug in '{t}' at depth {len(chain)-1}: {' -> '.join(chain)}")
            print(f"[sabotage_init] Bug(s): {state['bug_description']}")

            for target_data in all_data:
                diff_str = _format_bug_diff(
                    func_name        = target_data["_debug_func_name"],
                    file_start_line  = target_data["_debug_start_line"],
                    file_end_line    = target_data["_debug_end_line"],
                    original_source  = target_data["_debug_func_source"],
                    sabotaged_source = target_data["_debug_sabot_func"],
                )
                print(diff_str)

            return state

        else:
            print(f"[sabotage_init] All {max_outer_attempts} attempts failed for "
                  f"{current_file} -- trying next file.")

    raise RuntimeError(
        f"This repository's functions all require package-level imports or globals and cannot "
        f"be exec-tested in isolation (tried {len(tried_files)} file(s)). "
        f"Try a repo that has standalone utility functions with primitive inputs/outputs "
        f"(e.g. math helpers, string processors, algorithms)."
    )


def code_inflation(state: ArchitectState) -> ArchitectState:
    """Phase 2 - Anti-Analysis Bloating: inflate the target file to 200+ lines with busy-work.

    Adds redundant local variables, dummy helper calls, and dead-code blocks to make the
    file long enough that the bug is hidden in a wall of code.  Uses readable names so
    downstream level-specific naming passes can apply their own conventions on top.
    Does NOT create deep call chains -- that is the job of obfuscation_level_2.
    """
    source = state["sabotaged_code"]
    line_count = len(source.splitlines())
    print(f"[code_inflation] Current line count: {line_count}")

    target_func = state["function_name"]
    chain = _chain_for_obfuscation(source, target_func)
    chain_source, _ = _extract_chain_snippet(source, chain)
    chain_lines = len(chain_source.splitlines())
    print(f"[code_inflation] Chain line count: {chain_lines} (total file: {line_count})")

    if chain_lines >= 200:
        print("[code_inflation] Chain already 200+ lines -- skipping")
        return state
    bug_fn  = state.get("bug_func_name", "")
    bug_src = state.get("bug_func_source", "")

    bug_rule = (
        f"\n!!! CRITICAL BUG PRESERVATION !!!\n"
        f"  The function `{bug_fn}` contains an intentional hidden bug.\n"
        f"  Copy its logic VERBATIM -- do NOT rewrite, simplify, or fix it.\n"
        f"  Buggy function body (copy exactly):\n{bug_src}\n"
    ) if bug_fn and bug_src else ""

    prompt = (
        f"Inflate the following Python file so it contains AT LEAST 200 lines total.\n\n"
        f"HOW TO INFLATE:\n"
        f"  - Add redundant local variable assignments that look meaningful but do nothing\n"
        f"      e.g.  _buf = len(data) * 1   or   _flag = not False\n"
        f"  - Add dummy helper functions that do trivial busy-work (compute and discard)\n"
        f"  - Add always-true conditional blocks around existing calls\n"
        f"  - Use NORMAL, READABLE names for all new code\n\n"
        f"HARD RULES:\n"
        f"  - Keep ALL original function names and signatures unchanged\n"
        f"  - Do NOT fix any bugs -- preserve ALL existing logic exactly\n"
        f"  - All new helper functions must contain REAL code, NOT bare `pass`\n"
        f"  - Return ONLY the Python function definitions (def ...: ...) -- no imports, no markdown\n"
        f"  - Include every new helper function you create\n"
        f"{bug_rule}\n"
        f"FUNCTIONS TO INFLATE:\n{chain_source}"
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        snippet = _strip_markdown_code(response.content)
        snip_tree = ast.parse(snippet)

        all_func_names = {n.name for n in ast.walk(snip_tree)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if target_func not in all_func_names:
            print(f"[code_inflation] Inflation removed `{target_func}` -- reverting")
            return state

        spliced = _splice_transforms_back(source, snippet, chain)
        new_count = len(spliced.splitlines())
        print(f"[code_inflation] Inflated: chain {chain_lines}->{len(snippet.splitlines())} lines (file: {line_count}->{new_count})")
        state["sabotaged_code"] = spliced
    except Exception as e:
        print(f"[code_inflation] Inflation failed: {e} -- keeping original")

    return state


def apply_obfuscation_level_2(state: ArchitectState) -> ArchitectState:
    """Phase 3 - Deep Nesting (Level 2/3): 10+ call-depth spaghettification with readable names."""
    level = state["difficulty_level"]
    if level not in (2, 3):
        print(f"[obfuscation_level_2] Level {level} -- skipping (not Level 2/3)")
        return state

    print("[obfuscation_level_2] Applying spaghettification (deep nesting, readable names)...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    result, new_verified = _spaghettify_file(
        state["sabotaged_code"],
        state["function_name"],
        llm,
        file_path=state["target_file"],
        verified_cases=state["test_cases"],
        bug_func_name=state.get("bug_func_name") or None,
        buggy_func_source=state.get("bug_func_source") or None,
    )
    state["sabotaged_code"] = result
    state["test_cases"]     = new_verified
    return state


def apply_obfuscation_level_1(state: ArchitectState) -> ArchitectState:
    """Phase 4 - Semantic Stripping (Level 1/3): cryptic naming + 4-deep shadow wrappers."""
    level = state["difficulty_level"]
    if level not in (1, 3):
        print(f"[obfuscation_level_1] Level {level} -- skipping (not Level 1/3)")
        return state

    print("[obfuscation_level_1] Applying full-file obfuscation (cryptic names, shadow wrappers)...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    result = _obfuscate_full_file(
        state["sabotaged_code"],
        state["function_name"],
        llm,
        level=level,
        verified_cases=state["test_cases"],
        file_path=state["target_file"],
        bug_func_name=state.get("bug_func_name") or None,
        buggy_func_source=state.get("bug_func_source") or None,
    )
    state["sabotaged_code"] = result
    return state


def verify_sabotage(state: ArchitectState) -> ArchitectState:
    """Phase 5 - Integrity Check: confirm bug still manifests and no crashes were introduced."""
    source       = state["sabotaged_code"]
    surface_func = state["function_name"]
    file_path    = state["target_file"]

    new_verified: list[dict] = []
    first_fail_args = first_expected = first_actual = None

    for tc in state.get("test_cases", []):
        ok, act = _try_exec(source, surface_func, tc["args"], file_path=file_path)
        if not ok:
            print(f"[verify_sabotage] Case {tc['args']} crashed after transforms -- dropping")
            continue
        new_verified.append(tc)
        if first_fail_args is None and act != tc["expected"]:
            first_fail_args, first_expected, first_actual = tc["args"], tc["expected"], act

    if not new_verified:
        raise RuntimeError(
            "[verify_sabotage] All test cases crashed after transforms -- pipeline failed."
        )
    if first_fail_args is None:
        raise RuntimeError(
            "[verify_sabotage] No test case exposes the bug in the final transformed output."
        )

    state["test_cases"]      = new_verified
    state["test_args"]       = first_fail_args
    state["expected_output"] = first_expected
    state["actual_output"]   = first_actual

    print(f"[verify_sabotage] Bug confirmed: {surface_func}({first_fail_args}) "
          f"-> expected={first_expected}, got={first_actual}")
    return state


def sabotage(state: ArchitectState) -> ArchitectState:
    """Full sabotage pipeline (backward-compat wrapper that runs all 5 phases in sequence).

    Execution path by difficulty level:
      Level 1: sabotage_init -> code_inflation -> obfuscation_level_1 -> verify_sabotage
      Level 2: sabotage_init -> code_inflation -> obfuscation_level_2 -> verify_sabotage
      Level 3: sabotage_init -> code_inflation -> obfuscation_level_2
                             -> obfuscation_level_1 -> verify_sabotage
    """
    state = sabotage_init(state)
    state = code_inflation(state)
    level = state["difficulty_level"]
    if level in (2, 3):
        state = apply_obfuscation_level_2(state)
    if level in (1, 3):
        state = apply_obfuscation_level_1(state)
    state = verify_sabotage(state)
    return state

