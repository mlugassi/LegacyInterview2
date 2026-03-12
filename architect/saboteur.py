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
  "test_cases_public": [
    {"args": "<Python literal tuple of args, e.g. (10, 3) or ('hello',)>", "correct_output": "<correct return value for the ORIGINAL function>"},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."}
  ],
  "test_cases_secret": [
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."}
  ],
  "bug_description": "<one sentence describing the FUNCTIONAL behavior that is now broken and why>"
}

CRITICAL TEST CASE REQUIREMENTS:

PUBLIC TESTS (test_cases_public) - 5 SIMPLER TESTS FOR DEVELOPMENT:
- These help the student understand the basic problem
- Cover normal/common use cases
- 2-3 should PASS (function works for simple inputs where bug doesn't trigger)
- 2-3 should FAIL (exposing the bug in obvious cases)
- Examples: typical inputs, small numbers, simple strings
- Goal: Guide the student toward understanding what's broken
- IMPORTANT: At least 2 tests MUST expose the injected bug!

SECRET TESTS (test_cases_secret) - 5 HARDER TESTS FOR FINAL VALIDATION:
- These tests MUST be significantly harder and more precise
- At least 3-4 should expose the bug in subtle/edge-case scenarios
- Focus on boundary conditions, edge cases, corner cases where the bug manifests
- Examples: empty inputs, None, zero, negative numbers, very large values, special characters
- These tests should FAIL even if the student makes a partial/incomplete fix
- Goal: Ensure the fix is complete and handles all edge cases correctly
- CRITICAL: Design these tests specifically to trigger the bug you injected!

TEST COVERAGE REQUIREMENT:
- Your tests MUST actually detect the bug you injected
- At least 30-40% of all tests (public + secret combined) should FAIL on the sabotaged code
- If your bug is "off-by-one in loop", create tests with edge cases that expose it
- If your bug is "wrong operator", create tests where that operator matters
- Don't create generic tests that happen to pass despite the bug!

IMPORTANT: If all 5 public tests pass, it should NOT guarantee that all 5 secret tests pass!
The secret tests must catch incomplete fixes that work for simple cases but fail on edges.

Critical rules:
- sabotaged_function_code must be ONLY the function definition (def ...: ...) — nothing before or after.
- The `def` line MUST keep the EXACT SAME function name as the original. Do NOT rename the function.
- Preserve the EXACT original indentation level of the function (copy it from the input).
- Do NOT modify any docstrings or string literals that already exist in the function.
- CRITICAL: Do NOT add ANY comments to the code! No # comments, no explanatory notes, nothing!
  Follow ONLY the steps listed in the SABOTAGE INSTRUCTIONS — do not add renaming or comments
  unless the instructions explicitly ask for them (they never will).
- The bug MUST produce a WRONG RETURN VALUE — NOT a crash or exception.
  The sabotaged function must still run without raising exceptions; it just returns the wrong result.
  If your bug causes a TypeError, AttributeError, or any other exception, it will be REJECTED.
  Change a condition, a boundary, an operator, or a value — not the structure so drastically it crashes.
- Each "args" must be a Python tuple literal using ONLY these primitives: int, float, str, list, dict, bool.
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
    Find all reachable functions from start with their MAXIMUM depth.
    Returns {func_name: max_depth} for every reachable function (excluding start).
    Uses DFS to explore all paths and keep the longest depth for each function.
    Only includes functions that exist as keys in call_graph.
    """
    visited: dict[str, int] = {}
    
    def dfs(current: str, depth: int, seen: set[str]):
        if depth >= max_depth:
            return
        
        for callee in call_graph.get(current, set()):
            if callee not in call_graph:
                continue
            
            # Update with maximum depth found
            if callee not in visited or visited[callee] < depth + 1:
                visited[callee] = depth + 1
            
            # Continue exploring if we haven't been through this exact path
            if callee not in seen:
                seen.add(callee)
                dfs(callee, depth + 1, seen)
                seen.remove(callee)
    
    dfs(start, 0, {start})
    return visited


def _find_call_path(call_graph: dict, start: str, target: str, max_depth: int) -> list[str]:
    """
    Find the LONGEST call path from start to target (up to max_depth).
    This ensures we get chains matching the requested nesting level.
    Uses DFS to explore all paths and returns the longest one found.
    """
    if start == target:
        return [start]
    
    longest_path: list[str] = []
    
    def dfs(current: str, path: list[str], visited: set[str]):
        nonlocal longest_path
        
        if len(path) > max_depth + 1:
            return
        
        if current == target:
            if len(path) > len(longest_path):
                longest_path = path.copy()
            return
        
        for callee in call_graph.get(current, set()):
            if callee not in visited:
                visited.add(callee)
                path.append(callee)
                dfs(callee, path, visited)
                path.pop()
                visited.remove(callee)
    
    dfs(start, [start], {start})
    
    # If no path found, return direct connection as fallback
    return longest_path if longest_path else [start, target]


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
    # Heavily prefer functions that can reach targets at depths close to max_depth
    caller_candidates: list[tuple[int, str]] = []
    for name in module_func_nodes:
        reachable = _find_reachable(call_graph, name, max_depth)
        substantial = {
            h: d for h, d in reachable.items()
            if module_func_nodes[h].end_lineno - module_func_nodes[h].lineno >= 5
        }
        if not substantial:
            continue
        
        # Compute score with strong preference for deep chains
        score = 0
        max_reachable_depth = max(substantial.values()) if substantial else 0
        
        # Bonus for having functions at good depths (prefer depth >= max_depth - 1)
        for d in substantial.values():
            if d >= max_depth - 1:
                score += d * 100  # Very strong bonus for deep chains
            elif d >= max_depth - 2:
                score += d * 50   # Good bonus for moderately deep chains
            else:
                score += d * 10   # Small bonus for shallow chains
        
        # Additional bonus for maximum depth reached
        if max_reachable_depth >= max_depth - 1:
            score += 200
        
        # Function size contributes minimally
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
        f"(file lines {file_start_line + 1}–{file_end_line} - BEFORE inflation)\n"
        f"  │\n"
        f"  │  NOTE: Line numbers shown are BEFORE function inflation.\n"
        f"  │        In the final file, functions will be expanded (50+ lines each).\n"
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
    target_nesting: int = 3,
    debug_mode: bool = False,
) -> tuple[str, dict] | tuple[None, None]:
    """
    Ask GPT to sabotage bug_func_name within current_source.
    call_chain is the path from surface_func_name down to bug_func_name.
    previous_bugs is a list of bug_description strings from prior failed attempts.
    target_nesting indicates the desired call-chain depth.
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
        if debug_mode:
            print(f"[saboteur] GPT call (func='{bug_func_name}', attempt={attempt})…")
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        try:
            data = _parse_response(response.content)
        except json.JSONDecodeError as e:
            if debug_mode:
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
            if debug_mode:
                print(f"[saboteur] Syntax error: {e}")
            continue

        # Verify function name preserved
        new_names = {
            n.name for n in new_tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if bug_func_name not in new_names:
            if debug_mode:
                print(f"[saboteur] GPT renamed '{bug_func_name}' — retrying")
            continue

        # Reject if any new comment reveals the bug
        if _has_revealing_comment(func_source, sabotaged_func):
            if debug_mode:
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


def _augment_chain_depth(source: str, current_chain: list[str], target_depth: int, 
                         module_func_nodes: dict) -> tuple[str, list[str]]:
    """
    Add intermediate wrapper functions to extend a call chain to target_depth.
    
    If current_chain has depth < target_depth, this function creates new intermediate
    functions and inserts them at random positions in the chain to reach the target depth.
    
    Returns: (updated_source, augmented_chain)
    """
    current_depth = len(current_chain) - 1
    if current_depth >= target_depth:
        return source, current_chain
    
    needed_functions = target_depth - current_depth
    if needed_functions <= 0:
        return source, current_chain
    
    # Generate unique names for intermediate functions
    existing_names = set(module_func_nodes.keys())
    intermediate_funcs = []
    for i in range(needed_functions):
        base_name = f"_intermediate_processor_{i+1}"
        counter = 0
        func_name = base_name
        while func_name in existing_names:
            counter += 1
            func_name = f"{base_name}_{counter}"
        intermediate_funcs.append(func_name)
        existing_names.add(func_name)
    
    # Build augmented chain by inserting intermediates at various positions
    # Prefer inserting near the middle/end to make debugging harder
    augmented = current_chain.copy()
    insertion_positions = []
    
    # Choose random positions (avoiding position 0 since that's the surface function)
    available_positions = list(range(1, len(augmented)))
    if not available_positions:
        available_positions = [1]  # If chain has only 2 elements
    
    for func_name in intermediate_funcs:
        if available_positions:
            pos = random.choice(available_positions)
        else:
            pos = len(augmented) - 1
        insertion_positions.append((pos, func_name))
    
    # Sort by position (descending) to insert from end to avoid index shifting
    insertion_positions.sort(reverse=True, key=lambda x: x[0])
    
    for pos, func_name in insertion_positions:
        augmented.insert(pos, func_name)
    
    # Now generate the actual function code for each intermediate
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    
    # Find a good insertion point (after imports, before first function)
    first_func_line = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            first_func_line = node.lineno - 1
            break
    
    if first_func_line is None:
        first_func_line = len(lines)
    
    # Generate intermediate function definitions
    new_function_defs = []
    
    for idx, func_name in enumerate(intermediate_funcs):
        # Find what this function should call (the next in the augmented chain)
        func_pos = augmented.index(func_name)
        next_func = augmented[func_pos + 1]
        
        # Get signature info from the next function in chain
        if next_func in module_func_nodes:
            next_node = module_func_nodes[next_func]
            # Extract parameter names
            params = [arg.arg for arg in next_node.args.args]
            param_str = ", ".join(params) if params else "*args, **kwargs"
            call_str = f"{next_func}({', '.join(params)})" if params else f"{next_func}(*args, **kwargs)"
        else:
            param_str = "*args, **kwargs"
            call_str = f"{next_func}(*args, **kwargs)"
        
        # Create an expanded intermediate function (following expansion rules)
        func_code = f'''
def {func_name}({param_str}):
    """Intermediate processing layer {idx+1}."""
    import random as _rand_module
    _buffer_marker = len(str({params[0] if params else 'args'})) if {bool(params)} else 0
    _tmp_state = _buffer_marker * 1
    _validation_flag = True if _tmp_state >= 0 else False
    _entropy_counter = sum(i * 0 for i in range(5))
    _checksum_val = _entropy_counter + _buffer_marker - _buffer_marker
    _ref_copy = {params[0] if params else 'None'}
    _dummy_list = [x - x for x in range(4)]
    _dead_sum = sum(_dummy_list)
    
    if _validation_flag and _checksum_val == 0:
        _nested_check = _tmp_state >= 0 and _tmp_state is not None
        if _nested_check:
            _marker_a = len(_dummy_list) * 1
            _marker_b = _marker_a - _marker_a
            if _marker_b == 0:
                _result_ref = {call_str}
                _post_marker = len(str(_result_ref)) - len(str(_result_ref))
                if _post_marker == 0:
                    return _result_ref
    
    # Fallback path (never reached but looks plausible)
    return {call_str}
'''
        new_function_defs.append(func_code)
    
    # Insert the new functions
    insertion_block = "\n".join(new_function_defs) + "\n\n"
    lines.insert(first_func_line, insertion_block)
    
    updated_source = "".join(lines)
    
    return updated_source, augmented


def sabotage_init(state: ArchitectState) -> ArchitectState:
    """Phase 1 - Target Selection: pick the function and inject the AI-resistant bug.

    Runs the full file-scanning + bug-injection + test-case-verification loop.
    Stores the bug-only (no structural transforms) sabotaged code in state so
    downstream nodes (inflate_hierarchy, obfuscation passes) can build on top of it.
    """
    # Use nesting_level instead of difficulty_level
    target_nesting = state["nesting_level"]
    debug_mode = state.get("debug_mode", False)
    instructions = _BUG_INJECTION_INSTRUCTION  # Same instruction for all

    n_bugs    = max(1, state.get("num_bugs") or 1)
    max_depth = target_nesting  # Use the requested nesting level

    candidate_files: list[str] = list(state.get("candidate_files") or [state["target_file"]])
    tried_files: set[str] = set()

    max_outer_attempts = 5  # Increased from 3 to give more chances
    total_attempts = 0
    max_total_attempts = 15  # Safety limit to prevent infinite loops
    
    while candidate_files and total_attempts < max_total_attempts:
        current_file = candidate_files.pop(0)
        if current_file in tried_files:
            continue
        tried_files.add(current_file)

        with open(current_file, encoding="utf-8", errors="ignore") as _f:
            source = _f.read()

        if debug_mode:
            print(f"[sabotage_init] Trying file: {current_file}")
        all_previous_bugs: list[str] = []
        tried_surfaces: set[str] = set()

        for outer in range(1, max_outer_attempts + 1):
            total_attempts += 1
            if total_attempts > max_total_attempts:
                if debug_mode:
                    print(f"[sabotage_init] Reached maximum total attempts ({max_total_attempts}) -- giving up")
                break
                
            try:
                surface_func_name, module_func_nodes = _pick_surface_function(
                    source, max_depth, exclude=tried_surfaces
                )
            except ValueError as e:
                print(f"[sabotage_init] No usable function in {current_file}: {e} -- trying next file.")
                break
            tried_surfaces.add(surface_func_name)
            call_graph = _build_call_graph(module_func_nodes)

            # Find reachable functions (those in the call graph)
            reachable = _find_reachable(call_graph, surface_func_name, max_depth)
            
            # Important: Include ALL functions that are large enough, not just reachable ones
            # Because we can use _augment_chain_depth to add wrapper functions later!
            # This allows us to inject bugs into ANY function and then build the required nesting
            all_candidates = {
                h: reachable.get(h, 0)  # Use existing depth if reachable, otherwise 0 (direct call)
                for h, node in module_func_nodes.items()
                if h != surface_func_name  # Don't include the surface function itself
                and node.end_lineno - node.lineno >= 3  # Minimum 3 lines
            }
            
            # substantial now includes ALL functions, even direct ones (depth 0)
            substantial = all_candidates

            indirect_mode = bool(substantial)
            if substantial:
                max_avail = max(substantial.values()) if substantial.values() else 0
                
                # NEW DYNAMIC DEPTH STRATEGY:
                # We pick functions from ANY depth (including 0 = direct call)
                # and use _augment_chain_depth to build the required nesting level.
                # This maximizes the number of available functions for bug injection!
                
                # Collect candidates from all depths (including depth 0)
                depth_pools = {}
                for h, d in substantial.items():
                    if d not in depth_pools:
                        depth_pools[d] = []
                    depth_pools[d].append(h)
                
                # Weighted selection: prefer deeper chains but allow shallower ones too
                # 60% chance for deep chains (>= max_depth-1), 40% for shallower (including depth 0)
                use_deep = random.random() < 0.6
                
                if debug_mode:
                    print(f"[sabotage_init] Number of substantial functions available: {len(substantial)}")
                    print(f"[sabotage_init] Depth distribution: {dict((d, len(funcs)) for d, funcs in depth_pools.items())}")
                    print(f"[sabotage_init] Requested bugs: {n_bugs}, Selection mode: {'deep' if use_deep else 'shallow'}")
                
                bug_targets = []
                if use_deep and max_avail >= max_depth - 1:
                    # Prefer deep chains - collect from multiple depths if needed
                    for target_depth in range(max_avail, max(1, max_depth - 2), -1):
                        if len(bug_targets) >= n_bugs:
                            break  # We have enough bugs
                        
                        if target_depth in depth_pools:
                            pool = depth_pools[target_depth]
                            verified_pool = []
                            for candidate in pool:
                                test_path = _find_call_path(call_graph, surface_func_name, candidate, max_depth)
                                if len(test_path) - 1 == target_depth:
                                    verified_pool.append((candidate, test_path))
                            
                            if verified_pool:
                                # Take as many as we still need
                                needed = n_bugs - len(bug_targets)
                                selected = random.sample(verified_pool, min(needed, len(verified_pool)))
                                bug_targets.extend([(cand, path) for cand, path in selected])
                else:
                    # Pick from shallower depths and augment them - collect from multiple depths
                    # Now includes depth 0 (direct functions) since we can augment any function!
                    candidate_depths = sorted(depth_pools.keys())
                    for target_depth in candidate_depths:
                        if len(bug_targets) >= n_bugs:
                            break  # We have enough bugs
                        
                        # Accept ALL depths (including 0) - augmentation will handle creating the nesting
                        pool = depth_pools[target_depth]
                        if pool:
                            needed = n_bugs - len(bug_targets)
                            selected_funcs = random.sample(pool, min(needed, len(pool)))
                            for func in selected_funcs:
                                # Try to find an existing path, or create a direct one
                                try:
                                    path = _find_call_path(call_graph, surface_func_name, func, max_depth)
                                except:
                                    # If no path found, create a direct path (depth 0)
                                    path = [surface_func_name, func]
                                bug_targets.append((func, path))
                            # Continue to next depth if we still need more
                
                # If no targets selected, fall back to any substantial function
                if not bug_targets:
                    pool = list(substantial.keys())
                    selected_funcs = random.sample(pool, min(n_bugs, len(pool)))
                    if debug_mode:
                        print(f"[sabotage_init] Fallback: selecting {len(selected_funcs)} bugs from {len(pool)} substantial functions")
                    bug_targets = []
                    for func in selected_funcs:
                        try:
                            path = _find_call_path(call_graph, surface_func_name, func, max_depth)
                        except:
                            # If no path exists, create a direct path - augmentation will add the nesting
                            path = [surface_func_name, func]
                        bug_targets.append((func, path))
                
                # If we still don't have enough bugs, try to get more from any available depth
                if len(bug_targets) < n_bugs and len(bug_targets) > 0:
                    if debug_mode:
                        print(f"[sabotage_init] Only found {len(bug_targets)}/{n_bugs} bugs, trying to add more from other depths...")
                    
                    # Get all functions not already selected
                    already_selected = {func for func, _ in bug_targets}
                    remaining_pool = [f for f in substantial.keys() if f not in already_selected]
                    
                    if remaining_pool:
                        needed = n_bugs - len(bug_targets)
                        additional = random.sample(remaining_pool, min(needed, len(remaining_pool)))
                        for func in additional:
                            try:
                                path = _find_call_path(call_graph, surface_func_name, func, max_depth)
                            except:
                                # If no path exists, create a direct path
                                path = [surface_func_name, func]
                            bug_targets.append((func, path))
                        
                        if debug_mode:
                            print(f"[sabotage_init] Added {len(additional)} more bugs from remaining pool, total now: {len(bug_targets)}")
                
                # CRITICAL: If still not enough bugs, inject multiple bugs in the same functions
                # This is actually MORE challenging for students!
                if len(bug_targets) < n_bugs and len(bug_targets) > 0:
                    needed = n_bugs - len(bug_targets)
                    if debug_mode:
                        print(f"[sabotage_init] Still need {needed} more bug(s). Will inject multiple bugs into existing functions.")
                    
                    # Duplicate existing targets to reach the required number
                    # Prefer functions with higher depth (more interesting)
                    sorted_targets = sorted(bug_targets, key=lambda x: len(x[1]), reverse=True)
                    for i in range(needed):
                        # Cycle through targets to distribute bugs evenly
                        target_to_duplicate = sorted_targets[i % len(sorted_targets)]
                        bug_targets.append(target_to_duplicate)
                        if debug_mode:
                            func_name = target_to_duplicate[0]
                            print(f"[sabotage_init] Adding additional bug to function '{func_name}'")
                
                if debug_mode:
                    print(f"[sabotage_init] FINAL: Selected {len(bug_targets)} bug injection(s) (requested {n_bugs})")
                    unique_funcs = len(set(func for func, _ in bug_targets))
                    if unique_funcs < len(bug_targets):
                        print(f"[sabotage_init] Note: {len(bug_targets)} bugs distributed across {unique_funcs} functions (multiple bugs per function)")
                
                # Now augment chains that are too short
                augmented_targets = []
                for bug_func, original_chain in bug_targets:
                    current_depth = len(original_chain) - 1
                    if current_depth < max_depth:
                        # Augment this chain!
                        if debug_mode:
                            print(f"[sabotage_init] Augmenting chain to '{bug_func}' from depth {current_depth} to {max_depth}")
                        try:
                            source, augmented_chain = _augment_chain_depth(
                                source, original_chain, max_depth, module_func_nodes
                            )
                            # Re-parse to update module_func_nodes
                            tree = ast.parse(source)
                            for node in tree.body:
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    if node.name not in module_func_nodes:
                                        module_func_nodes[node.name] = node
                            # Rebuild call graph with new functions
                            call_graph = _build_call_graph(module_func_nodes)
                            augmented_targets.append((bug_func, augmented_chain))
                            if debug_mode:
                                print(f"[sabotage_init]   Augmented chain: {' -> '.join(augmented_chain)}")
                        except Exception as e:
                            if debug_mode:
                                print(f"[sabotage_init]   Augmentation failed: {e}, using original chain")
                            augmented_targets.append((bug_func, original_chain))
                    else:
                        augmented_targets.append((bug_func, original_chain))
                
                # Extract just the function names and chains
                bug_targets = [func for func, _ in augmented_targets]
                bug_chains: dict[str, list[str]] = {
                    func: chain for func, chain in augmented_targets
                }
            else:
                bug_targets = [surface_func_name]
                bug_chains: dict[str, list[str]] = {
                    surface_func_name: [surface_func_name]
                }

            if debug_mode:
                print(f"[sabotage_init] Outer attempt {outer}: surface='{surface_func_name}', "
                      f"bug_targets={bug_targets}, max_depth={max_depth}, indirect={indirect_mode}")
                print(f"[sabotage_init] Number of bugs requested: {n_bugs}, Number of targets selected: {len(bug_targets)}")
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

            if debug_mode:
                print(f"[sabotage_init] Starting bug injection for {len(bug_targets_sorted)} bugs")

            for bug_func_name in bug_targets_sorted:
                new_source, data = _sabotage_one_helper(
                    bug_func_name, current_source,
                    surface_func_name, surface_source,
                    instructions, llm, indirect_mode,
                    call_chain=bug_chains.get(bug_func_name),
                    previous_bugs=list(all_previous_bugs),
                    target_nesting=target_nesting,
                    debug_mode=debug_mode,
                )
                if new_source is None:
                    if debug_mode:
                        print(f"[sabotage_init] Failed to sabotage '{bug_func_name}' -- skipping this bug and continuing")
                    continue  # Skip this bug, try next one
                
                current_source = new_source
                all_descriptions.append(data["bug_description"])
                all_data.append(data)
                last_data = data
                if debug_mode:
                    print(f"[sabotage_init] Successfully injected bug #{len(all_data)} into '{bug_func_name}'")
                if desc := data.get("bug_description", ""):
                    all_previous_bugs.append(desc)

            # Check if we successfully injected at least one bug
            if not all_data or last_data is None:
                if debug_mode:
                    print(f"[sabotage_init] All {len(bug_targets_sorted)} bug injections failed -- retrying outer loop")
                continue

            # MERGE test cases from ALL bugs (not just last one!)
            # This ensures every bug is covered by its own tests
            raw_public_cases = []
            raw_secret_cases = []
            
            for bug_data in all_data:
                # Get test cases from this bug
                bug_public = bug_data.get("test_cases_public", [])
                bug_secret = bug_data.get("test_cases_secret", [])
                
                # Fallback: if old format, split the test_cases
                if not bug_public and not bug_secret:
                    raw_all = bug_data.get("test_cases", [])
                    mid = len(raw_all) // 2
                    bug_public = raw_all[:mid] if mid > 0 else raw_all
                    bug_secret = raw_all[mid:] if mid > 0 else []
                
                raw_public_cases.extend(bug_public)
                raw_secret_cases.extend(bug_secret)
            
            if not raw_public_cases and not raw_secret_cases:
                if debug_mode:
                    print("[sabotage_init] GPT returned no test cases from any bug -- retrying outer loop")
                continue
            
            if debug_mode:
                print(f"[sabotage_init] Collected {len(raw_public_cases)} public + {len(raw_secret_cases)} secret test cases from {len(all_data)} bug(s)")

            # Verify public tests (simpler, for development)
            verified_public: list[dict] = []
            public_failing_count = 0
            for tc in raw_public_cases:
                args = tc.get("args", "()")
                if "lambda" in args or "range(" in args or "<function" in args:
                    continue
                try:
                    eval(args, {"__builtins__": {}})
                except:
                    continue

                orig_ok, true_exp = _try_exec(source, surface_func_name, args, file_path=current_file)
                if not orig_ok:
                    continue
                try:
                    eval(true_exp, {"__builtins__": __builtins__})
                except:
                    continue

                sabot_ok, true_act = _try_exec(current_source, surface_func_name, args, file_path=current_file)
                if not sabot_ok:
                    continue

                verified_public.append({"args": args, "expected": true_exp})
                if true_exp != true_act:
                    public_failing_count += 1
            
            # Verify secret tests (harder, for final validation)  
            verified_secret: list[dict] = []
            secret_failing_count = 0
            first_fail_args = first_expected = first_actual = None
            exec_possible = False
            
            for tc in raw_secret_cases:
                args = tc.get("args", "()")
                if "lambda" in args or "range(" in args or "<function" in args:
                    continue
                try:
                    eval(args, {"__builtins__": {}})
                except:
                    continue
                
                orig_ok, true_exp = _try_exec(source, surface_func_name, args, file_path=current_file)
                if not orig_ok:
                    continue
                try:
                    eval(true_exp, {"__builtins__": __builtins__})
                except:
                    continue
                
                sabot_ok, true_act = _try_exec(current_source, surface_func_name, args, file_path=current_file)
                if not sabot_ok:
                    continue
                
                exec_possible = True
                verified_secret.append({"args": args, "expected": true_exp})
                
                if true_exp != true_act:
                    secret_failing_count += 1
                    if first_fail_args is None:
                        first_fail_args, first_expected, first_actual = args, true_exp, true_act
            
            # Also check public tests for failures
            if first_fail_args is None:
                for tc in verified_public:
                    _, act = _try_exec(current_source, surface_func_name, tc["args"], file_path=current_file)
                    if tc["expected"] != act:
                        first_fail_args = tc["args"]
                        first_expected = tc["expected"]
                        first_actual = act
                        break

            # Combine for total verification
            verified_cases = verified_public + verified_secret

            if not exec_possible and not verified_public:
                if debug_mode:
                    print("[sabotage_init] Bug produces crashes instead of wrong values -- retrying outer loop.")
                continue

            if first_fail_args is None:
                if debug_mode:
                    print("[sabotage_init] No test case exposes the bug after exec -- retrying outer loop")
                continue

            if public_failing_count < 2:
                if debug_mode:
                    print(f"[sabotage_init] Only {public_failing_count} public tests fail on sabotaged code "
                          f"(need >= 2) -- retrying outer loop")
                continue

            # Ensure enough secret tests fail (at least 2 out of the secret tests)
            if secret_failing_count < 2:
                if debug_mode:
                    print(f"[sabotage_init] Only {secret_failing_count} secret tests fail on sabotaged code "
                          f"(need >= 2) -- retrying outer loop")
                continue

            # Ensure overall failure rate is significant (at least 30% of all tests should fail)
            total_failing = public_failing_count + secret_failing_count
            total_tests = len(verified_public) + len(verified_secret)
            failure_rate = total_failing / total_tests if total_tests > 0 else 0
            
            if failure_rate < 0.30:
                if debug_mode:
                    print(f"[sabotage_init] Only {total_failing}/{total_tests} tests fail ({failure_rate:.1%}) "
                          f"(need >= 30%) -- retrying outer loop")
                continue

            if debug_mode:
                print(f"[sabotage_init] Verified {len(verified_public)} public tests ({public_failing_count} fail) and "
                      f"{len(verified_secret)} secret tests ({secret_failing_count} fail); "
                      f"overall failure rate: {failure_rate:.1%}")
                print(f"[sabotage_init] First failing: {first_fail_args} -> expected={first_expected}, got={first_actual}")

            # Use the public/secret split from AI generation
            public_tests = verified_public
            secret_tests = verified_secret
            
            if debug_mode:
                print(f"[sabotage_init] Using AI-generated split: {len(public_tests)} public tests (simpler), "
                      f"{len(secret_tests)} secret tests (harder edge cases)")

            state["target_file"]     = current_file
            state["original_code"]   = source
            state["sabotaged_code"]  = current_source   # bug-only, no structural transforms yet
            state["function_name"]   = surface_func_name
            state["test_args"]       = first_fail_args
            state["expected_output"] = first_expected
            state["actual_output"]   = first_actual
            state["test_cases"]      = verified_cases  # All test cases
            state["public_tests"]    = public_tests     # First 5 (for students to see)
            state["secret_tests"]    = secret_tests     # Last 5 (hidden until final submission)
            state["bug_description"] = " | ".join(all_descriptions)
            state["bug_func_name"]          = all_data[0]["_debug_func_name"]   if all_data else ""
            state["bug_func_source"]        = all_data[0]["_debug_sabot_func"]  if all_data else ""
            state["original_bug_func_source"] = all_data[0]["_debug_func_source"] if all_data else ""
            state["all_bug_data"]           = all_data  # Store all bug data for misleading comments
            
            # Store the call chain for debugging and documentation
            # Format: {bug_function_name: [surface_func, ..., bug_func]}
            state["call_chain"] = bug_chains

            if debug_mode:
                print(f"\n[sabotage_init] Surface: {surface_func_name}")
                print(f"[sabotage_init] Bug(s): {state['bug_description']}")
                print("\n" + "=" * 70)
                print("CALL CHAIN VISUALIZATION (for debugging)")
                print("=" * 70)
                for t in bug_targets:
                    chain = bug_chains.get(t, [surface_func_name, t])
                    depth = len(chain) - 1
                    print(f"\nBug Location: {t} (Depth: {depth})")
                    print("Call Path:")
                    for i, func in enumerate(chain):
                        indent = "  " * i
                        if i == 0:
                            marker = "┌─"
                        elif i == len(chain) - 1:
                            marker = "└─"
                        else:
                            marker = "├─"
                        
                        if i == len(chain) - 1:
                            print(f"{indent}{marker}> {func}  ⚠️ BUG HERE")
                        else:
                            print(f"{indent}{marker}> {func}")
                            if i < len(chain) - 1:
                                print(f"{indent}│")
                print("\n" + "=" * 70 + "\n")
            
            # Generate detailed explanation for instructor
            detailed_parts = []
            detailed_parts.append("=" * 80)
            detailed_parts.append("DETAILED EXPLANATION FOR INSTRUCTOR")
            detailed_parts.append("=" * 80)
            detailed_parts.append(f"\nTarget File: {current_file}")
            detailed_parts.append(f"Surface Function: {surface_func_name}")
            detailed_parts.append(f"Number of Bugs: {len(bug_targets)}")
            detailed_parts.append(f"\nBug Locations:")
            for t in bug_targets:
                chain = bug_chains.get(t, [surface_func_name, t])
                detailed_parts.append(f"  - Function: {t}")
                detailed_parts.append(f"    Call Chain: {' -> '.join(chain)}")
                detailed_parts.append(f"    Depth: {len(chain) - 1}")
            detailed_parts.append(f"\nBug Description: {state['bug_description']}")
            detailed_parts.append(f"\nFirst Failing Test:")
            detailed_parts.append(f"  Args: {first_fail_args}")
            detailed_parts.append(f"  Expected: {first_expected}")
            detailed_parts.append(f"  Actual: {first_actual}")
            detailed_parts.append(f"\nTest Case Summary:")
            detailed_parts.append(f"  Total verified: {len(verified_cases)}")
            detailed_parts.append(f"  Public tests: {len(public_tests)}")
            detailed_parts.append(f"  Secret tests: {len(secret_tests)}")
            detailed_parts.append("\n" + "=" * 80)
            state["detailed_explanation"] = "\n".join(detailed_parts)

            # Print detailed diff only in debug mode
            if state.get("debug_mode", False):
                print(f"\n[sabotage_init] Displaying diffs for {len(all_data)} bug(s):")
                for idx, target_data in enumerate(all_data, 1):
                    print(f"\n[sabotage_init] === Bug #{idx}/{len(all_data)} ===")
                    diff_str = _format_bug_diff(
                        func_name        = target_data["_debug_func_name"],
                        file_start_line  = target_data["_debug_start_line"],
                        file_end_line    = target_data["_debug_end_line"],
                        original_source  = target_data["_debug_func_source"],
                        sabotaged_source = target_data["_debug_sabot_func"],
                    )
                    print(diff_str)

            # Success! We have at least one bug injected
            # Note: We might have fewer bugs than requested if some injections failed,
            # but that's OK - we tried our best with this file
            if debug_mode and len(all_data) < n_bugs:
                print(f"\n[sabotage_init] Note: Successfully injected {len(all_data)}/{n_bugs} bugs")
                print(f"[sabotage_init] (Some bug injections may have failed verification)")
            
            return state

        else:
            print(f"[sabotage_init] All {max_outer_attempts} attempts failed for "
                  f"{current_file} -- trying next file.")

    # If we get here, no file worked
    error_msg = (
        f"Failed to inject bug after {total_attempts} attempts across {len(tried_files)} file(s). "
        f"Possible reasons:\n"
        f"  - Functions require complex imports/globals that cannot be exec-tested in isolation\n"
        f"  - Bug injection failed to produce tests with sufficient failure rate (need 2+ public, 2+ secret, 30%+ overall)\n"
        f"  - Try a repo with simpler standalone utility functions (e.g. math helpers, string processors)\n"
        f"Files tried: {', '.join(tried_files)}"
    )
    raise RuntimeError(error_msg)


def inflate_hierarchy(state: ArchitectState) -> ArchitectState:
    """Phase 2 - Hierarchy Inflation: inflate ALL functions in the call chain to make debugging harder.

    This inflates ONLY the functions in the actual call path from surface to bug location.
    Each function in the chain is expanded with dummy code, redundant operations, and busy-work
    to make it very difficult to spot where the bug actually is.

    Uses readable names so downstream refactoring passes can apply their own conventions.
    """
    source = state["sabotaged_code"]
    debug_mode = state.get("debug_mode", False)
    line_count = len(source.splitlines())
    if debug_mode:
        print(f"[inflate_hierarchy] Current line count: {line_count}")

    target_func = state["function_name"]
    
    # Get the call chains from state (set during sabotage_init)
    bug_chains = state.get("call_chain", {})
    if not bug_chains:
        if debug_mode:
            print("[inflate_hierarchy] No call chain found -- skipping")
        return state
    
    # Extract unique functions from all call chains
    all_funcs = set()
    for chain in bug_chains.values():
        all_funcs.update(chain)
    all_funcs = sorted(list(all_funcs))  # Convert to sorted list for consistency
    
    if not all_funcs:
        if debug_mode:
            print("[inflate_hierarchy] No functions in call chain -- skipping")
        return state
    
    if debug_mode:
        print(f"[inflate_hierarchy] Found {len(all_funcs)} functions in call chains to inflate: {all_funcs}")

    # Extract all functions as a snippet
    func_sources = []
    for func_name in all_funcs:
        try:
            func_src, _, _ = _extract_function_source(source, func_name)
            func_sources.append(func_src)
        except Exception:
            pass
    
    if not func_sources:
        if debug_mode:
            print("[inflate_hierarchy] Could not extract functions -- skipping")
        return state
    
    chain_source = "\n\n".join(func_sources)
    chain_lines = len(chain_source.splitlines())
    if debug_mode:
        print(f"[inflate_hierarchy] All functions combined: {chain_lines} lines")

    bug_fn  = state.get("bug_func_name", "")
    bug_src = state.get("bug_func_source", "")

    bug_rule = (
        f"\n!!! CRITICAL BUG PRESERVATION !!!\n"
        f"  The function `{bug_fn}` contains an intentional hidden bug.\n"
        f"  Copy its logic VERBATIM -- do NOT rewrite, simplify, or fix it.\n"
        f"  Buggy function body (copy exactly):\n{bug_src}\n"
    ) if bug_fn and bug_src else ""

    prompt = (
        f"Inflate ALL of the following Python functions to make debugging MUCH harder.\n\n"
        f"INFLATION GOALS:\n"
        f"  - Make EVERY function VERY LONG (aim for 50-80+ lines each, more is better!)\n"
        f"  - NO function should remain simple (single return statement is NOT acceptable)\n"
        f"  - Add MANY redundant local variable assignments: _buf, _tmp, _state, _flag, _cache, _context, _metadata\n"
        f"  - Add MULTIPLE dummy computations in each function:\n"
        f"      * `_n = [x*0 for x in range(5)]`\n"
        f"      * `_c = len(data) * 1`\n"
        f"      * `_hash = hash(str(data)) % 999999`\n"
        f"      * `_checksum = sum(ord(c) for c in str(data)[:10])`\n"
        f"      * `_marker = len(str(_hash)) - len(str(_hash))`\n"
        f"  - Add SEVERAL realistic-looking validation checks:\n"
        f"      * `if data is None: return None`\n"
        f"      * `if not isinstance(x, str): x = str(x)`\n"
        f"      * `if not data: return default_value`\n"
        f"      * Type checking: `if not isinstance(count, int): count = int(count)`\n"
        f"  - Wrap logic in MULTIPLE nested always-true conditional blocks:\n"
        f"      * `if True:`, `if _flag == True:`, `if len(_tmp) >= 0:`\n"
        f"      * Create nested if statements (3-4 levels deep)\n"
        f"  - Insert MULTIPLE redundant loops: `for _ in range(1): ...`, `while False: break`\n"
        f"  - Add SEVERAL try-except-finally blocks throughout each function\n"
        f"  - Add MANY logging-style comments: `# Processing data`, `# Validate input`, `# Calculate result`, `# Initialize state`, `# Cleanup resources`\n"
        f"  - Add intermediate result variables even when not needed: `_intermediate_1 = step1()`, `_intermediate_2 = process(_intermediate_1)`\n"
        f"  - Use NORMAL, READABLE names for all new code (inflation only, no obfuscation yet)\n"
        f"  - MANDATORY: Inflate ALL {len(all_funcs)} functions equally -- no exceptions!\n\n"
        f"STRICTLY FORBIDDEN PATTERNS:\n"
        f"  - NEVER create simple nested return chains like:\n"
        f"      return execute_core(args)\n"
        f"      return apply_transformations(args)\n"
        f"      return normalize_arguments(args)\n"
        f"  - NEVER just forward arguments through multiple trivial helper functions\n"
        f"  - Such patterns are TOO OBVIOUS and defeat the purpose of inflation\n"
        f"  - Instead, add REAL logic between operations: variable assignments, conditions, loops, validations\n\n"
        f"HARD RULES:\n"
        f"  - Keep ALL original function names and signatures unchanged\n"
        f"  - Do NOT fix any bugs -- preserve ALL existing logic exactly\n"
        f"  - Do NOT create nested functions (def inside def) -- all functions must be at module level\n"
        f"  - If you need helper functions, define them separately at the module level, not inside other functions\n"
        f"  - Every function you return must be SUBSTANTIALLY expanded (MINIMUM 30+ lines, aim for 50-80+)\n"
        f"  - Return ONLY the Python function definitions (def ...: ...) -- no imports, no markdown\n"
        f"  - Include every new helper function you create at module level\n"
        f"  - The bug must remain hidden among all the inflated code\n"
        f"{bug_rule}\n"
        f"FUNCTIONS TO INFLATE:\n{chain_source}"
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        snippet = _strip_markdown_code(response.content)
        snip_tree = ast.parse(snippet)

        # Check that all functions from the chain are present
        all_func_names = {n.name for n in ast.walk(snip_tree)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        
        # Verify target function exists
        if target_func not in all_func_names:
            if debug_mode:
                print(f"[inflate_hierarchy] Inflation removed `{target_func}` -- reverting")
            return state
        
        # Verify all chain functions are present
        missing = set(all_funcs) - all_func_names
        if missing:
            if debug_mode:
                print(f"[inflate_hierarchy] Warning: Missing functions after inflation: {missing}")
        
        # Check for nested functions (functions defined inside other functions)
        for node in ast.walk(snip_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if this function contains nested function definitions
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if debug_mode:
                            print(f"[inflate_hierarchy] Nested function detected in '{node.name}' -- reverting")
                        return state
        
        # Verify that each function in the chain was actually inflated (not just a single return)
        for node in snip_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in all_funcs:
                # Count non-trivial statements (exclude docstrings)
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    body = body[1:]  # Skip docstring
                
                if len(body) < 8:  # Minimum 8 statements to be considered "inflated"
                    if debug_mode:
                        print(f"[inflate_hierarchy] Function '{node.name}' not sufficiently inflated ({len(body)} statements) -- reverting")
                    return state

        # Splice all functions back
        spliced = _splice_transforms_back(source, snippet, all_funcs)
        new_count = len(spliced.splitlines())
        if debug_mode:
            print(f"[inflate_hierarchy] Inflated: {chain_lines} -> {len(snippet.splitlines())} lines (file: {line_count} -> {new_count})")
        state["sabotaged_code"] = spliced
        
    except Exception as e:
        if debug_mode:
            print(f"[inflate_hierarchy] Inflation failed: {e} -- keeping original")

    return state


def apply_obfuscation_level_2(state: ArchitectState) -> ArchitectState:
    """Phase 3 - Deep Nesting: spaghettification with readable names."""
    if not state.get("refactoring_enabled", False):
        print("[obfuscation_level_2] Refactoring disabled -- skipping")
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
    """Phase 4 - Semantic Stripping: cryptic naming + shadow wrappers."""
    if not state.get("refactoring_enabled", False):
        print("[obfuscation_level_1] Refactoring disabled -- skipping")
        return state

    print("[obfuscation_level_1] Applying full-file obfuscation (cryptic names, shadow wrappers)...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    # Always use level=3 for maximum obfuscation when refactoring is enabled
    result = _obfuscate_full_file(
        state["sabotaged_code"],
        state["function_name"],
        llm,
        level=3,  # Maximum obfuscation
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


def add_misleading_comments(state: ArchitectState) -> ArchitectState:
    """Add deliberately misleading comments in random locations to confuse students and AI.
    
    These comments falsely suggest bugs exist in locations where there are no bugs,
    making it harder to find the actual injected bugs.
    """
    debug_mode = state.get("debug_mode", False)
    
    if debug_mode:
        print("\n[misleading_comments] Adding false bug hints to confuse analysis...")
    
    sabotaged_code = state.get("sabotaged_code", "")
    if not sabotaged_code:
        return state
    
    try:
        tree = ast.parse(sabotaged_code)
    except SyntaxError:
        if debug_mode:
            print("[misleading_comments] Syntax error - skipping")
        return state
    
    lines = sabotaged_code.splitlines()
    
    # Find all function definitions
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_intermediate_processor"):
                functions.append((node.name, node.lineno, node.end_lineno))
    
    if len(functions) < 3:
        return state  # Not enough functions to add misleading comments
    
    # Templates for misleading comments
    misleading_templates = [
        "# BUG: Check the return value here",
        "# TODO: Fix off-by-one error in this function",
        "# FIXME: This condition looks suspicious",
        "# NOTE: Edge case handling might be incorrect",
        "# WARNING: Potential bug with empty inputs",
        "# BUG: Loop boundary needs review",
        "# TODO: Verify this calculation is correct",
        "# FIXME: Check for None values here",
        "# BUG: Index manipulation looks wrong",
        "# NOTE: Return value might be incorrect for edge cases",
    ]
    
    # Select 3-5 random functions (excluding bug locations)
    bug_func_names = set()
    for bug_data in state.get("all_bug_data", []):
        bug_func_names.add(bug_data.get("_debug_func_name", ""))
    
    # Filter out bug functions
    safe_functions = [(name, start, end) for name, start, end in functions 
                      if name not in bug_func_names]
    
    if len(safe_functions) < 2:
        # If almost all functions have bugs, pick any function
        safe_functions = functions
    
    num_comments = min(random.randint(3, 5), len(safe_functions))
    selected_funcs = random.sample(safe_functions, num_comments)
    
    # Add comments to selected functions
    insertions = []  # List of (line_index, comment_text)
    
    for func_name, start_line, end_line in selected_funcs:
        # Pick a random line within the function (not the def line)
        if end_line - start_line < 3:
            target_line = start_line  # Insert before function def
        else:
            # Insert somewhere in the middle of the function
            target_line = random.randint(start_line + 1, min(start_line + 3, end_line - 1))
        
        comment = random.choice(misleading_templates)
        
        # Find the indentation of the target line
        if target_line - 1 < len(lines):
            target_text = lines[target_line - 1]
            indent = len(target_text) - len(target_text.lstrip())
            indented_comment = " " * indent + comment
            insertions.append((target_line - 1, indented_comment))  # 0-indexed
    
    # Sort by line number in reverse to insert from bottom to top
    insertions.sort(reverse=True)
    
    # Insert comments
    for line_idx, comment in insertions:
        lines.insert(line_idx, comment)
        if debug_mode:
            print(f"[misleading_comments] Added at line {line_idx + 1}: {comment.strip()}")
    
    state["sabotaged_code"] = "\n".join(lines)
    
    if debug_mode:
        print(f"[misleading_comments] Added {len(insertions)} misleading bug hints")
    
    return state


def sabotage(state: ArchitectState) -> ArchitectState:
    """Full sabotage pipeline (backward-compat wrapper that runs all 5 phases in sequence).

    Execution path by difficulty level:
      Level 1: sabotage_init -> inflate_hierarchy -> obfuscation_level_1 -> verify_sabotage
      Level 2: sabotage_init -> inflate_hierarchy -> obfuscation_level_2 -> verify_sabotage
      Level 3: sabotage_init -> inflate_hierarchy -> obfuscation_level_2
                             -> obfuscation_level_1 -> verify_sabotage
    """
    state = sabotage_init(state)
    state = inflate_hierarchy(state)
    level = state["difficulty_level"]
    if level in (2, 3):
        state = apply_obfuscation_level_2(state)
    if level in (1, 3):
        state = apply_obfuscation_level_1(state)
    state = verify_sabotage(state)
    state = add_misleading_comments(state)
    return state

