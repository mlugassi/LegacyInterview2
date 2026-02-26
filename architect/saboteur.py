import ast
import difflib
import json
import random
import textwrap

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from architect.state import ArchitectState

# Number of helper functions to inject bugs into, per difficulty level
_BUGS_PER_LEVEL = {1: 1, 2: 2, 3: 2}

_LEVEL_INSTRUCTIONS = {
    1: """
You are performing a Level 1 ("Messy Code") sabotage on a SINGLE Python function.

STEP 1 — Understand the function contract:
  Read the function carefully. Identify exactly what it is SUPPOSED to do:
  its inputs, its expected outputs, and its core logic (loop, filter, accumulation, transformation).

STEP 2 — Inject a FUNCTIONAL bug (the code runs but does the wrong thing):
  Break the core behavior in a way that is non-obvious. Good examples:
    * A loop that should process ALL elements but skips one (off-by-one in range or index)
    * A filter/condition that should INCLUDE a class of inputs but now EXCLUDES them (or vice versa)
    * An accumulator initialized to the wrong value (e.g., 1 instead of 0)
    * A return that happens one iteration too early
    * A comparison that causes the wrong branch to be taken for a common input
  You may make 1–3 small coordinated changes to produce the behavioral failure.
  The bug must make the function return a WRONG RESULT for a clear, representable test case.

STEP 3 — Obfuscate variable names: rename every internal local variable to meaningless names
  (var1, temp_x, result_buf, d_ptr, etc.).
  Do NOT rename the function itself, its parameters, or any imported names.

STEP 4 — Structural obfuscation (make the code harder to read without breaking it):
  Apply 2–3 of the following techniques to parts of the function NOT involved in the bug:
    * Split a simple expression into multiple unnecessary intermediate steps
      e.g.: `return x * 2 + 1` → `_t1 = x * 2; _t2 = _t1 + 1; return _t2`
    * Replace a readable `for` loop with an equivalent `while` loop (or vice versa)
    * Add an unreachable / dead-code branch that looks plausible
      e.g.: `if var1 is None: return var1` before a line where var1 is never None
    * Inline a trivial helper expression using a redundant boolean cast or identity op
      e.g.: `bool(flag_a) == True` instead of just `flag_a`
    * Use a list/dict construction in an unnecessarily roundabout way
  These changes must NOT affect the observable output — they only reduce readability.

STEP 5 — Add 1–2 misleading inline comments near the bug site.
  The comment must describe what the code APPEARS to be doing correctly — as if it is right.
  NEVER use words like "incorrect", "wrong", "bug", "error", "incorrectly", "off-by-one",
  "broken", "corrupted", "intentional", "mistake" or any synonym that hints at a problem.
  Example of BAD comment: `# Incorrectly appending closing brace`  ← reveals the bug
  Example of GOOD comment: `# Append the closing brace to complete the field`  ← looks correct
""",
    2: """
You are performing a Level 2 ("Spaghetti Logic") sabotage on a SINGLE Python function.

STEP 1 — Understand the function contract:
  Read the function carefully. Identify its purpose, its conditional branches,
  and what each path is supposed to return.

STEP 2 — Inject a FUNCTIONAL bug inside a branch or loop:
  Break logic so a specific class of inputs is handled incorrectly. Good examples:
    * A branch that should handle a special case (zero, empty, negative) now handles the wrong case
    * A loop that builds a result correctly for most inputs but silently drops or duplicates one category
    * Two variables swapped in a conditional body (uses var_a where var_b was intended)
    * A guard condition inverted so the default path handles what the special path should
  You may make 1–3 coordinated changes. The code must still run without errors.

STEP 3 — Obfuscate variable names: rename every internal local variable (flag_a, temp_x, ptr_b, etc.).
  Do NOT rename the function name, parameters, or imported names.

STEP 4 — Structural obfuscation (make the code harder to read without breaking it):
  Apply 2–3 of the following to parts NOT involved in the bug:
    * Split a simple expression into multiple unnecessary intermediate steps
    * Replace a `for` loop with a `while` loop (or vice versa)
    * Add a plausible-looking but unreachable dead-code branch
    * Use identity/tautology expressions: `bool(flag_a) == True`, `x + 0`, etc.
    * Construct a list or dict in a roundabout way
  These must NOT change the observable output.

STEP 5 — Add 1–2 misleading inline comments near the bug that accurately describe
  the INTENDED behavior, making the student read "correct" documentation next to wrong code.
  NEVER use words like "incorrect", "wrong", "bug", "error", "broken", "corrupted",
  "intentional", "mistake", or any synonym. The comment must read as if the code is correct.
""",
    3: """
You are performing a Level 3 ("Sensitive Code") sabotage on a SINGLE Python function.

STEP 1 — Understand the function contract:
  Read the function carefully. Identify the mathematical formula or algorithm it implements
  and which numeric constants are critical to its correctness.

STEP 2 — Inject a FUNCTIONAL numeric bug:
  Corrupt the behavior by changing one or two numeric constants so that:
    * The result is plausibly wrong (not obviously zero or infinity)
    * Diagnosing it requires understanding the formula, not just reading the code
  Good examples:
    * Change a divisor from N to N+1 (average becomes wrong)
    * Change an exponent from 2 to 3 (quadratic becomes cubic)
    * Swap the order of subtraction (a-b becomes b-a, sign flips)
    * Change a threshold constant that controls branching
  You may make 1–3 changes. The code must still run without errors.

STEP 3 — Obfuscate variable names: rename every internal local variable (coeff_a, magic_val, var1, etc.).
  Do NOT rename the function name, parameters, or imported names.

STEP 4 — Structural obfuscation (make the code harder to read without breaking it):
  Apply 2–3 of the following to parts NOT involved in the bug:
    * Split a simple computation into multiple unnecessary intermediate variables
    * Replace a `for` loop with a `while` loop (or vice versa)
    * Add a plausible-looking but unreachable dead-code branch
    * Use identity expressions or redundant casts that don't affect output
  These must NOT change the observable output.

STEP 5 — Add 1–2 misleading inline comments near the corrupted constants
  that describe the CORRECT formula, while the code implements the wrong one.
  NEVER use words like "incorrect", "wrong", "bug", "error", "broken", "corrupted",
  "intentional", "mistake", or any synonym. The comment must read as if the formula is right.
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
  "test_args": "<example argument list as a Python literal, e.g. (10, 3) or ('hello',)>",
  "expected_output": "<string representation of the CORRECT return value>",
  "actual_output": "<string representation of the BUGGY return value>",
  "bug_description": "<one sentence describing the FUNCTIONAL behavior that is now broken and why>"
}

Critical rules:
- sabotaged_function_code must be ONLY the function definition (def ...: ...) — nothing before or after.
- The `def` line MUST keep the EXACT SAME function name as the original. Do NOT rename the function.
- Preserve the EXACT original indentation level of the function (copy it from the input).
- Do NOT modify any docstrings or string literals that already exist in the function.
  You may only ADD new misleading comments on new lines inside the function body.
- MANDATORY OBFUSCATION: Every internal local variable (assigned inside the function body, NOT
  parameters) MUST be renamed to a meaningless name: var1, temp_x, result_buf, d_ptr, idx_k, etc.
  Keeping any original local variable name is a failure. Rename ALL of them — no exceptions.
- test_args MUST use only primitive Python literals: int, float, str, list, tuple, dict.
  Example: (10, 3) or ([1, 2, 3],) or ("hello world",) — NOT custom class instances.
- expected_output and actual_output must be primitive Python literals too (int, float, str, list…).
  They must differ.
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


def _pick_best_function(source: str) -> str:
    """Pick randomly from the top-3 MODULE-LEVEL functions that work with primitive types."""
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
        raise ValueError("No suitable public module-level function found in the target file.")

    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = scored[:3]
    _, chosen = random.choice(top_n)
    print(f"[saboteur] Function candidates: {[n for _, n in top_n]} → chose '{chosen}'")
    return chosen


def _find_called_module_functions(func_node: ast.FunctionDef, module_func_names: set) -> list:
    """Return names of module-level functions directly called inside func_node."""
    called = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in module_func_names:
                called.add(node.func.id)
    return list(called)


def _pick_surface_function(source: str) -> tuple[str, dict]:
    """
    Pick the surface function (entry point reported as broken to the student).
    Strongly prefers functions that call other module-level functions,
    so the bug can be hidden in a helper the student must discover.
    Returns (surface_func_name, module_func_nodes).
    """
    tree = ast.parse(source)
    module_func_nodes = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
        and not node.name.startswith("test")
        and node.end_lineno - node.lineno >= 5
    }
    module_func_names = set(module_func_nodes.keys())

    # Score each function as a surface candidate: large bonus for having callable helpers
    caller_candidates: list[tuple[int, str]] = []
    for name, node in module_func_nodes.items():
        helpers = _find_called_module_functions(node, module_func_names - {name})
        substantial_helpers = [
            h for h in helpers
            if module_func_nodes[h].end_lineno - module_func_nodes[h].lineno >= 5
        ]
        if not substantial_helpers:
            continue
        score = len(substantial_helpers) * 30 + (node.end_lineno - node.lineno)
        caller_candidates.append((score, name))

    if caller_candidates:
        caller_candidates.sort(key=lambda x: x[0], reverse=True)
        top_n = caller_candidates[:3]
        _, chosen = random.choice(top_n)
        print(f"[saboteur] Surface candidates (with helpers): {[n for _, n in top_n]} → '{chosen}'")
        return chosen, module_func_nodes

    # Fallback: no function calls helpers — pick the most complex function directly
    print("[saboteur] No caller-surface found — falling back to direct mode.")
    return _pick_best_function(source), module_func_nodes


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _try_exec(full_source: str, func_name: str, test_args_str: str) -> tuple[bool, str]:
    """
    Execute the full file source in an isolated namespace, call func_name(*args),
    and return (success, repr_of_result).  Falls back to (False, "") on any error.
    """
    namespace: dict = {}
    try:
        exec(compile(full_source, "<exec>", "exec"), namespace)  # runs all imports + defs
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
) -> tuple[str, dict] | tuple[None, None]:
    """
    Ask GPT to sabotage bug_func_name within current_source.
    Splices result back. Returns (new_source, data_dict) or (None, None) on failure.
    """
    func_source, start_line, end_line = _extract_function_source(current_source, bug_func_name)

    if indirect_mode:
        user_content = (
            f"SABOTAGE INSTRUCTIONS:\n{instructions}\n\n"
            f"SURFACE FUNCTION (entry point the student tests — do NOT modify it):\n"
            f"```python\n{surface_source}\n```\n\n"
            f"HELPER FUNCTION TO SABOTAGE (inject the bug into THIS function only):\n"
            f"```python\n{func_source}\n```\n\n"
            f"CRITICAL: test_args must be valid arguments for calling `{surface_func_name}`. "
            f"expected_output and actual_output must be the results of calling "
            f"`{surface_func_name}(test_args)` — correct vs. buggy. "
            f"Trace how this helper's bug propagates up to the surface return value."
        )
    else:
        user_content = (
            f"SABOTAGE INSTRUCTIONS:\n{instructions}\n\n"
            f"FUNCTION TO SABOTAGE:\n```python\n{func_source}\n```"
        )

    for attempt in range(1, 4):
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

        # Warn if no local variables were renamed, but only hard-block on first two attempts
        if not _variables_were_renamed(func_source, sabotaged_func):
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
    source = state["original_code"]

    max_outer_attempts = 3
    for outer in range(1, max_outer_attempts + 1):
        # Pick the surface function: prefer ones that call other module-level functions
        surface_func_name, module_func_nodes = _pick_surface_function(source)
        module_func_names = set(module_func_nodes.keys())
        surface_node = module_func_nodes.get(surface_func_name)

        # Find substantial helpers called by the surface function
        helpers: list[str] = []
        if surface_node:
            raw = _find_called_module_functions(surface_node, module_func_names - {surface_func_name})
            helpers = [h for h in raw
                       if module_func_nodes[h].end_lineno - module_func_nodes[h].lineno >= 5]

        # Decide bug targets and mode
        n_bugs = _BUGS_PER_LEVEL.get(level, 1)
        if helpers:
            # Indirect mode: bugs hidden in helpers — student must investigate to find them
            bug_targets = random.sample(helpers, min(n_bugs, len(helpers)))
            indirect_mode = True
        else:
            # Fallback: sabotage the surface function directly
            bug_targets = [surface_func_name]
            indirect_mode = False

        print(f"[saboteur] Outer attempt {outer}: surface='{surface_func_name}', "
              f"bug_targets={bug_targets}, indirect={indirect_mode}")

        surface_source = ""
        if indirect_mode:
            surface_source, _, _ = _extract_function_source(source, surface_func_name)

        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

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
            )
            if new_source is None:
                print(f"[saboteur] Failed to sabotage '{bug_func_name}' — retrying outer loop")
                failed = True
                break
            current_source = new_source
            all_descriptions.append(data["bug_description"])
            all_data.append(data)
            last_data = data

        if failed or last_data is None:
            continue

        # Verify the combined bug effect through the surface function
        test_args = last_data["test_args"]
        orig_ok, true_expected = _try_exec(source, surface_func_name, test_args)
        sabot_ok, true_actual  = _try_exec(current_source, surface_func_name, test_args)

        if orig_ok and sabot_ok:
            if true_expected == true_actual:
                print(f"[saboteur] Combined effect identical — retrying outer loop")
                continue
            print(f"[saboteur] Verified: EXPECTED={true_expected}  ACTUAL={true_actual}")
        else:
            print(f"[saboteur] exec fallback — orig_ok={orig_ok}, sabot_ok={sabot_ok}")
            true_expected = last_data["expected_output"]
            true_actual   = last_data["actual_output"]
            if true_expected.strip() == true_actual.strip():
                print(f"[saboteur] GPT fallback also identical — retrying outer loop")
                continue

        # Success — student sees surface_func_name as broken, bug(s) are in helpers
        state["sabotaged_code"]  = current_source
        state["function_name"]   = surface_func_name
        state["test_args"]       = test_args
        state["expected_output"] = true_expected
        state["actual_output"]   = true_actual
        state["bug_description"] = " | ".join(all_descriptions)

        print(f"[saboteur] Reported broken : {surface_func_name}")
        print(f"[saboteur] Bug location(s) : {bug_targets}")
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

    raise RuntimeError(f"Failed to produce valid sabotaged code after {max_outer_attempts} outer attempts.")
