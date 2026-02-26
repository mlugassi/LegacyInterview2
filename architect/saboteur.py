import ast
import json
import textwrap

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from architect.state import ArchitectState

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

STEP 3 — Obfuscate: rename every internal local variable to meaningless names
  (var1, temp_x, result_buf, d_ptr, etc.).
  Do NOT rename the function itself, its parameters, or any imported names.

STEP 4 — Add 1–2 misleading inline comments near the bug site that describe
  what the code SHOULD be doing, while it is actually doing the wrong thing.
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

STEP 3 — Obfuscate: rename every internal local variable (flag_a, temp_x, ptr_b, etc.).
  Do NOT rename the function name, parameters, or imported names.

STEP 4 — Add 1–2 misleading inline comments near the bug that accurately describe
  the INTENDED behavior, making the student read "correct" documentation next to wrong code.
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

STEP 3 — Obfuscate: rename every internal local variable (coeff_a, magic_val, var1, etc.).
  Do NOT rename the function name, parameters, or imported names.

STEP 4 — Add 1–2 misleading inline comments near the corrupted constants
  that describe the CORRECT formula, while the code implements the wrong one.
""",
}

_SYSTEM_PROMPT = """
You are the Legacy Challenge Architect. You will receive ONE Python function and must return
a sabotaged version of THAT FUNCTION ONLY — not the whole file.

Reply with ONLY a valid JSON object (no markdown, no explanation), matching this exact schema:

{
  "sabotaged_function_code": "<the complete sabotaged function as a Python string — only the function, nothing else>",
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
    """Return the best MODULE-LEVEL function that works with primitive types."""
    tree = ast.parse(source)
    best_name = ""
    best_score = -1

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") or node.name.startswith("_"):
            continue
        if node.end_lineno - node.lineno < 5:
            continue

        score = 0

        # Reward numeric/string/list logic — these produce testable primitive outputs
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

        # Penalise functions that call unknown callables (likely custom class methods)
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                score -= 1   # obj.method() suggests class usage

        if score > best_score:
            best_score = score
            best_name = node.name

    if not best_name:
        raise ValueError("No suitable public module-level function found in the target file.")
    return best_name


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def sabotage(state: ArchitectState) -> ArchitectState:
    level = state["difficulty_level"]
    instructions = _LEVEL_INSTRUCTIONS.get(level, _LEVEL_INSTRUCTIONS[1])
    source = state["original_code"]

    func_name = _pick_best_function(source)
    func_source, start_line, end_line = _extract_function_source(source, func_name)

    print(f"[saboteur] Target function: {func_name} (lines {start_line+1}–{end_line})")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    user_content = (
        f"SABOTAGE INSTRUCTIONS:\n{instructions}\n\n"
        f"FUNCTION TO SABOTAGE:\n```python\n{func_source}\n```"
    )

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"[saboteur] Calling GPT-4o (level={level}, attempt={attempt})…")
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        try:
            data = _parse_response(response.content)
        except json.JSONDecodeError as e:
            print(f"[saboteur] JSON parse error on attempt {attempt}: {e}")
            continue

        sabotaged_func = data["sabotaged_function_code"]

        # Splice the sabotaged function back into the original file
        lines = source.splitlines(keepends=True)
        new_source = "".join(lines[:start_line]) + sabotaged_func + "\n" + "".join(lines[end_line:])

        # Validate the result is valid Python
        try:
            new_tree = ast.parse(new_source)
        except SyntaxError as e:
            print(f"[saboteur] Syntax error in sabotaged code on attempt {attempt}: {e}")
            continue

        # Verify GPT didn't rename the function
        module_func_names = {
            node.name for node in new_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if func_name not in module_func_names:
            print(f"[saboteur] GPT renamed function on attempt {attempt} — retrying")
            continue

        # Success
        state["sabotaged_code"] = new_source
        state["function_name"] = func_name
        state["test_args"] = data["test_args"]
        state["expected_output"] = data["expected_output"]
        state["actual_output"] = data["actual_output"]
        state["bug_description"] = data["bug_description"]

        print(f"[saboteur] Sabotaged function: {func_name}")
        print(f"[saboteur] Bug: {state['bug_description']}")
        return state

    raise RuntimeError(f"Failed to produce valid sabotaged code after {max_attempts} attempts.")
