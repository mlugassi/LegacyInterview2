import ast
import json
import textwrap

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from architect.state import ArchitectState

_LEVEL_INSTRUCTIONS = {
    1: """
You are performing a Level 1 ("Messy Code") sabotage on a SINGLE Python function.
Rules:
- Rename ALL internal local variables to meaningless names like var1, var2, temp_x, result_val, etc.
  (Do NOT rename the function itself or its parameters — those are the public API.)
- Add misleading inline comments that describe the CORRECT behavior while the code does the WRONG thing.
- Inject EXACTLY ONE subtle logical bug. Acceptable bug types:
    * Off-by-one error (e.g., < instead of <=, range(n) instead of range(n+1))
    * Wrong arithmetic operator (e.g., - instead of +, * instead of /)
    * Wrong comparison operator (e.g., > instead of >=)
  The bug must be fixable in 1–3 lines without renaming, refactoring, or adding imports.
""",
    2: """
You are performing a Level 2 ("Spaghetti Logic") sabotage on a SINGLE Python function.
Rules:
- Rename ALL internal local variables to meaningless names (var1, temp_x, flag_a, etc.).
  (Do NOT rename the function itself or its parameters.)
- Add misleading comments inside conditionals that describe correct behavior while code is wrong.
- Inject EXACTLY ONE subtle logical bug:
    * A wrong condition branch (e.g., `if x > 0` instead of `if x >= 0`)
    * A loop boundary error
    * A wrong variable used inside a nested if
  The bug must be fixable in 1–3 lines.
""",
    3: """
You are performing a Level 3 ("Sensitive Code") sabotage on a SINGLE Python function.
Rules:
- Rename ALL internal local variables to meaningless names (var1, coeff_a, magic_val, etc.).
  (Do NOT rename the function itself or its parameters.)
- Corrupt EXACTLY ONE numeric constant (magic number) in the formula by a small but impactful
  amount (e.g., change 1.0 to 1.1, change 2 to 3, change 0.5 to 0.05).
- Add a comment beside the corrupted constant that describes what the CORRECT value should produce.
- The bug must be fixable by changing only that one number (1–3 lines max).
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
  "bug_description": "<one sentence describing exactly what you changed and why it breaks things>"
}

Critical rules:
- sabotaged_function_code must be ONLY the function definition (def ...: ...) — nothing before or after.
- The `def` line MUST keep the EXACT SAME function name as the original. Do NOT rename the function.
- Preserve the EXACT original indentation level of the function (copy it from the input).
- Do NOT modify any docstrings or string literals that already exist in the function.
  You may only ADD new misleading comments on new lines inside the function body.
- expected_output and actual_output must differ.
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
    """Return the name of the best MODULE-LEVEL function (not a class method)."""
    tree = ast.parse(source)
    best_name = ""
    best_score = -1

    # Only iterate direct children of the module — excludes class methods
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") or node.name.startswith("_"):
            continue
        if node.end_lineno - node.lineno < 5:
            continue
        score = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                score += 3
            if isinstance(child, ast.If):
                score += 2
            if isinstance(child, ast.Return):
                score += 2
            if isinstance(child, ast.BinOp):
                score += 1
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
