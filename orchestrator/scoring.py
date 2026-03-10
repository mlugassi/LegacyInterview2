"""
Scoring Engine for the Legacy Code Challenge.

Score breakdown:
  - LLM score    : 0–100 pts  (LLM evaluates fix quality, minimality, and test results)
  - Hint penalty : cumulative deduction per hint used (−2, −6, −12, −20, −30 max)
  - Total        : max(0, llm_score − hint_penalty)
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Cumulative penalty after N hints used (index = number of hints)
HINT_PENALTY = [0, 2, 6, 12, 20, 30]


# ── Test runner ───────────────────────────────────────────────────────────────

def _run_test_file(workspace_path: str, filename: str) -> dict:
    """
    Execute a test file (challenge_run.py or challenge_run_secret.py) and parse output.

    Returns a dict:
      passed      : int   — number of passing cases
      total       : int   — total cases executed
      score       : int   — 0–80 pts
      output      : str   — full stdout+stderr text
      all_passed  : bool
    """
    workspace = Path(workspace_path)
    test_file = workspace / filename

    if not test_file.exists():
        return {
            "passed": 0, "total": 0, "score": 0,
            "output": f"Error: {filename} not found in workspace.",
            "all_passed": False,
        }

    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        output = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return {
            "passed": 0, "total": 0, "score": 0,
            "output": "Error: Tests timed out (possible infinite loop).",
            "all_passed": False,
        }
    except Exception as exc:
        return {
            "passed": 0, "total": 0, "score": 0,
            "output": f"Error running tests: {exc}",
            "all_passed": False,
        }

    # Parse "Test N: [PASS] / [FAIL] / [CRASH]" lines
    passed = total = 0
    per_test: list[dict] = []   # [{num, status}]
    for line in output.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Test") and ("PASS" in stripped or "FAIL" in stripped or "CRASH" in stripped):
            import re as _re
            m = _re.match(r"Test\s+(\d+):\s+\[?(PASS|FAIL|CRASH)", stripped)
            if m:
                num, status = int(m.group(1)), m.group(2)
                per_test.append({"num": num, "status": status})
                total += 1
                if status == "PASS":
                    passed += 1

    all_passed = passed > 0 and passed == total
    score = int((passed / total) * 80) if total > 0 else 0

    return {
        "passed": passed,
        "total": total,
        "score": score,
        "output": output.strip(),
        "all_passed": all_passed,
        "per_test": per_test,
    }


def run_tests(workspace_path: str) -> dict:
    """Run public tests (challenge_run.py) — shown to student during practice."""
    return _run_test_file(workspace_path, "challenge_run.py")


def run_secret_tests(workspace_path: str) -> dict:
    """Run secret tests (challenge_run_secret.py) — used for final submission scoring."""
    return _run_test_file(workspace_path, "challenge_run_secret.py")


# ── Function extractor ────────────────────────────────────────────────────────

def _extract_all_funcs(code: str) -> dict[str, str]:
    """Return {func_name: source_lines} for every top-level or nested function."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    result = {}
    lines = code.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = "\n".join(lines[node.lineno - 1: node.end_lineno])
    return result


# ── Hint penalty ──────────────────────────────────────────────────────────────

def hint_penalty(hints_used: int) -> int:
    """Return the cumulative score deduction for the number of hints used."""
    return HINT_PENALTY[min(hints_used, len(HINT_PENALTY) - 1)]


# ── Location check ────────────────────────────────────────────────────────────

def check_fix_location(original_code: str, student_code: str, bug_func_name: str) -> bool:
    """
    Return True if the student's change touched the function that contains
    the bug (bug_func_name).  Used only for feedback; does not affect score.
    """
    if not bug_func_name:
        return True

    try:
        orig_tree = ast.parse(original_code)
        student_tree = ast.parse(student_code)
    except SyntaxError:
        return False

    def func_source(tree: ast.AST, name: str) -> str | None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return ast.unparse(node)
        return None

    orig_func = func_source(orig_tree, bug_func_name)
    student_func = func_source(student_tree, bug_func_name)

    if orig_func is None or student_func is None:
        return True

    return orig_func != student_func


# ── LLM scorer ────────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """\
You are an expert code reviewer evaluating a student's fix for a programming challenge.

The student received a Python file with a deliberately injected bug and was asked to fix it
with MINIMAL changes — ideally 1–3 lines, without renaming variables or restructuring logic.

You will be given:
1. For each function the student changed: the buggy version, their fix, and the expected fix.
2. Test results BEFORE the student's change (run on the buggy code).
3. Test results AFTER the student's change (run on their submitted code).

Evaluate the submission on these criteria:
- Correctness  (0–50 pts): Do the tests pass after the fix? Is the logic correct?
- Minimality   (0–30 pts): Did the student change only what was necessary?
  A fix that changes exactly the right token(s) scores 30. Extra rewrites lose points.
- Quality      (0–20 pts): Does the fix match the expected solution in intent and style?
  A fix that perfectly matches the expected approach scores 20.

Respond ONLY with a valid JSON object — no markdown, no extra text:
{"score": <integer 0-100>, "explanation": "<2-4 sentence explanation>"}
"""


def llm_score_submission(
    changed_funcs: list[dict],
    test_before_passed: int,
    test_before_total: int,
    test_after_passed: int,
    test_after_total: int,
    test_output: str,
    per_test_before: list[dict] | None = None,
    per_test_after: list[dict] | None = None,
) -> dict:
    """
    Ask the LLM to score the submission 0–100 and return an explanation.

    changed_funcs: list of {name, buggy, student, expected}
    Returns: {score: int, explanation: str}
    """
    # Build the function-by-function section
    func_sections = []
    for i, f in enumerate(changed_funcs, 1):
        func_sections.append(
            f"### Function {i}: `{f['name']}`\n\n"
            f"**Buggy version (received by student):**\n```python\n{f['buggy']}\n```\n\n"
            f"**Student's fix:**\n```python\n{f['student']}\n```\n\n"
            f"**Expected correct version:**\n```python\n{f['expected']}\n```"
        )

    funcs_text = "\n\n".join(func_sections) if func_sections else "(no function changes detected)"

    # Build per-test comparison table if we have both before and after breakdowns
    per_test_section = ""
    if per_test_before and per_test_after:
        before_map = {t["num"]: t["status"] for t in per_test_before}
        after_map  = {t["num"]: t["status"] for t in per_test_after}
        all_nums = sorted(set(before_map) | set(after_map))
        rows = []
        for n in all_nums:
            b = before_map.get(n, "—")
            a = after_map.get(n, "—")
            change = ""
            if b != a:
                change = f"  ← changed ({b} → {a})"
            rows.append(f"  Test {n:2d}: before={b:<5}  after={a}{change}")
        per_test_section = "\n\n## Per-Test Breakdown\n```\n" + "\n".join(rows) + "\n```"

    human_text = (
        f"## Changed Functions\n\n{funcs_text}\n\n"
        f"## Test Results\n"
        f"- Before student's fix: {test_before_passed}/{test_before_total} tests passed\n"
        f"- After student's fix:  {test_after_passed}/{test_after_total} tests passed"
        f"{per_test_section}\n\n"
        f"## Test Output (after fix)\n```\n{test_output[:2000]}\n```"
    )

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke([
            SystemMessage(content=_LLM_SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ])
        raw = response.content.strip()
        # Strip markdown code fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        score = max(0, min(100, int(data.get("score", 0))))
        explanation = str(data.get("explanation", ""))
        return {"score": score, "explanation": explanation}
    except Exception as exc:
        # Fallback: derive a simple score from test results so the UI never crashes
        if test_after_total > 0:
            ratio = test_after_passed / test_after_total
            fallback_score = int(ratio * 100)
        else:
            fallback_score = 0
        return {
            "score": fallback_score,
            "explanation": f"LLM scoring unavailable ({exc}). Score based on test results only.",
        }


# ── Baseline test runner (sabotaged code) ─────────────────────────────────────

def _run_tests_on_sabotaged(workspace_path: str, target_file: str, sabotaged_code: str) -> dict:
    """
    Temporarily write sabotaged_code to target_file, run both test files,
    then restore the original content.
    Returns {passed, total, per_test: [{num, status}]}.
    """
    target = Path(target_file)
    try:
        original_content = target.read_text(encoding="utf-8")
    except Exception:
        return {"passed": 0, "total": 0, "per_test": []}
    try:
        target.write_text(sabotaged_code, encoding="utf-8")
        pub = run_tests(workspace_path)
        sec = run_secret_tests(workspace_path)
        # Offset secret test numbers so they don't collide with public ones
        offset = pub["total"]
        sec_per = [{"num": t["num"] + offset, "status": t["status"]} for t in sec.get("per_test", [])]
        return {
            "passed":   pub["passed"] + sec["passed"],
            "total":    pub["total"]  + sec["total"],
            "per_test": pub.get("per_test", []) + sec_per,
        }
    except Exception:
        return {"passed": 0, "total": 0, "per_test": []}
    finally:
        try:
            target.write_text(original_content, encoding="utf-8")
        except Exception:
            pass


# ── Master evaluator ─────────────────────────────────────────────────────────

def evaluate_submission(
    workspace_path: str,
    student_code: str,
    original_code: str,
    bug_func_name: str,
    hints_used: int,
    sabotaged_code: str = "",
    target_file: str = "",
) -> dict:
    """
    Full evaluation of a student submission.

    Runs both public and secret tests, then asks an LLM to score the fix quality.

    Returns a dict with all score components and diagnostic fields:
      llm_score         : int  (0-100, from LLM)
      llm_explanation   : str  (LLM's explanation)
      hint_penalty      : int  (0-30, subtracted)
      total_score       : int  (0-100, clamped to 0)
      passed            : int  (combined public + secret)
      total_tests       : int
      all_passed        : bool
      test_output       : str  (combined output)
      public_output     : str
      secret_output     : str
      correct_location  : bool  (informational only)
    """
    public_result = run_tests(workspace_path)
    secret_result = run_secret_tests(workspace_path)

    combined_passed = public_result["passed"] + secret_result["passed"]
    combined_total  = public_result["total"]  + secret_result["total"]
    all_passed      = combined_passed > 0 and combined_passed == combined_total

    combined_output = (
        "-- Public Tests --\n" + public_result["output"] + "\n\n"
        "-- Secret Tests --\n" + secret_result["output"]
    ).strip()

    # Identify which functions the student changed vs the sabotaged baseline
    ref_code = sabotaged_code if sabotaged_code else original_code
    ref_funcs = _extract_all_funcs(ref_code)
    stu_funcs = _extract_all_funcs(student_code)
    orig_funcs = _extract_all_funcs(original_code)

    common = set(ref_funcs) & set(stu_funcs)
    changed_funcs = []
    for name in sorted(common):
        if ref_funcs[name] != stu_funcs[name]:
            changed_funcs.append({
                "name":     name,
                "buggy":    ref_funcs[name],
                "student":  stu_funcs[name],
                "expected": orig_funcs.get(name, "(not found in original)"),
            })

    # Collect per-test results for the "after" state (offset secret test numbers)
    offset = public_result["total"]
    sec_per_after = [{"num": t["num"] + offset, "status": t["status"]} for t in secret_result.get("per_test", [])]
    per_test_after = public_result.get("per_test", []) + sec_per_after

    # Run tests on the sabotaged baseline so we can report accurate before/after counts.
    if target_file and sabotaged_code:
        before_result   = _run_tests_on_sabotaged(workspace_path, target_file, sabotaged_code)
        before_passed   = before_result["passed"]
        before_total    = before_result["total"]
        per_test_before = before_result["per_test"]
    else:
        before_passed, before_total, per_test_before = 0, combined_total, []

    # LLM scores the submission 0-100
    llm_result = llm_score_submission(
        changed_funcs=changed_funcs,
        test_before_passed=before_passed,
        test_before_total=before_total,
        test_after_passed=combined_passed,
        test_after_total=combined_total,
        test_output=combined_output,
        per_test_before=per_test_before,
        per_test_after=per_test_after,
    )

    penalty    = hint_penalty(hints_used)
    total      = max(0, llm_result["score"] - penalty)
    correct_loc = check_fix_location(original_code, student_code, bug_func_name)

    return {
        "llm_score":        llm_result["score"],
        "llm_explanation":  llm_result["explanation"],
        "hint_penalty":     penalty,
        "total_score":      total,
        "passed":           combined_passed,
        "total_tests":      combined_total,
        "all_passed":       all_passed,
        "test_output":      combined_output,
        "public_output":    public_result["output"],
        "secret_output":    secret_result["output"],
        "correct_location": correct_loc,
        # Keep these for backwards compatibility with any display code
        "test_score":       int((combined_passed / combined_total) * 80) if combined_total > 0 else 0,
        "diff_score":       0,
    }
