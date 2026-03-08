"""
Scoring Engine for the Legacy Code Challenge.

Score breakdown:
  - Test score   : 0–80 pts  (80 × passed_tests / total_tests)
  - Diff score   : 0–20 pts  (20 × SequenceMatcher similarity vs original code)
  - Hint penalty : cumulative deduction per hint used (−2, −6, −12, −20, −30 max)
  - Total        : max(0, test_score + diff_score − hint_penalty)
"""

import ast
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

# Cumulative penalty after N hints used (index = number of hints)
HINT_PENALTY = [0, 2, 6, 12, 20, 30]


# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests(workspace_path: str) -> dict:
    """
    Execute challenge_run.py and parse its output.

    Returns a dict:
      passed      : int   — number of passing cases
      total       : int   — total cases executed
      score       : int   — 0–80 pts
      output      : str   — full stdout+stderr text
      all_passed  : bool
    """
    workspace = Path(workspace_path)
    challenge_run = workspace / "challenge_run.py"

    if not challenge_run.exists():
        return {
            "passed": 0, "total": 0, "score": 0,
            "output": "Error: challenge_run.py not found in workspace.",
            "all_passed": False,
        }

    try:
        result = subprocess.run(
            [sys.executable, str(challenge_run)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout + (result.stderr or "")
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

    # Parse "Case N: PASS / FAIL / CRASH" lines
    passed = total = 0
    for line in output.split("\n"):
        stripped = line.strip()
        if ": PASS" in stripped:
            passed += 1
            total += 1
        elif ": FAIL" in stripped or ": CRASH" in stripped:
            total += 1

    all_passed = "ALL PASS" in output or "bug is fixed" in output.lower()
    score = int((passed / total) * 80) if total > 0 else 0

    return {
        "passed": passed,
        "total": total,
        "score": score,
        "output": output.strip(),
        "all_passed": all_passed,
    }


# ── Diff / similarity score ───────────────────────────────────────────────────

def compute_diff_score(original_code: str, student_code: str) -> int:
    """
    Return 0–20 pts based on how similar the student's code is to the
    original (pre-sabotage) source.  Higher similarity = higher score,
    reflecting that a minimal, targeted fix stays close to the original.
    """
    ratio = SequenceMatcher(None, original_code, student_code).ratio()
    return int(20 * ratio)


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
        return True  # cannot validate — assume correct

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
        return True  # function may have been renamed by obfuscation — assume OK

    return orig_func != student_func  # True = student changed the right function


# ── Master evaluator ─────────────────────────────────────────────────────────

def evaluate_submission(
    workspace_path: str,
    student_code: str,
    original_code: str,
    bug_func_name: str,
    hints_used: int,
) -> dict:
    """
    Full evaluation of a student submission.

    Returns a dict with all score components and diagnostic fields:
      test_score      : int  (0–80)
      diff_score      : int  (0–20)
      hint_penalty    : int  (0–30, subtracted)
      total_score     : int  (0–100, clamped to 0)
      passed          : int
      total_tests     : int
      all_passed      : bool
      test_output     : str
      correct_location: bool  (informational only)
    """
    test_result = run_tests(workspace_path)
    diff_score = compute_diff_score(original_code, student_code)
    penalty = hint_penalty(hints_used)
    correct_loc = check_fix_location(original_code, student_code, bug_func_name)
    total = max(0, test_result["score"] + diff_score - penalty)

    return {
        "test_score": test_result["score"],
        "diff_score": diff_score,
        "hint_penalty": penalty,
        "total_score": total,
        "passed": test_result["passed"],
        "total_tests": test_result["total"],
        "all_passed": test_result["all_passed"],
        "test_output": test_result["output"],
        "correct_location": correct_loc,
    }
