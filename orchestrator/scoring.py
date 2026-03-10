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

    # Parse "Test N: ✓ PASS / ✗ FAIL / ✗ CRASH" lines
    passed = total = 0
    for line in output.split("\n"):
        stripped = line.strip()
        if "PASS" in stripped and stripped.startswith("Test"):
            passed += 1
            total += 1
        elif ("FAIL" in stripped or "CRASH" in stripped) and stripped.startswith("Test"):
            total += 1

    all_passed = passed > 0 and passed == total
    score = int((passed / total) * 80) if total > 0 else 0

    return {
        "passed": passed,
        "total": total,
        "score": score,
        "output": output.strip(),
        "all_passed": all_passed,
    }


def run_tests(workspace_path: str) -> dict:
    """Run public tests (challenge_run.py) — shown to student during practice."""
    return _run_test_file(workspace_path, "challenge_run.py")


def run_secret_tests(workspace_path: str) -> dict:
    """Run secret tests (challenge_run_secret.py) — used for final submission scoring."""
    return _run_test_file(workspace_path, "challenge_run_secret.py")


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
    sabotaged_code: str = "",
) -> dict:
    """
    Full evaluation of a student submission.
    Runs both public (challenge_run.py) and secret (challenge_run_secret.py) tests,
    combines their results, and returns all score components.

    Returns a dict with all score components and diagnostic fields:
      test_score        : int  (0-80, combined pass rate)
      diff_score        : int  (0-20)
      hint_penalty      : int  (0-30, subtracted)
      total_score       : int  (0-100, clamped to 0)
      passed            : int  (combined)
      total_tests       : int  (combined)
      all_passed        : bool
      test_output       : str  (combined output with section headers)
      public_output     : str  (public test output only)
      secret_output     : str  (secret test output only)
      correct_location  : bool  (informational only)
    """
    public_result = run_tests(workspace_path)
    secret_result = run_secret_tests(workspace_path)

    combined_passed = public_result["passed"] + secret_result["passed"]
    combined_total  = public_result["total"]  + secret_result["total"]
    combined_score  = int((combined_passed / combined_total) * 80) if combined_total > 0 else 0
    all_passed      = combined_passed > 0 and combined_passed == combined_total

    combined_output = (
        "-- Public Tests --\n" + public_result["output"] + "\n\n"
        "-- Secret Tests --\n" + secret_result["output"]
    ).strip()

    # Diff score: measures how minimal the fix was (similarity to the sabotaged code
    # they received). Perfect minimal fix = 20/20. Comparing to original_code is wrong
    # when obfuscation is applied, since the student's code can never match the clean original.
    diff_ref   = sabotaged_code if sabotaged_code else original_code
    diff_score = compute_diff_score(diff_ref, student_code)
    penalty     = hint_penalty(hints_used)
    correct_loc = check_fix_location(original_code, student_code, bug_func_name)
    total       = max(0, combined_score + diff_score - penalty)

    return {
        "test_score":       combined_score,
        "diff_score":       diff_score,
        "hint_penalty":     penalty,
        "total_score":      total,
        "passed":           combined_passed,
        "total_tests":      combined_total,
        "all_passed":       all_passed,
        "test_output":      combined_output,
        "public_output":    public_result["output"],
        "secret_output":    secret_result["output"],
        "correct_location": correct_loc,
    }
