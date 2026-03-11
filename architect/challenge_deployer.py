import os
import stat

from architect.state import ArchitectState


def _module_import_path(clone_path: str, target_file: str) -> str:
    """Convert an absolute file path to a dotted module path relative to clone_path."""
    rel = os.path.relpath(target_file, clone_path)          # e.g. utils/math_utils.py
    without_ext = os.path.splitext(rel)[0]                  # e.g. utils/math_utils
    module_path = without_ext.replace(os.sep, ".")          # e.g. utils.math_utils
    return module_path


def deploy_challenge(state: ArchitectState) -> ArchitectState:
    clone_path  = state["clone_path"]
    module_path = _module_import_path(clone_path, state["target_file"])
    func        = state["function_name"]
    
    # Get public and secret tests
    public_tests = state.get("public_tests") or []
    secret_tests = state.get("secret_tests") or []
    
    # Fallback: if we don't have split tests, split test_cases
    if not public_tests and not secret_tests:
        all_tests = state.get("test_cases") or []
        if not all_tests:
            all_tests = [{"args": state["test_args"], "expected": state["expected_output"]}]
        mid = len(all_tests) // 2
        public_tests = all_tests[:mid] if mid > 0 else all_tests
        secret_tests = all_tests[mid:] if mid > 0 else []
    
    def _build_test_file_content(tests: list, test_type: str = "public") -> str:
        """Build the content for a test file."""
        case_lines = []
        for tc in tests:
            args_repr = tc.get("args", "()")
            # Support both "expected" (from verification) and "correct_output" (from GPT)
            exp_value = tc.get("expected") or tc.get("correct_output")
            if not exp_value:
                exp_value = "None"
            
            # Validate that exp_value is a valid Python literal
            # Try to evaluate it - if it works, it's valid and we use it as-is
            try:
                eval(exp_value, {"__builtins__": {}})
                exp_repr = exp_value  # Already a valid Python literal
            except:
                # If eval fails, wrap it as a string literal
                exp_repr = repr(str(exp_value))
            
            # args_repr should be a properly formatted tuple literal from GPT
            # However, GPT sometimes forgets the comma for single-element tuples
            # e.g. ('hello') instead of ('hello',)
            args_str = args_repr.strip()
            
            # Try to evaluate the args to check if it's actually a tuple
            try:
                evaluated = eval(args_str)
                # If eval succeeds but result is NOT a tuple, wrap it
                if not isinstance(evaluated, tuple):
                    args_str = f"({args_str},)"
            except:
                # If eval fails, assume it's already correct format
                pass
            
            case_lines.append(f"    ({args_str}, {exp_repr}),")
        cases_literal = "\n".join(case_lines) if case_lines else "    # No tests"

        return (
            "import sys\n"
            "import os\n\n"
            "# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError with ✓/✗ chars)\n"
            "if hasattr(sys.stdout, 'reconfigure'):\n"
            "    sys.stdout.reconfigure(encoding='utf-8')\n\n"
            "# Make the repo root importable\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n\n"
            f"from {module_path} import {func}\n\n"
            f"# {test_type.upper()} TEST CASES\n"
            "# (args_tuple, correct_expected_output)\n"
            f"_TEST_CASES = [\n{cases_literal}\n]\n\n"
            "def run_tests():\n"
            "    passed = crashed = 0\n"
            "    for i, (args, expected) in enumerate(_TEST_CASES, 1):\n"
            "        try:\n"
            f"            result = {func}(*args)\n"
            "            ok = result == expected\n"
            "            status = '[PASS]' if ok else '[FAIL]'\n"
            "            passed += ok\n"
            "            print(f'  Test {i}: {status}')\n"
            "            if not ok:\n"
            "                print(f'           args     = {args!r}')\n"
            "                print(f'           expected = {expected}')\n"
            "                print(f'           got      = {result}')\n"
            "        except Exception as exc:\n"
            "            crashed += 1\n"
            "            print(f'  Test {i}: [CRASH] - {type(exc).__name__}: {exc}')\n"
            "            print(f'           args = {args!r}')\n\n"
            "    total = len(_TEST_CASES)\n"
            "    broken = total - passed\n"
            "    if passed == total:\n"
            "        print(f'\\n=== ALL TESTS PASSED ({passed}/{total}) ===')\n"
            "        return True\n"
            "    elif crashed == total:\n"
            "        print(f'\\n=== ALL TESTS CRASHED ({crashed}/{total}) ===')\n"
            "        return False\n"
            "    else:\n"
            "        print(f'\\n=== {broken}/{total} tests failed ({crashed} crash, {broken-crashed} wrong value) ===')\n"
            "        return False\n\n"
            "if __name__ == '__main__':\n"
            "    run_tests()\n"
        )
    
    # Create public test file (5 tests that students can see and run)
    challenge_run_path = os.path.join(clone_path, "challenge_run.py")
    public_content = _build_test_file_content(public_tests, "public")
    with open(challenge_run_path, "w", encoding="utf-8") as f:
        f.write(public_content)
    print(f"[deployer] Written: {challenge_run_path}  ({len(public_tests)} public test cases)")
    
    # Create secret test file (5 hidden tests for final validation)
    secret_run_path = os.path.join(clone_path, "challenge_run_secret.py")
    secret_content = _build_test_file_content(secret_tests, "secret")
    with open(secret_run_path, "w", encoding="utf-8") as f:
        f.write(secret_content)
    print(f"[deployer] Written: {secret_run_path}  ({len(secret_tests)} secret test cases)")
    
    # Write detailed explanation for instructor
    detailed_path = os.path.join(clone_path, "detailed_explanation.txt")
    with open(detailed_path, "w", encoding="utf-8") as f:
        f.write(state.get("detailed_explanation", "No detailed explanation available."))
    print(f"[deployer] Written: {detailed_path}")

    # Write snapshot of every file modified by the saboteur to .challenge_snapshot/
    # This lets the student interface compute exact diffs against what was received,
    # even if the JSON is later regenerated or becomes stale.
    snapshot_dir = os.path.join(clone_path, ".challenge_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)

    target_file = state.get("target_file", "")
    sabotaged_code = state.get("sabotaged_code", "")
    if target_file and sabotaged_code:
        rel = os.path.relpath(target_file, clone_path)           # e.g. boltons/strutils.py
        snap_path = os.path.join(snapshot_dir, rel.replace(os.sep, "__"))
        # Remove read-only flag first in case we're overwriting a previous run
        if os.path.exists(snap_path):
            os.chmod(snap_path, stat.S_IWRITE)
        with open(snap_path, "w", encoding="utf-8") as f:
            f.write(sabotaged_code)
        # Mark read-only so the student can't accidentally overwrite it
        os.chmod(snap_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        # Hide the directory on Windows
        try:
            import subprocess as _sp
            _sp.run(["attrib", "+H", snapshot_dir], check=False, capture_output=True)
        except Exception:
            pass
        print(f"[deployer] Snapshot: {snap_path}")

    return state
