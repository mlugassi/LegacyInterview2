import os

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
            exp_repr  = tc.get("expected", "None")
            
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
    
    return state
