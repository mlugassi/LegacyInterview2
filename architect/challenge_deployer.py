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
    target_file = state["target_file"]
    module_path = _module_import_path(clone_path, target_file)
    
    # Get bug-specific data
    all_bug_data = state.get("all_bug_data") or []
    bug_specific_tests = state.get("bug_specific_tests") or {}
    
    # If no per-bug data, fall back to old single-bug behavior
    if not all_bug_data:
        all_bug_data = [{
            "function_name": state.get("function_name", "target_func"),
            "bug_description": state.get("bug_description", "Unknown bug")  # Changed from bug_variant
        }]
        # Use the global tests
        public_tests = state.get("public_tests") or []
        secret_tests = state.get("secret_tests") or []
        if not public_tests and not secret_tests:
            all_tests = state.get("test_cases") or []
            if not all_tests:
                all_tests = [{"args": state["test_args"], "expected": state["expected_output"]}]
            mid = len(all_tests) // 2
            public_tests = all_tests[:mid] if mid > 0 else all_tests
            secret_tests = all_tests[mid:] if mid > 0 else []
        func_name = all_bug_data[0]["function_name"]
        bug_specific_tests[func_name] = public_tests + secret_tests
    
    # Split tests for each bug into public/secret
    bugs_with_tests = []
    for bug in all_bug_data:
        func_name = bug["function_name"]
        all_tests = bug_specific_tests.get(func_name) or []
        mid = len(all_tests) // 2
        public = all_tests[:mid] if mid > 0 else all_tests[:1] if all_tests else []
        secret = all_tests[mid:] if mid > 0 else all_tests[1:] if len(all_tests) > 1 else []
        bugs_with_tests.append({
            "function_name": func_name,
            "bug_description": bug.get("bug_description", ""),  # Changed from bug_variant
            "public_tests": public,
            "secret_tests": secret
        })
    
    def _build_multi_bug_test_file(bugs: list, test_type: str = "public") -> str:
        """Build a test file with separate sections for each bug."""
        
        # Helper to format test cases
        def format_tests(tests):
            case_lines = []
            for tc in tests:
                args_repr = tc.get("args", "()")
                exp_value = tc.get("expected") or tc.get("correct_output")
                if not exp_value:
                    exp_value = "None"
                
                # Validate that exp_value is a valid Python literal
                try:
                    eval(exp_value, {"__builtins__": {}})
                    exp_repr = exp_value
                except:
                    exp_repr = repr(str(exp_value))
                
                # Handle tuple formatting
                args_str = args_repr.strip()
                try:
                    evaluated = eval(args_str)
                    if not isinstance(evaluated, tuple):
                        args_str = f"({args_str},)"
                except:
                    pass
                
                case_lines.append(f"    ({args_str}, {exp_repr}),")
            return "\n".join(case_lines) if case_lines else "    # No tests"
        
        # Build imports section
        imports = []
        for bug in bugs:
            func = bug["function_name"]
            imports.append(f"from {module_path} import {func}")
        imports_str = "\n".join(imports)
        
        # Build test sections for each bug
        bug_sections = []
        test_functions = []
        for i, bug in enumerate(bugs, 1):
            func = bug["function_name"]
            tests = bug.get(f"{test_type}_tests", [])
            bug_desc = bug.get("bug_description", "")  # Changed from bug_variant
            
            # Test cases for this bug
            cases_literal = format_tests(tests)
            
            # Create test function for this bug
            bug_sections.append(
                f"# ===== Bug #{i}: {func} =====\n"
                f"# {bug_desc}\n"
                f"_BUG{i}_TESTS = [\n{cases_literal}\n]\n"
            )
            
            test_functions.append(
                f"def test_bug{i}():\n"
                f"    \"\"\"Test Bug #{i}: {func}\"\"\"\n"
                f"    print(f'\\n===== Testing Bug #{i}: {func} =====')\n"
                f"    passed = crashed = 0\n"
                f"    for j, (args, expected) in enumerate(_BUG{i}_TESTS, 1):\n"
                f"        try:\n"
                f"            result = {func}(*args)\n"
                f"            ok = result == expected\n"
                f"            status = '[PASS]' if ok else '[FAIL]'\n"
                f"            passed += ok\n"
                f"            print(f'  Test {{j}}: {{status}}')\n"
                f"            if not ok:\n"
                f"                print(f'           args     = {{args!r}}')\n"
                f"                print(f'           expected = {{expected}}')\n"
                f"                print(f'           got      = {{result}}')\n"
                f"        except Exception as exc:\n"
                f"            crashed += 1\n"
                f"            print(f'  Test {{j}}: [CRASH] - {{type(exc).__name__}}: {{exc}}')\n"
                f"            print(f'           args = {{args!r}}')\n"
                f"    \n"
                f"    total = len(_BUG{i}_TESTS)\n"
                f"    return passed, crashed, total\n"
            )
        
        # Build main runner
        bug_calls = []
        for i in range(1, len(bugs) + 1):
            bug_calls.append(f"    p{i}, c{i}, t{i} = test_bug{i}()")
        
        total_calc = " + ".join([f"t{i}" for i in range(1, len(bugs) + 1)])
        passed_calc = " + ".join([f"p{i}" for i in range(1, len(bugs) + 1)])
        crashed_calc = " + ".join([f"c{i}" for i in range(1, len(bugs) + 1)])
        
        main_runner = (
            "def run_all_tests():\n"
            "    \"\"\"Run tests for all bugs.\"\"\"\n"
            + "\n".join(bug_calls) + "\n"
            f"    \n"
            f"    total = {total_calc}\n"
            f"    passed = {passed_calc}\n"
            f"    crashed = {crashed_calc}\n"
            f"    broken = total - passed\n"
            f"    \n"
            f"    print(f'\\n=========================================')\n"
            f"    if passed == total:\n"
            f"        print(f'=== ALL TESTS PASSED ({{passed}}/{{total}}) ===')\n"
            f"        return True\n"
            f"    elif crashed == total:\n"
            f"        print(f'=== ALL TESTS CRASHED ({{crashed}}/{{total}}) ===')\n"
            f"        return False\n"
            f"    else:\n"
            f"        print(f'=== {{broken}}/{{total}} tests failed ({{crashed}} crash, {{broken-crashed}} wrong) ===')\n"
            f"        return False\n"
        )
        
        # Combine everything
        return (
            "import sys\n"
            "import os\n\n"
            "# Ensure UTF-8 output on Windows\n"
            "if hasattr(sys.stdout, 'reconfigure'):\n"
            "    sys.stdout.reconfigure(encoding='utf-8')\n\n"
            "# Make the repo root importable\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n\n"
            f"{imports_str}\n\n"
            f"# {test_type.upper()} TEST CASES\n\n"
            + "\n".join(bug_sections) + "\n"
            + "\n".join(test_functions) + "\n"
            + main_runner + "\n"
            "if __name__ == '__main__':\n"
            "    run_all_tests()\n"
        )
    
    # Create public test file
    challenge_run_path = os.path.join(clone_path, "challenge_run.py")
    public_content = _build_multi_bug_test_file(bugs_with_tests, "public")
    with open(challenge_run_path, "w", encoding="utf-8") as f:
        f.write(public_content)
    total_public = sum(len(b["public_tests"]) for b in bugs_with_tests)
    print(f"[deployer] Written: {challenge_run_path}  ({total_public} public tests across {len(bugs_with_tests)} bugs)")
    
    # Create secret test file
    secret_run_path = os.path.join(clone_path, "challenge_run_secret.py")
    secret_content = _build_multi_bug_test_file(bugs_with_tests, "secret")
    with open(secret_run_path, "w", encoding="utf-8") as f:
        f.write(secret_content)
    total_secret = sum(len(b["secret_tests"]) for b in bugs_with_tests)
    print(f"[deployer] Written: {secret_run_path}  ({total_secret} secret tests across {len(bugs_with_tests)} bugs)")
    
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
