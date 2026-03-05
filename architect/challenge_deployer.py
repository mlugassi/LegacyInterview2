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
    test_cases  = state.get("test_cases") or []

    # Fallback to the single legacy case if test_cases wasn't populated
    if not test_cases:
        test_cases = [{"args": state["test_args"], "expected": state["expected_output"]}]

    # Build the _TEST_CASES literal — each entry is (args_tuple, expected_value)
    case_lines = []
    for tc in test_cases:
        args_repr = tc["args"]          # already a Python tuple literal, e.g. "('hello',)"
        exp_repr  = tc["expected"]      # already a repr string — use directly as Python literal
        case_lines.append(f"    ({args_repr}, {exp_repr}),")
    cases_literal = "\n".join(case_lines)

    challenge_run_path = os.path.join(clone_path, "challenge_run.py")
    content = (
        "import sys\n"
        "import os\n\n"
        "# Make the repo root importable\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n\n"
        f"from {module_path} import {func}\n\n"
        "# (args_tuple, correct_expected_output)\n"
        f"_TEST_CASES = [\n{cases_literal}\n]\n\n"
        "passed = crashed = 0\n"
        "for i, (args, expected) in enumerate(_TEST_CASES, 1):\n"
        "    try:\n"
        f"        result = {func}(*args)\n"
        "        ok = result == expected\n"
        "        status = 'PASS' if ok else 'FAIL'\n"
        "        passed += ok\n"
        "        print(f'  Case {i}: {status}')\n"
        "        if not ok:\n"
        "            print(f'           args     = {args!r}')\n"
        "            print(f'           expected = {expected}')\n"
        "            print(f'           got      = {result}')\n"
        "    except Exception as exc:\n"
        "        crashed += 1\n"
        "        print(f'  Case {i}: CRASH — {type(exc).__name__}: {exc}')\n"
        "        print(f'           args = {args!r}')\n\n"
        "total = len(_TEST_CASES)\n"
        "broken = total - passed\n"
        "if passed == total:\n"
        "    verdict = 'ALL PASS — bug is fixed!'\n"
        "elif crashed == total:\n"
        "    verdict = f'ALL CRASH ({crashed}/{total}) — function throws exceptions on every input'\n"
        "else:\n"
        "    verdict = f'FAILED — {broken}/{total} cases wrong ({crashed} crash, {broken-crashed} wrong value)'\n"
        "print(f'\\n{verdict}')\n"
    )

    with open(challenge_run_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[deployer] Written: {challenge_run_path}  ({len(test_cases)} test cases)")
    return state
