import os

from architect.state import ArchitectState


def _module_import_path(clone_path: str, target_file: str) -> str:
    """Convert an absolute file path to a dotted module path relative to clone_path."""
    rel = os.path.relpath(target_file, clone_path)          # e.g. utils/math_utils.py
    without_ext = os.path.splitext(rel)[0]                  # e.g. utils/math_utils
    module_path = without_ext.replace(os.sep, ".")          # e.g. utils.math_utils
    return module_path


def deploy_challenge(state: ArchitectState) -> ArchitectState:
    clone_path = state["clone_path"]
    module_path = _module_import_path(clone_path, state["target_file"])
    func = state["function_name"]
    args = state["test_args"]
    expected = state["expected_output"]
    actual = state["actual_output"]

    challenge_run_path = os.path.join(clone_path, "challenge_run.py")
    content = f"""\
import sys
import os

# Make the repo root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from {module_path} import {func}

result = {func}{args}
print(f"EXPECTED: {expected} | ACTUAL: {{result}}")
"""
    with open(challenge_run_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[deployer] Written: {challenge_run_path}")
    return state
