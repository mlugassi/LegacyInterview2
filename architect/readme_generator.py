import os

from architect.state import ArchitectState

def create_readme(state: ArchitectState) -> ArchitectState:
    func = state["function_name"]
    args = state["test_args"]
    expected = state["expected_output"]
    actual = state["actual_output"]
    target_rel = os.path.relpath(state["target_file"], state["clone_path"])
    num_bugs = state.get("num_bugs", 1)
    
    refactoring_note = ""
    if state.get("refactoring_enabled", False):
        refactoring_note = "\n\n**Note:** The code has been intentionally obfuscated with confusing variable names and structure."
    
    # Calculate max lines allowed based on number of bugs
    max_lines = num_bugs * 3  # Up to 3 lines per bug

    readme_path = os.path.join(state["clone_path"], "STUDENT_README.md")
    content = f"""\
# Legacy Code Challenge

## Your Mission

You have just joined the team. Your tech lead hands you an urgent ticket:

> **"Something broke in production. The function `{func}` in `{target_rel}` is returning wrong
> results. We cannot ship until it's fixed. Good luck — the original author left the company."**

The codebase you've inherited is messy, poorly documented, and nobody fully understands it.
Your job is to find the bug and fix it — **without breaking anything else**.

---

## Bug Report

| | |
|---|---|
| **Affected function** | `{func}` |
| **File** | `{target_rel}` |
| **Number of bugs** | **{num_bugs}** |
| **Test input** | `{func}{args}` |
| **Expected output** | `{expected}` |
| **Actual output** | `{actual}` |

Run `python challenge_run.py` at any time to test your solution against public test cases.
Run `python challenge_run_secret.py` to test against ALL tests (public + secret).{refactoring_note}

---

## Constraints (READ CAREFULLY — violations = automatic disqualification)

1. **No renaming variables or parameters.** Even if the names are confusing, changing them is forbidden.
2. **No splitting or merging functions.** The function signature must stay exactly as-is.
3. **No adding external imports.** You may only use what is already imported.
4. **Fix must be minimal.** You may add, remove, or change at most {max_lines} lines total (up to 3 lines per bug).
5. **All existing functionality must remain intact.** Do not change code you do not need to touch.

---

## Submission

When your fix makes `challenge_run.py` print matching EXPECTED and ACTUAL values:

1. Submit the modified `{target_rel}` file.
2. Include a **single sentence** explaining *why* your specific change fixes the bug(s) with minimum risk.

---

## Evaluation Criteria

| Criterion | Points |
|---|---|
| All unit tests pass (existing functionality preserved) | 50 |
| Fix is minimal (up to {max_lines} lines, correct location) | 30 |
| Technical explanation is accurate and concise | 20 |

Good luck. The codebase is yours now.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[readme] Written: {readme_path}")
    return state
