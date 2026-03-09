import os

from architect.state import ArchitectState

def create_readme(state: ArchitectState) -> ArchitectState:
    nesting = state.get("nesting_level", 3)
    func = state["function_name"]
    args = state["test_args"]
    expected = state["expected_output"]
    actual = state["actual_output"]
    target_rel = os.path.relpath(state["target_file"], state["clone_path"])
    
    # Generate hint based on nesting level
    if nesting <= 2:
        tip = (f"The bug might be directly in `{func}` or one level deep in a helper function it calls. "
               f"Start by reading `{func}` carefully, then check any functions it uses.")
    elif nesting <= 4:
        tip = (f"The bug is hidden {nesting} levels deep in the call chain starting from `{func}`. "
               f"You'll need to trace through multiple function calls to find it. "
               f"Don't get lost in the details — focus on following the data flow.")
    else:
        tip = (f"This is a deep rabbit hole — the bug is {nesting} levels down from `{func}`. "
               f"The code has been inflated and the bug is buried in a deeply nested helper function. "
               f"Use the test cases to guide you to which branch of logic is broken.")
    
    refactoring_note = ""
    if state.get("refactoring_enabled", False):
        refactoring_note = "\n\n**Note:** The code has been intentionally obfuscated with confusing variable names and structure."

    readme_path = os.path.join(state["clone_path"], "STUDENT_README.md")
    content = f"""\
# Legacy Code Challenge — Nesting Level {nesting}

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
4. **Fix must be 1–3 lines.** You may add, remove, or change at most 3 lines.
5. **All existing functionality must remain intact.** Do not change code you do not need to touch.

---

## Hint

{tip}

---

## Submission

When your fix makes `challenge_run.py` print matching EXPECTED and ACTUAL values:

1. Submit the modified `{target_rel}` file.
2. Include a **single sentence** explaining *why* your specific change fixes the bug with minimum risk.

---

## Evaluation Criteria

| Criterion | Points |
|---|---|
| All unit tests pass (existing functionality preserved) | 50 |
| Fix is minimal (1–3 lines, correct location) | 30 |
| Technical explanation is accurate and concise | 20 |

Good luck. The codebase is yours now.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[readme] Written: {readme_path}")
    return state
