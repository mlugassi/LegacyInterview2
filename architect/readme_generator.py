import os

from architect.state import ArchitectState

_LEVEL_NAMES = {1: "Level 1 — Messy Code", 2: "Level 2 — Spaghetti Logic", 3: "Level 3 — Sensitive Code"}

_LEVEL_TIPS = {
    1: "The code has been obfuscated with meaningless variable names and misleading comments. "
       "Focus on understanding what the function is *supposed* to return, then trace the logic carefully.",
    2: "Watch out for nested conditions and variables that look global. "
       "Trace every branch of the logic before concluding where the error is.",
    3: "The bug is a numeric constant (a 'magic number'). "
       "Do NOT try to understand every detail of the formula — focus on identifying which number is wrong "
       "by comparing the actual output to the expected output.",
}


def create_readme(state: ArchitectState) -> ArchitectState:
    level = state["difficulty_level"]
    func = state["function_name"]
    args = state["test_args"]
    expected = state["expected_output"]
    actual = state["actual_output"]
    target_rel = os.path.relpath(state["target_file"], state["clone_path"])
    level_name = _LEVEL_NAMES.get(level, f"Level {level}")
    tip = _LEVEL_TIPS.get(level, "")

    readme_path = os.path.join(state["clone_path"], "STUDENT_README.md")
    content = f"""\
# Legacy Code Challenge — {level_name}

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

Run `python challenge_run.py` at any time to see the current output vs expected.

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
