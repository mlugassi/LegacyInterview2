import os

from architect.state import ArchitectState


def create_readme(state: ArchitectState) -> ArchitectState:
    num_bugs   = state.get("num_bugs", 1)
    clone_path = state["clone_path"]
    repo_name  = os.path.basename(clone_path)
    target_rel = os.path.relpath(state["target_file"], clone_path)

    bugs_str = "1 bug" if num_bugs == 1 else f"{num_bugs} bugs"

    refactoring_note = ""
    if state.get("refactoring_enabled", False):
        refactoring_note = (
            "\n\n> **⚠️ Note:** The code has been intentionally obfuscated with "
            "confusing variable names and structure."
        )

    readme_path = os.path.join(clone_path, "STUDENT_README.md")
    content = f"""\
# Legacy Code Challenge

## Your Mission

You have just joined the team. Your tech lead hands you an urgent ticket:

> **"Something broke in production — the tests are failing and we cannot ship.
> Good luck. The original author left the company."**

Your task: find and fix **{bugs_str}** hidden in `{target_rel}` — without breaking anything else.{refactoring_note}

### ⚠️ Only modify `{target_rel}`. Do not touch any other file.

---

## How to Work

You can edit the code in two ways:

- **Built-in Code Editor** — use the **💻 Code Editor** tab to browse and edit files
  directly in the challenge UI. Click **💾 Save** to save your changes.
- **Your local IDE** — open the file in VS Code, PyCharm, or any editor you prefer.
  Your saved changes are picked up automatically when you submit.

To run the tests at any time, either:

- Go to the **🧪 Test Results** tab in the challenge UI and click **▶ Run Tests**.
- Or To run the tests at any time, by: 
```bash
python {repo_name}\challenge_run.py
```

The test cases are also visible in the **💻 Code Editor** tab — open `{repo_name}\challenge_run.py`
to read them.

---

## Constraints - 
> ### ⚠️ Read carefully — violations will affect your score.


1. **Only fix `{target_rel}`.** Do not change any other file.
2. **No renaming variables or parameters.**
3. **No splitting or merging functions.** Signatures must stay exactly as-is.
4. **No adding external imports.**
5. **Minimal fix only.** At most 1–3 lines changed per bug.
6. **Do not break existing behaviour.**

---

## AI Assistant

On the right side of the challenge you have an **AI Assistant** you can chat with.
It can help you understand the codebase, navigate the challenge UI, explain what a piece
of code does, or point you in the right direction — but it will **not** give you the fix.

> ### **⚠️ Using hints from the AI carries a scoring penalty, so use it wisely.**

---

## Submitting

You can submit at any time using the **🚀 Submit Fix** button.
Running the tests first lets you verify whether you have found and fixed all the bugs.

---

## Scoring

Your submission is evaluated automatically by an AI reviewer:

- **Correctness** — did all the tests pass and were all bugs fixed?
- **Minimality** — did you change only the lines that needed to change?
- **Quality** — does your fix match the expected correct approach?

---

# Good luck. The codebase is yours now.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[readme] Written: {readme_path}")
    return state
