"""
Student Interface — Gradio 5-tab GUI for the Legacy Code Challenge.

Tabs:
  1. 📋 Challenge   — STUDENT_README.md rendered as Markdown
  2. 💻 Code Editor — edit the target file, submit, reset
  3. 🗂️ File Browser — read any .py file in the workspace
  4. 🤖 AI Helper   — progressive hint chatbot (LangGraph hint sub-graph)
  5. 📊 History     — table of all past submissions with scores

Entry point:
    create_interface(workspace_path: str) -> gr.Blocks
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import gradio as gr

from orchestrator.scoring import evaluate_submission
from orchestrator.hint_graph import get_hint


# ── Challenge state loader ────────────────────────────────────────────────────

class ChallengeState:
    """Loads and exposes data from challenge_state.json."""

    def __init__(self, workspace_path: str) -> None:
        self.workspace = Path(workspace_path)
        state_file = self.workspace / "challenge_state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"challenge_state.json not found in {workspace_path}")
        with open(state_file, encoding="utf-8") as f:
            data = json.load(f)

        self.github_url:       str  = data.get("github_url", "")
        self.target_file:      str  = data.get("target_file", "")
        self.original_code:    str  = data.get("original_code", "")
        self.sabotaged_code:   str  = data.get("sabotaged_code", "")
        self.function_name:    str  = data.get("function_name", "")
        self.bug_func_name:    str  = data.get("bug_func_name", "")
        self.difficulty_level: int  = data.get("difficulty_level", 1)

    @property
    def target_path(self) -> Path:
        return self.workspace / self.target_file

    def read_target(self) -> str:
        if self.target_path.exists():
            return self.target_path.read_text(encoding="utf-8")
        return self.sabotaged_code

    def write_target(self, code: str) -> None:
        self.target_path.write_text(code, encoding="utf-8")

    def reset_target(self) -> None:
        self.write_target(self.sabotaged_code)

    def list_py_files(self) -> list[str]:
        files = sorted(self.workspace.rglob("*.py"))
        return [str(f.relative_to(self.workspace)) for f in files]

    def read_py_file(self, rel_path: str) -> str:
        full = self.workspace / rel_path
        if full.exists():
            return full.read_text(encoding="utf-8")
        return f"# File not found: {rel_path}"

    def readme(self) -> str:
        readme_path = self.workspace / "STUDENT_README.md"
        if readme_path.exists():
            return readme_path.read_text(encoding="utf-8")
        return "# Challenge\n\nREADME not found."

    def challenge_info(self) -> dict:
        return {
            "function_name":    self.function_name,
            "bug_func_name":    self.bug_func_name,
            "target_file":      self.target_file,
            "difficulty_level": self.difficulty_level,
        }


# ── Submission log ────────────────────────────────────────────────────────────

class SubmissionLog:
    """Persists submissions to <workspace>/submissions/ as JSON files."""

    def __init__(self, workspace: Path) -> None:
        self.log_dir = workspace / "submissions"
        self.log_dir.mkdir(exist_ok=True)

    def save(self, student_code: str, result: dict, hints_used: int) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        entry = {
            "timestamp":    ts,
            "hints_used":   hints_used,
            "test_score":   result["test_score"],
            "diff_score":   result["diff_score"],
            "hint_penalty": result["hint_penalty"],
            "total_score":  result["total_score"],
            "passed":       result["passed"],
            "total_tests":  result["total_tests"],
            "all_passed":   result["all_passed"],
            "student_code": student_code,
        }
        path = self.log_dir / f"{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

    def history_rows(self) -> list[list]:
        rows = []
        for p in sorted(self.log_dir.glob("*.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    e = json.load(f)
                ts = e.get("timestamp", "?")
                rows.append([
                    ts[:15].replace("_", " "),
                    e.get("total_score", 0),
                    e.get("test_score", 0),
                    e.get("diff_score", 0),
                    e.get("hint_penalty", 0),
                    f"{e.get('passed', 0)}/{e.get('total_tests', 0)}",
                    "Yes" if e.get("all_passed") else "No",
                ])
            except Exception:
                pass
        return rows


# ── Interface factory ─────────────────────────────────────────────────────────

PENALTY_TABLE = [0, 2, 6, 12, 20, 30]


def create_interface(workspace_path: str) -> gr.Blocks:
    """Build and return the Gradio Blocks interface (do not launch here)."""

    cs = ChallengeState(workspace_path)
    log = SubmissionLog(cs.workspace)

    py_files = cs.list_py_files()
    default_file = cs.target_file if cs.target_file in py_files else (py_files[0] if py_files else None)

    with gr.Blocks(title="Legacy Code Challenge", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# 🐛 Legacy Code Challenge\n"
            f"**Repo:** `{cs.github_url}` &nbsp;|&nbsp; "
            f"**Target:** `{cs.target_file}` &nbsp;|&nbsp; "
            f"**Difficulty:** Level {cs.difficulty_level}"
        )

        # Shared state
        hints_used_state       = gr.State(0)
        submission_count_state = gr.State(0)

        # ── Tab 1: Challenge ──────────────────────────────────────────────────
        with gr.Tab("📋 Challenge"):
            gr.Markdown(cs.readme())

        # ── Tab 2: Code Editor ────────────────────────────────────────────────
        with gr.Tab("💻 Code Editor"):
            gr.Markdown(
                f"Edit **`{cs.target_file}`** below, then click **Submit** to evaluate your fix."
            )
            code_editor = gr.Code(
                value=cs.read_target(),
                language="python",
                label=cs.target_file,
                interactive=True,
                lines=35,
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary")
                reset_btn  = gr.Button("↩️ Reset to Challenge Code")

            with gr.Row():
                score_box  = gr.Textbox(label="Score Breakdown", interactive=False, scale=1)
                result_box = gr.Textbox(label="Test Output", interactive=False, lines=12, scale=3)

            gr.Textbox(
                label="Optional: 1-line explanation of your fix",
                placeholder="e.g. Changed '<' to '<=' to fix off-by-one in boundary check",
                lines=2,
            )

            def on_submit(code, hints_used, submit_count):
                cs.write_target(code)
                result = evaluate_submission(
                    workspace_path=workspace_path,
                    student_code=code,
                    original_code=cs.original_code,
                    bug_func_name=cs.bug_func_name,
                    hints_used=hints_used,
                )
                log.save(code, result, hints_used)
                new_count = submit_count + 1

                score_text = (
                    f"Total: {result['total_score']}/100\n"
                    f"  Tests : {result['test_score']}/80 "
                    f"({result['passed']}/{result['total_tests']} passed)\n"
                    f"  Diff  : {result['diff_score']}/20\n"
                    f"  Hints : −{result['hint_penalty']} pts ({hints_used} used)"
                )
                if result["all_passed"]:
                    score_text = "🎉 ALL TESTS PASS!\n\n" + score_text
                if not result["correct_location"] and result["test_score"] < 80:
                    score_text += "\n\n⚠️  You may have edited the wrong function."

                return score_text, result["test_output"], new_count

            def on_reset():
                cs.reset_target()
                return cs.sabotaged_code

            submit_btn.click(
                on_submit,
                inputs=[code_editor, hints_used_state, submission_count_state],
                outputs=[score_box, result_box, submission_count_state],
            )
            reset_btn.click(on_reset, outputs=[code_editor])

        # ── Tab 3: File Browser ───────────────────────────────────────────────
        with gr.Tab("🗂️ File Browser"):
            gr.Markdown("Browse any Python file in the workspace (read-only).")
            file_dropdown = gr.Dropdown(
                choices=py_files,
                label="Select file",
                value=default_file,
            )
            file_viewer = gr.Code(
                value=cs.read_py_file(default_file) if default_file else "",
                language="python",
                label="File contents",
                interactive=False,
                lines=35,
            )

            file_dropdown.change(
                lambda rel: cs.read_py_file(rel),
                inputs=[file_dropdown],
                outputs=[file_viewer],
            )

        # ── Tab 4: AI Helper ──────────────────────────────────────────────────
        with gr.Tab("🤖 AI Helper"):
            gr.Markdown(
                "Ask the AI mentor for hints. "
                "**Each hint costs points** — penalty accumulates: "
                "−2 / −6 / −12 / −20 / −30 pts after 1/2/3/4/5+ hints."
            )
            chatbot = gr.Chatbot(label="AI Mentor", height=420, type="tuples")
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask for a hint…",
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            hint_counter = gr.Markdown("**Hints used:** 0 &nbsp;|&nbsp; **Current penalty:** 0 pts")

            def on_send(message, history, hints_used, submit_count):
                if not message.strip():
                    return history, "", hints_used, f"**Hints used:** {hints_used} &nbsp;|&nbsp; **Current penalty:** {PENALTY_TABLE[min(hints_used, len(PENALTY_TABLE)-1)]} pts"
                response = get_hint(
                    user_message=message,
                    history=history,
                    hints_used=hints_used,
                    submission_attempts=submit_count,
                    challenge_info=cs.challenge_info(),
                )
                history = list(history) + [[message, response]]
                new_hints = hints_used + 1
                penalty = PENALTY_TABLE[min(new_hints, len(PENALTY_TABLE) - 1)]
                counter_md = f"**Hints used:** {new_hints} &nbsp;|&nbsp; **Current penalty:** {penalty} pts"
                return history, "", new_hints, counter_md

            send_btn.click(
                on_send,
                inputs=[chat_input, chatbot, hints_used_state, submission_count_state],
                outputs=[chatbot, chat_input, hints_used_state, hint_counter],
            )
            chat_input.submit(
                on_send,
                inputs=[chat_input, chatbot, hints_used_state, submission_count_state],
                outputs=[chatbot, chat_input, hints_used_state, hint_counter],
            )

        # ── Tab 5: History ────────────────────────────────────────────────────
        with gr.Tab("📊 History"):
            gr.Markdown("All your previous submissions (newest last). Click **Refresh** to update.")
            refresh_btn = gr.Button("🔄 Refresh")
            history_table = gr.Dataframe(
                headers=["Time", "Total", "Tests", "Diff", "Hint−", "Cases", "All Pass?"],
                value=log.history_rows(),
                interactive=False,
            )
            refresh_btn.click(lambda: log.history_rows(), outputs=[history_table])

    return demo
