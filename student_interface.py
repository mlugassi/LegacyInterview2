"""
Student Interface — Gradio GUI for the Legacy Code Challenge.

Two-page flow:
  Page 1 (Setup):     student name, GitHub URL, difficulty level, num-bugs → Start
  Page 2 (Challenge): README, Code Editor, Test runner + AI Assistant sidebar
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
        # Normalise to forward slashes so comparisons work cross-platform
        self.target_file:      str  = Path(data.get("target_file", "")).as_posix()
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
        return [f.relative_to(self.workspace).as_posix() for f in files]

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


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(github_url: str, difficulty_level: int, num_bugs: int) -> str:
    """Invoke the architect pipeline and return the workspace path."""
    from architect.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "github_url":       github_url,
        "difficulty_level": difficulty_level,
        "num_bugs":         num_bugs,
        "clone_path":       "",
        "target_file":      "",
        "original_code":    "",
        "sabotaged_code":   "",
        "function_name":    "",
        "test_args":        "",
        "expected_output":  "",
        "actual_output":    "",
        "bug_description":  "",
        "challenge_summary": "",
        "test_cases":       [],
        "candidate_files":  [],
        "bug_func_name":    "",
        "bug_func_source":  "",
    })

    workspace_path = result["clone_path"]

    # Persist challenge_state.json
    challenge_state = {
        "github_url":       github_url,
        "workspace_path":   workspace_path,
        "target_file":      result.get("target_file",      ""),
        "original_code":    result.get("original_code",    ""),
        "sabotaged_code":   result.get("sabotaged_code",   ""),
        "function_name":    result.get("function_name",    ""),
        "bug_func_name":    result.get("bug_func_name",    ""),
        "bug_func_source":  result.get("bug_func_source",  ""),
        "test_cases":       result.get("test_cases",       []),
        "difficulty_level": difficulty_level,
        "bug_description":  result.get("bug_description",  ""),
    }
    state_path = Path(workspace_path) / "challenge_state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(challenge_state, f, indent=2, ensure_ascii=False)

    return workspace_path


# ── Helpers ───────────────────────────────────────────────────────────────────

PENALTY_TABLE = [0, 2, 6, 12, 20, 30]

# Sets dark mode as the default on first visit.
# The built-in Gradio toggle button (in the footer) lets users switch themes.
_JS = """
() => {
    if (!localStorage.getItem('gradio-theme')) {
        localStorage.setItem('gradio-theme', 'dark');
    }
}
"""

_CSS = """
/* Keep footer visible — it contains the built-in dark/light toggle button */
footer svg { display: inline !important; }
.gradio-container { max-width: 100% !important; }

/* Setup page — centred card */
.setup-card { max-width: 700px; margin: 40px auto !important; }

/* Code editor — scrollable */
#code-editor .cm-scroller { overflow-y: auto !important; max-height: 62vh !important; }
#code-editor .cm-editor   { max-height: 62vh !important; }

/* Chatbot */
#ai-chatbot .wrap { height: 58vh !important; overflow-y: auto !important; }

/* Tab content */
.left-tabs .tabitem { overflow-y: auto !important; max-height: 80vh !important; }
"""


def _hint_md(hints_used: int, penalty: int) -> str:
    return (
        f"🤖 **Assistant** &nbsp;|&nbsp; "
        f"Hints used: **{hints_used}** &nbsp;|&nbsp; Penalty: **{penalty} pts** &nbsp; "
        f"_(−2 / −6 / −12 / −20 / −30)_"
    )


def _colorise_test_output(raw: str) -> str:
    """Wrap test output lines in coloured HTML spans."""
    html_lines = []
    for line in raw.split("\n"):
        if ": PASS" in line or "ALL PASS" in line:
            html_lines.append(f'<span style="color:#22c55e;font-weight:bold;">{line}</span>')
        elif ": FAIL" in line or ": CRASH" in line or "FAILED" in line:
            html_lines.append(f'<span style="color:#ef4444;font-weight:bold;">{line}</span>')
        else:
            html_lines.append(f'<span style="color:#d4d4d4;">{line}</span>')
    return (
        '<pre style="font-family:monospace;padding:12px;background:#1e1e1e;'
        'color:#d4d4d4;border-radius:8px;overflow-y:auto;max-height:60dvh;'
        'white-space:pre-wrap;">'
        + "<br>".join(html_lines)
        + "</pre>"
    )


# ── Full two-page interface ───────────────────────────────────────────────────

def create_full_interface() -> gr.Blocks:
    """
    Returns a Gradio app with two logical pages:
      Page 1: Setup form (name / URL / level / bugs)
      Page 2: Challenge interface (shown after pipeline finishes)
    """

    with gr.Blocks(title="Legacy Code Challenge") as demo:

        # Shared state
        workspace_state        = gr.State("")
        hints_used_state       = gr.State(0)
        submission_count_state = gr.State(0)

        # ════════════════════════════════════════════════════════════════════
        # PAGE 1 — Setup
        # ════════════════════════════════════════════════════════════════════
        with gr.Column(visible=True, elem_classes=["setup-card"]) as setup_page:
            gr.Markdown(
                "# 🐛 Legacy Code Challenge\n"
                "Fill in the details below, then click **Start Challenge**."
            )

            name_box = gr.Textbox(
                label="Your Name",
                placeholder="e.g. Alice Smith",
            )
            url_box = gr.Textbox(
                label="GitHub Repository URL",
                placeholder="https://github.com/mahmoud/boltons",
            )
            with gr.Row():
                level_radio = gr.Radio(
                    choices=["1 — Messy Code", "2 — Spaghetti Logic", "3 — Sensitive Code"],
                    value="1 — Messy Code",
                    label="Difficulty Level",
                )
                bugs_slider = gr.Slider(
                    minimum=1, maximum=5, value=1, step=1,
                    label="Number of Bugs to Inject",
                )

            start_btn  = gr.Button("🚀 Start Challenge", variant="primary", size="lg")
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False,
                lines=3,
            )

        # ════════════════════════════════════════════════════════════════════
        # PAGE 2 — Challenge (hidden until pipeline finishes)
        # ════════════════════════════════════════════════════════════════════
        with gr.Column(visible=False) as challenge_page:

            header_md = gr.Markdown("### 🐛 Legacy Code Challenge")

            with gr.Row(elem_classes=["main-row"]):

                # ── Left column: tabs ─────────────────────────────────────
                with gr.Column(scale=3, elem_classes=["left-col"]):
                    with gr.Tabs(elem_classes=["left-tabs"]):

                        with gr.Tab("📋 Challenge"):
                            challenge_readme = gr.Markdown("")

                        with gr.Tab("💻 Code Editor"):
                            file_dropdown = gr.Dropdown(
                                choices=[],
                                label="File",
                                info="Select a file to view; only the target file is editable.",
                            )
                            code_editor = gr.Code(
                                value="",
                                language="python",
                                label="",
                                interactive=True,
                                lines=30,
                                elem_id="code-editor",
                            )

                        with gr.Tab("🧪 Test"):
                            gr.Markdown("Run `challenge_run.py` against your current saved code.")
                            run_tests_btn = gr.Button("▶ Run Tests", variant="primary")
                            test_output   = gr.HTML(
                                "<p style='color:#888;font-family:monospace;'>"
                                "Click 'Run Tests' to execute…</p>"
                            )

                # ── Right column: AI assistant + submit ───────────────────
                with gr.Column(scale=2, elem_classes=["right-col"]):
                    hint_counter = gr.Markdown(_hint_md(0, 0))
                    chatbot = gr.Chatbot(label="AI Assistant", elem_id="ai-chatbot")
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask the AI assistant for a hint…",
                            show_label=False,
                            scale=5,
                        )
                        send_btn = gr.Button("Send", variant="secondary", scale=1)

                    gr.Markdown("---")

                    with gr.Accordion("📊 Results", open=True, visible=False) as results_accordion:
                        score_box  = gr.Textbox(label="Score",       interactive=False, lines=2)
                        result_box = gr.Textbox(label="Test Output", interactive=False, lines=5)

                    with gr.Row():
                        submit_btn = gr.Button("🚀 Submit Fix", variant="primary", scale=2)
                        reset_btn  = gr.Button("↩️ Reset", scale=1)

        # ── Setup callback ─────────────────────────────────────────────────

        def on_start(name, url, level_str, num_bugs):
            level = int(str(level_str).strip()[0])

            # Phase 1: show loading indicator
            yield (
                gr.update(visible=True,
                          value="⏳ Cloning repository and generating challenge…"
                                " this may take 1–2 minutes."),  # status_box
                gr.update(interactive=False),   # start_btn
                gr.update(),                    # setup_page
                gr.update(),                    # challenge_page
                gr.update(),                    # header_md
                gr.update(),                    # challenge_readme
                gr.update(),                    # file_dropdown
                gr.update(),                    # code_editor
                "",                             # workspace_state
            )

            try:
                workspace_path = _run_pipeline(url.strip(), level, int(num_bugs))
                cs       = ChallengeState(workspace_path)
                py_files = cs.list_py_files()
                default  = cs.target_file if cs.target_file in py_files else (py_files[0] if py_files else "")
                suffix   = f" &nbsp;|&nbsp; {name.strip()}" if name.strip() else ""

                # Phase 2: switch to challenge page
                yield (
                    gr.update(visible=False),   # status_box
                    gr.update(interactive=True), # start_btn
                    gr.update(visible=False),    # setup_page
                    gr.update(visible=True),     # challenge_page
                    gr.update(value=f"### 🐛 Legacy Code Challenge &nbsp;|&nbsp; Level {level}{suffix}"),
                    gr.update(value=cs.readme()),
                    gr.update(choices=py_files, value=default),
                    gr.update(value=cs.read_target(), label=cs.target_file),
                    workspace_path,             # workspace_state
                )

            except Exception as exc:
                yield (
                    gr.update(visible=True, value=f"❌ Error: {exc}"),
                    gr.update(interactive=True),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "",
                )

        start_btn.click(
            on_start,
            inputs=[name_box, url_box, level_radio, bugs_slider],
            outputs=[
                status_box, start_btn,
                setup_page, challenge_page,
                header_md, challenge_readme,
                file_dropdown, code_editor,
                workspace_state,
            ],
        )

        # ── Challenge callbacks ────────────────────────────────────────────

        def on_file_select(rel_path, workspace_path):
            if not workspace_path:
                return gr.update()
            cs = ChallengeState(workspace_path)
            content   = cs.read_py_file(rel_path)
            is_target = (rel_path == cs.target_file)
            return gr.update(value=content, interactive=is_target, label=rel_path)

        def on_run_tests(workspace_path):
            if not workspace_path:
                return "<p style='color:#888'>No challenge loaded.</p>"
            from orchestrator.scoring import run_tests
            r   = run_tests(workspace_path)
            raw = r["output"] if r["output"] else "No output."
            return _colorise_test_output(raw)

        def on_submit(code, hints_used, submit_count, workspace_path):
            if not workspace_path:
                return gr.update(), "No challenge loaded.", "", submit_count
            cs  = ChallengeState(workspace_path)
            log = SubmissionLog(cs.workspace)
            cs.write_target(code)
            result = evaluate_submission(
                workspace_path=workspace_path,
                student_code=code,
                original_code=cs.original_code,
                bug_func_name=cs.bug_func_name,
                hints_used=hints_used,
            )
            log.save(code, result, hints_used)
            new_count  = submit_count + 1
            score_text = (
                f"Total: {result['total_score']}/100  |  "
                f"Tests: {result['test_score']}/80 ({result['passed']}/{result['total_tests']} passed)  |  "
                f"Diff: {result['diff_score']}/20  |  "
                f"Hints: −{result['hint_penalty']} pts"
            )
            if result["all_passed"]:
                score_text = "🎉 ALL TESTS PASS!  " + score_text
            if not result["correct_location"] and result["test_score"] < 80:
                score_text += "  ⚠️ Wrong function?"
            return (
                gr.Accordion(open=True, visible=True),
                score_text,
                result["test_output"],
                new_count,
            )

        def on_reset(workspace_path):
            if not workspace_path:
                return ""
            cs = ChallengeState(workspace_path)
            cs.reset_target()
            return cs.sabotaged_code

        def on_send(message, history, hints_used, submit_count, workspace_path):
            if not message.strip():
                penalty = PENALTY_TABLE[min(hints_used, len(PENALTY_TABLE) - 1)]
                return history, "", hints_used, _hint_md(hints_used, penalty)
            if not workspace_path:
                return history + [[message, "No challenge loaded yet."]], "", hints_used, _hint_md(hints_used, 0)
            cs = ChallengeState(workspace_path)
            response = get_hint(
                user_message=message,
                history=history,
                hints_used=hints_used,
                submission_attempts=submit_count,
                challenge_info=cs.challenge_info(),
            )
            history   = list(history) + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": response},
            ]
            new_hints = hints_used + 1
            penalty   = PENALTY_TABLE[min(new_hints, len(PENALTY_TABLE) - 1)]
            return history, "", new_hints, _hint_md(new_hints, penalty)

        file_dropdown.change(
            on_file_select,
            inputs=[file_dropdown, workspace_state],
            outputs=[code_editor],
        )
        run_tests_btn.click(on_run_tests, inputs=[workspace_state], outputs=[test_output])
        submit_btn.click(
            on_submit,
            inputs=[code_editor, hints_used_state, submission_count_state, workspace_state],
            outputs=[results_accordion, score_box, result_box, submission_count_state],
        )
        reset_btn.click(on_reset, inputs=[workspace_state], outputs=[code_editor])
        send_btn.click(
            on_send,
            inputs=[chat_input, chatbot, hints_used_state, submission_count_state, workspace_state],
            outputs=[chatbot, chat_input, hints_used_state, hint_counter],
        )
        chat_input.submit(
            on_send,
            inputs=[chat_input, chatbot, hints_used_state, submission_count_state, workspace_state],
            outputs=[chatbot, chat_input, hints_used_state, hint_counter],
        )

    return demo


# ── Legacy entry point (used by orchestrator) ─────────────────────────────────

def create_interface(workspace_path: str) -> gr.Blocks:
    """Build the challenge-only interface for a pre-existing workspace."""

    cs  = ChallengeState(workspace_path)
    log = SubmissionLog(cs.workspace)

    py_files     = cs.list_py_files()
    default_file = cs.target_file if cs.target_file in py_files else (py_files[0] if py_files else None)

    with gr.Blocks(title="Legacy Code Challenge") as demo:

        hints_used_state       = gr.State(0)
        submission_count_state = gr.State(0)

        gr.Markdown(
            f"### 🐛 Legacy Code Challenge &nbsp;|&nbsp; Level {cs.difficulty_level}",
            elem_classes=["app-header"],
        )

        with gr.Row(elem_classes=["main-row"]):
            with gr.Column(scale=3, elem_classes=["left-col"]):
                with gr.Tabs(elem_classes=["left-tabs"]):
                    with gr.Tab("📋 Challenge"):
                        gr.Markdown(cs.readme())
                    with gr.Tab("💻 Code Editor"):
                        file_dropdown = gr.Dropdown(choices=py_files, value=default_file, label="File")
                        code_editor   = gr.Code(
                            value=cs.read_target(), language="python", label=cs.target_file,
                            interactive=True, lines=30, elem_id="code-editor",
                        )
                    with gr.Tab("🧪 Test"):
                        gr.Markdown("Run `challenge_run.py` against your current saved code.")
                        run_tests_btn = gr.Button("▶ Run Tests", variant="primary")
                        test_output   = gr.HTML(
                            "<p style='color:#888;font-family:monospace;'>Click 'Run Tests'…</p>"
                        )

            with gr.Column(scale=2, elem_classes=["right-col"]):
                hint_counter = gr.Markdown(_hint_md(0, 0))
                chatbot = gr.Chatbot(label="AI Assistant", elem_id="ai-chatbot")
                with gr.Row():
                    chat_input = gr.Textbox(placeholder="Ask the AI assistant for a hint…", show_label=False, scale=5)
                    send_btn   = gr.Button("Send", variant="secondary", scale=1)
                gr.Markdown("---")
                with gr.Accordion("📊 Results", open=True, visible=False) as results_accordion:
                    score_box  = gr.Textbox(label="Score",       interactive=False, lines=2)
                    result_box = gr.Textbox(label="Test Output", interactive=False, lines=5)
                with gr.Row():
                    submit_btn = gr.Button("🚀 Submit Fix", variant="primary", scale=2)
                    reset_btn  = gr.Button("↩️ Reset", scale=1)

        def on_file_select(rel_path):
            content   = cs.read_py_file(rel_path)
            is_target = (rel_path == cs.target_file)
            return gr.update(value=content, interactive=is_target, label=rel_path)

        def on_run_tests():
            from orchestrator.scoring import run_tests
            r   = run_tests(workspace_path)
            raw = r["output"] if r["output"] else "No output."
            return _colorise_test_output(raw)

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
            new_count  = submit_count + 1
            score_text = (
                f"Total: {result['total_score']}/100  |  "
                f"Tests: {result['test_score']}/80 ({result['passed']}/{result['total_tests']} passed)  |  "
                f"Diff: {result['diff_score']}/20  |  Hints: −{result['hint_penalty']} pts"
            )
            if result["all_passed"]:
                score_text = "🎉 ALL TESTS PASS!  " + score_text
            if not result["correct_location"] and result["test_score"] < 80:
                score_text += "  ⚠️ Wrong function?"
            return gr.Accordion(open=True, visible=True), score_text, result["test_output"], new_count

        def on_reset():
            cs.reset_target()
            return cs.sabotaged_code

        def on_send(message, history, hints_used, submit_count):
            if not message.strip():
                penalty = PENALTY_TABLE[min(hints_used, len(PENALTY_TABLE) - 1)]
                return history, "", hints_used, _hint_md(hints_used, penalty)
            response  = get_hint(
                user_message=message, history=history,
                hints_used=hints_used, submission_attempts=submit_count,
                challenge_info=cs.challenge_info(),
            )
            history   = list(history) + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": response},
            ]
            new_hints = hints_used + 1
            penalty   = PENALTY_TABLE[min(new_hints, len(PENALTY_TABLE) - 1)]
            return history, "", new_hints, _hint_md(new_hints, penalty)

        file_dropdown.change(on_file_select, inputs=[file_dropdown], outputs=[code_editor])
        run_tests_btn.click(on_run_tests, outputs=[test_output])
        submit_btn.click(
            on_submit,
            inputs=[code_editor, hints_used_state, submission_count_state],
            outputs=[results_accordion, score_box, result_box, submission_count_state],
        )
        reset_btn.click(on_reset, outputs=[code_editor])
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

    return demo
