"""
Student Interface — Gradio GUI for the Legacy Code Challenge.

Two-page flow:
  Page 1 (Setup):     student name, GitHub URL, nesting level, num-bugs, options → Start
  Page 2 (Challenge): README, Code Editor, Test runner + AI Assistant sidebar
  Page 3 (Results):   Score breakdown, diffs, test results, hints review
"""

from __future__ import annotations

import difflib
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
        self.nesting_level:    int  = data.get("nesting_level", 3)
        self.refactoring_enabled: bool = data.get("refactoring_enabled", False)
        self.debug_mode:       bool = data.get("debug_mode", False)

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
            "function_name":       self.function_name,
            "bug_func_name":       self.bug_func_name,
            "target_file":         self.target_file,
            "nesting_level":       self.nesting_level,
            "refactoring_enabled": self.refactoring_enabled,
            "debug_mode":          self.debug_mode,
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

def _run_pipeline(github_url: str, nesting_level: int, num_bugs: int, 
                  refactoring_enabled: bool = False, debug_mode: bool = False) -> str:
    """Invoke the architect pipeline and return the workspace path."""
    from architect.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "github_url":          github_url,
        "nesting_level":       nesting_level,
        "refactoring_enabled": refactoring_enabled,
        "debug_mode":          debug_mode,
        "num_bugs":            num_bugs,
        "clone_path":          "",
        "target_file":         "",
        "original_code":       "",
        "sabotaged_code":      "",
        "function_name":       "",
        "test_args":           "",
        "expected_output":     "",
        "actual_output":       "",
        "bug_description":     "",
        "detailed_explanation": "",
        "challenge_summary":   "",
        "test_cases":          [],
        "public_tests":        [],
        "secret_tests":        [],
        "candidate_files":     [],
        "bug_func_name":       "",
        "bug_func_source":     "",
        "call_chain":          {},
    })

    workspace_path = result["clone_path"]

    # Persist challenge_state.json
    challenge_state = {
        "github_url":          github_url,
        "workspace_path":      workspace_path,
        "target_file":         result.get("target_file",      ""),
        "original_code":       result.get("original_code",    ""),
        "sabotaged_code":      result.get("sabotaged_code",   ""),
        "function_name":       result.get("function_name",    ""),
        "bug_func_name":       result.get("bug_func_name",    ""),
        "bug_func_source":     result.get("bug_func_source",  ""),
        "test_cases":          result.get("test_cases",       []),
        "public_tests":        result.get("public_tests",     []),
        "secret_tests":        result.get("secret_tests",     []),
        "nesting_level":       nesting_level,
        "refactoring_enabled": refactoring_enabled,
        "debug_mode":          debug_mode,
        "bug_description":     result.get("bug_description",  ""),
    }
    state_path = Path(workspace_path) / "challenge_state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(challenge_state, f, indent=2, ensure_ascii=False)

    return workspace_path


# ── Helpers ───────────────────────────────────────────────────────────────────

PENALTY_TABLE = [0, 2, 6, 12, 20, 30]

# Sets dark mode as the default on first visit and defines window.startTimer.
def _make_js(timer_minutes: int = 0) -> str:
    auto_start = (
        f"\n    setTimeout(() => {{ window.startTimer({timer_minutes}); }}, 500);"
        if timer_minutes > 0 else ""
    )
    return """() => {
    if (!localStorage.getItem('gradio-theme')) {
        localStorage.setItem('gradio-theme', 'dark');
    }
    window.startTimer = function(minutes) {
        if (!minutes || minutes <= 0) return;
        if (window._timerInterval) clearInterval(window._timerInterval);
        const endTime = Date.now() + minutes * 60 * 1000;
        function tick() {
            const remaining = Math.max(0, endTime - Date.now());
            const m = Math.floor(remaining / 60000);
            const s = Math.floor((remaining % 60000) / 1000);
            const el = document.getElementById('challenge-timer');
            if (!el) return;
            if (remaining === 0) {
                el.innerHTML = '<span style="color:#ef4444;font-weight:bold;font-size:1.1em;">⏱️ TIME\\'S UP!</span>';
                clearInterval(window._timerInterval);
            } else {
                const color = remaining < 300000 ? '#ef4444' : (remaining < 600000 ? '#f59e0b' : '#22c55e');
                el.innerHTML = '⏱️ <span style="font-weight:bold;font-size:1.1em;color:' + color + ';">'
                    + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0') + '</span>';
            }
        }
        tick();
        window._timerInterval = setInterval(tick, 1000);
    };""" + auto_start + "\n}"


_JS = _make_js(0)

_CSS = """
/* Keep footer visible */
footer svg { display: inline !important; }
.gradio-container { max-width: 100% !important; }

/* Setup page — centred card */
.setup-card { max-width: 700px; margin: 40px auto !important; }

/* Code editor — scrollable */
#code-editor .cm-scroller { overflow-y: auto !important; max-height: 60vh !important; }
#code-editor .cm-editor   { max-height: 60vh !important; }

/* Chatbot */
#ai-chatbot .wrap { height: 58vh !important; overflow-y: auto !important; }

/* Tab content */
.left-tabs .tabitem { overflow-y: auto !important; max-height: 80vh !important; }

/* Results diff blocks */
.diff-block { font-family: monospace; font-size: 0.85em; padding: 12px;
              border-radius: 8px; overflow-y: auto; max-height: 45vh;
              white-space: pre; background: #1e1e1e; color: #d4d4d4; }

/* Challenge timer — right-aligned in header row */
#challenge-timer { text-align: right; padding: 6px 8px; font-family: monospace; min-width: 120px; }
.header-row { align-items: center !important; }
"""

_SEARCH_JS = """(query) => {
  if (!query) return query;
  const st = window._searchState;
  if (st && st.query === query && st.matches.length > 0) {
    st.index = (st.index + 1) % st.matches.length;
    st.matches[st.index].scrollIntoView({behavior:'smooth', block:'center'});
  } else {
    const matches = [];
    document.querySelectorAll('#code-editor .cm-line').forEach(l => {
      if (l.textContent.includes(query)) matches.push(l);
    });
    window._searchState = {query, index: 0, matches};
    if (matches.length) matches[0].scrollIntoView({behavior:'smooth', block:'center'});
  }
  return query;
}"""

_SEARCH_NEXT_JS = """(query) => {
  const st = window._searchState;
  if (!st || !st.matches.length) return query;
  st.index = (st.index + 1) % st.matches.length;
  st.matches[st.index].scrollIntoView({behavior:'smooth', block:'center'});
  return query;
}"""

_SEARCH_PREV_JS = """(query) => {
  const st = window._searchState;
  if (!st || !st.matches.length) return query;
  st.index = (st.index - 1 + st.matches.length) % st.matches.length;
  st.matches[st.index].scrollIntoView({behavior:'smooth', block:'center'});
  return query;
}"""


def _hint_md(hints_used: int, penalty: int) -> str:
    return (
        f"🤖 **AI Assistant** &nbsp;|&nbsp; "
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


def _normalize(code: str) -> str:
    """Strip trailing whitespace per line and normalize line endings."""
    return "\n".join(line.rstrip() for line in code.replace("\r\n", "\n").replace("\r", "\n").splitlines())


def _diff_html(a: str, b: str, from_label: str, to_label: str) -> str:
    """Generate coloured unified-diff HTML between two code strings."""
    a = _normalize(a)
    b = _normalize(b)
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    diff = list(difflib.unified_diff(a_lines, b_lines,
                                     fromfile=from_label, tofile=to_label, lineterm=""))
    if not diff:
        return '<div class="diff-block" style="color:#22c55e;">No changes detected.</div>'

    html_lines = []
    for line in diff:
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if line.startswith("+++") or line.startswith("---"):
            html_lines.append(f'<span style="color:#888;font-style:italic;">{escaped}</span>')
        elif line.startswith("@@"):
            html_lines.append(f'<span style="color:#60a5fa;">{escaped}</span>')
        elif line.startswith("+"):
            html_lines.append(f'<span style="color:#22c55e;background:#052e16;">{escaped}</span>')
        elif line.startswith("-"):
            html_lines.append(f'<span style="color:#ef4444;background:#2d0a0a;">{escaped}</span>')
        else:
            html_lines.append(f'<span style="color:#d4d4d4;">{escaped}</span>')

    return (
        '<div class="diff-block">'
        + "\n".join(html_lines)
        + "</div>"
    )


def _extract_function_source(code: str, func_name: str) -> str:
    """Extract a named function's source lines from code using ast."""
    import ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    lines = code.splitlines()
                    return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    except Exception:
        pass
    return ""


def _expected_fix_diff_html(cs: "ChallengeState") -> str:
    """Diff only the bug function between sabotaged and original code."""
    func = cs.bug_func_name
    if func:
        sabotaged_func = _extract_function_source(cs.sabotaged_code, func)
        original_func  = _extract_function_source(cs.original_code,  func)
        if sabotaged_func and original_func:
            return _diff_html(sabotaged_func, original_func,
                              f"{func} — buggy", f"{func} — original")
    return _diff_html(cs.sabotaged_code, cs.original_code,
                      "buggy (received)", "correct original")


def _hints_html(history: list) -> str:
    """Render the hint conversation history as HTML."""
    if not history:
        return "<p style='color:#888;font-style:italic;'>No hints were used.</p>"

    parts = []
    for entry in history:
        if isinstance(entry, dict):
            role    = entry.get("role", "")
            content = entry.get("content", "")
        else:
            role, content = ("user", entry[0]) if entry[0] else ("assistant", entry[1])

        escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if role == "user":
            parts.append(
                f'<div style="margin:6px 0;padding:8px 12px;background:#1e3a5f;'
                f'border-radius:6px;color:#93c5fd;">'
                f'<strong>You:</strong> {escaped}</div>'
            )
        elif role == "assistant":
            parts.append(
                f'<div style="margin:6px 0;padding:8px 12px;background:#1a2e1a;'
                f'border-radius:6px;color:#86efac;">'
                f'<strong>AI:</strong> {escaped}</div>'
            )

    return (
        '<div style="max-height:50vh;overflow-y:auto;padding:4px;">'
        + "".join(parts)
        + "</div>"
    )


def _score_summary_html(result: dict) -> str:
    """Build a prominent score summary block."""
    total   = result["total_score"]
    tests   = result["test_score"]
    diff    = result["diff_score"]
    penalty = result["hint_penalty"]
    passed  = result["passed"]
    ttl     = result["total_tests"]
    all_ok  = result["all_passed"]

    color   = "#22c55e" if all_ok else ("#f59e0b" if total >= 50 else "#ef4444")
    badge   = "🎉 ALL TESTS PASS!" if all_ok else ("⚠️ Partial" if total > 0 else "❌ Failed")

    return f"""
<div style="padding:20px;border-radius:12px;background:#1e1e1e;border:2px solid {color};">
  <div style="font-size:2.5em;font-weight:bold;color:{color};text-align:center;">{total}/100</div>
  <div style="text-align:center;color:{color};font-size:1.1em;margin-bottom:16px;">{badge}</div>
  <table style="width:100%;border-collapse:collapse;font-family:monospace;">
    <tr>
      <td style="padding:6px 12px;color:#d4d4d4;">🧪 Test Score</td>
      <td style="padding:6px 12px;color:#22c55e;text-align:right;font-weight:bold;">{tests}/80</td>
      <td style="padding:6px 12px;color:#888;">({passed}/{ttl} cases passed)</td>
    </tr>
    <tr>
      <td style="padding:6px 12px;color:#d4d4d4;">📐 Diff Score</td>
      <td style="padding:6px 12px;color:#60a5fa;text-align:right;font-weight:bold;">{diff}/20</td>
      <td style="padding:6px 12px;color:#888;">(similarity to original)</td>
    </tr>
    <tr>
      <td style="padding:6px 12px;color:#d4d4d4;">💡 Hint Penalty</td>
      <td style="padding:6px 12px;color:#f87171;text-align:right;font-weight:bold;">−{penalty}</td>
      <td style="padding:6px 12px;color:#888;"></td>
    </tr>
  </table>
</div>
"""


# ── Full two-page interface ───────────────────────────────────────────────────

def create_full_interface() -> gr.Blocks:
    """
    Returns a Gradio app with three logical pages:
      Page 1: Setup form (name / URL / level / bugs)
      Page 2: Challenge interface (shown after pipeline finishes)
      Page 3: Results (shown after Submit)
    """

    with gr.Blocks(title="Legacy Code Challenge") as demo:

        # Shared state
        workspace_state        = gr.State("")
        hints_used_state       = gr.State(0)
        submission_count_state = gr.State(0)
        timer_trigger          = gr.Number(value=0, visible=False)

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
                nesting_slider = gr.Slider(
                    minimum=1, maximum=6, value=3, step=1,
                    label="🔗 Nesting Level (call-chain depth)",
                    info="Higher = deeper call chains (1=simple, 6=very deep)",
                )
                bugs_slider = gr.Slider(
                    minimum=1, maximum=5, value=1, step=1,
                    label="🐛 Number of Bugs to Inject",
                )
            with gr.Row():
                refactoring_check = gr.Checkbox(
                    label="🔀 Enable Refactoring (obfuscation/spaghettification)",
                    value=False,
                )
                debug_check = gr.Checkbox(
                    label="🐞 Debug Mode (verbose output, show bug locations)",
                    value=False,
                )
            timer_slider = gr.Slider(
                minimum=0, maximum=120, value=0, step=5,
                label="⏱️ Challenge Timer (minutes — 0 = no timer)",
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

            with gr.Row(elem_classes=["header-row"]):
                header_md  = gr.Markdown("### 🐛 Legacy Code Challenge")
                timer_html = gr.HTML("", elem_id="challenge-timer")

            with gr.Row(elem_classes=["main-row"]):

                # ── Left column: tabs ─────────────────────────────────────
                with gr.Column(scale=3, elem_classes=["left-col"]):
                    with gr.Tabs(selected=0, elem_classes=["left-tabs"]) as left_tabs:

                        with gr.Tab("📋 Challenge", id=0):
                            challenge_readme = gr.Markdown("")

                        with gr.Tab("💻 Code Editor", id=1):
                            file_dropdown = gr.Dropdown(
                                choices=[],
                                label="File",
                                info="Select a file to view. Submit saves to the target file.",
                            )
                            with gr.Row():
                                btn_tests   = gr.Button("▶ Run Tests",        variant="secondary")
                                btn_changes = gr.Button("👁️ View My Changes", variant="secondary")
                            with gr.Row():
                                search_box  = gr.Textbox(
                                    placeholder="Search in file… (Enter = next match)", show_label=False, scale=5,
                                )
                                search_prev = gr.Button("◀", variant="secondary", scale=1, min_width=40)
                                search_btn  = gr.Button("🔍 Find", variant="secondary", scale=1)
                                search_next = gr.Button("▶", variant="secondary", scale=1, min_width=40)
                            code_editor = gr.Code(
                                value="",
                                language="python",
                                label="",
                                interactive=True,
                                lines=30,
                                elem_id="code-editor",
                            )
                            save_status = gr.Markdown("")
                            save_btn = gr.Button("💾 Save", variant="secondary")

                        with gr.Tab("🧪 Test Results", id=2):
                            test_output = gr.HTML(
                                "<p style='color:#888;font-family:monospace;'>Click '▶ Run Tests' to run…</p>"
                            )
                            run_tests_tab_btn = gr.Button("▶ Run Tests", variant="secondary")

                        with gr.Tab("👁️ My Changes", id=3):
                            gr.Markdown("_Diff: your saved file vs the original buggy code you received_")
                            changes_diff_html = gr.HTML("")
                            with gr.Row():
                                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                                revert_btn  = gr.Button("↩️ Revert to Original", variant="secondary")

                # ── Right column: AI assistant + submit ───────────────────
                with gr.Column(scale=2, elem_classes=["right-col"]):
                    hint_counter = gr.Markdown(_hint_md(0, 0))
                    chatbot = gr.Chatbot(
                        label="AI Assistant",
                        elem_id="ai-chatbot",
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask the AI assistant for a hint…",
                            show_label=False,
                            scale=5,
                        )
                        send_btn = gr.Button("Send", variant="secondary", scale=1)

                    gr.Markdown("---")
                    submit_btn = gr.Button("🚀 Submit Fix", variant="primary")

        # ════════════════════════════════════════════════════════════════════
        # PAGE 3 — Results (hidden until Submit clicked)
        # ════════════════════════════════════════════════════════════════════
        with gr.Column(visible=False) as results_page:
            gr.Markdown("## 📊 Submission Results")

            score_summary_html = gr.HTML("")

            with gr.Tabs():
                with gr.Tab("🧪 Test Results"):
                    results_test_html = gr.HTML("")

                with gr.Tab("✏️ Your Changes"):
                    gr.Markdown("_Diff: your submitted code vs the original buggy code you received_")
                    results_your_diff_html = gr.HTML("")

                with gr.Tab("🔍 Expected Fix"):
                    gr.Markdown("_Diff: what the original correct code looks like vs the buggy version_")
                    results_expected_diff_html = gr.HTML("")

                with gr.Tab("💡 Hints Used"):
                    results_hints_html = gr.HTML("")

        # ── Setup callback ─────────────────────────────────────────────────

        def on_start(name, url, nesting_lvl, num_bugs, refactoring, debug, timer_mins):
            nesting = int(nesting_lvl)

            yield (
                gr.update(visible=True,
                          value="⏳ Cloning repository and generating challenge…"
                                " this may take 1–2 minutes."),
                gr.update(interactive=False),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                "", 0,
            )

            try:
                workspace_path = _run_pipeline(url.strip(), nesting, int(num_bugs), refactoring, debug)
                cs       = ChallengeState(workspace_path)
                py_files = cs.list_py_files()
                default  = cs.target_file if cs.target_file in py_files else (py_files[0] if py_files else "")
                name_str = name.strip()
                suffix   = f" &nbsp;|&nbsp; {name_str}" if name_str else ""

                yield (
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(value=f"### 🐛 Legacy Code Challenge &nbsp;|&nbsp; Nesting Level {nesting}{suffix}"),
                    gr.update(value=cs.readme()),
                    gr.update(choices=py_files, value=default),
                    gr.update(value=cs.read_target(), label=cs.target_file, interactive=True),
                    workspace_path,
                    int(timer_mins),
                )

            except Exception as exc:
                yield (
                    gr.update(visible=True, value=f"❌ Error: {exc}"),
                    gr.update(interactive=True),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    "", 0,
                )

        start_btn.click(
            on_start,
            inputs=[name_box, url_box, nesting_slider, bugs_slider, refactoring_check, debug_check, timer_slider],
            outputs=[
                status_box, start_btn,
                setup_page, challenge_page,
                header_md, challenge_readme,
                file_dropdown, code_editor,
                workspace_state, timer_trigger,
            ],
        )

        timer_trigger.change(
            fn=None, inputs=[timer_trigger],
            js="(n) => { window.startTimer && window.startTimer(n); return n; }",
            outputs=[timer_trigger],
        )

        # ── Challenge callbacks ────────────────────────────────────────────

        def on_file_select(rel_path, workspace_path):
            if not workspace_path:
                return gr.update()
            cs = ChallengeState(workspace_path)
            content = cs.read_py_file(rel_path)
            return gr.update(value=content, interactive=True, label=rel_path)

        def on_save(code, workspace_path):
            if not workspace_path:
                return "⚠️ No challenge loaded."
            cs = ChallengeState(workspace_path)
            cs.write_target(code)
            return "✅ Saved"

        def on_run_tests(workspace_path):
            loading = '<p style="text-align:center;padding:20px;color:#888;">⏳ Running tests…</p>'
            yield (gr.update(selected=2), loading)
            if not workspace_path:
                yield (gr.update(selected=2), "<p style='color:#888'>No challenge loaded.</p>")
                return
            try:
                from orchestrator.scoring import run_tests
                r   = run_tests(workspace_path)
                raw = r["output"] if r["output"] else "No output."
                yield (gr.update(selected=2), _colorise_test_output(raw))
            except Exception as exc:
                yield (gr.update(selected=2), f"<p style='color:#ef4444;font-family:monospace;'>❌ Error: {exc}</p>")

        def on_show_changes(workspace_path):
            if not workspace_path:
                diff = "<p style='color:#888'>No challenge loaded.</p>"
            else:
                cs      = ChallengeState(workspace_path)
                current = cs.read_target()
                diff    = _diff_html(cs.sabotaged_code, current, "buggy (received)", "your saved code")
            return (gr.update(selected=3), diff)

        def on_revert(workspace_path):
            if not workspace_path:
                return gr.update(), gr.update(selected=1)
            cs = ChallengeState(workspace_path)
            cs.reset_target()
            return cs.sabotaged_code, gr.update(selected=1)

        def on_submit(code, hints_used, submit_count, workspace_path, history):
            _loading = '<p style="text-align:center;padding:40px;color:#888;font-size:1.2em;">⏳ Running tests…</p>'
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                _loading, "", "", "", "",
                submit_count,
            )
            if not workspace_path:
                yield (gr.update(), gr.update(), "No challenge loaded.", "", "", "", "", submit_count)
                return
            try:
                cs  = ChallengeState(workspace_path)
                log = SubmissionLog(cs.workspace)
                result = evaluate_submission(
                    workspace_path=workspace_path,
                    student_code=code,
                    original_code=cs.original_code,
                    bug_func_name=cs.bug_func_name,
                    hints_used=hints_used,
                )
                log.save(code, result, hints_used)
                new_count = submit_count + 1

                score_html    = _score_summary_html(result)
                your_diff     = _diff_html(cs.sabotaged_code, code, "buggy (received)", "your fix")
                expected_diff = _expected_fix_diff_html(cs)
                test_html     = _colorise_test_output(result["test_output"] or "No test output.")
                hints_html    = _hints_html(history or [])

                yield (
                    gr.update(), gr.update(),
                    score_html, your_diff, expected_diff, test_html, hints_html,
                    new_count,
                )
            except Exception as exc:
                err = f"<p style='color:#ef4444;padding:20px;font-family:monospace;'>❌ Error during evaluation:<br>{exc}</p>"
                yield (gr.update(), gr.update(), err, "", "", "", "", submit_count)

        def on_send(message, history, hints_used, submit_count, workspace_path):
            if not message.strip():
                penalty = PENALTY_TABLE[min(hints_used, len(PENALTY_TABLE) - 1)]
                return history, "", hints_used, _hint_md(hints_used, penalty)
            if not workspace_path:
                history = list(history or []) + [
                    {"role": "user",      "content": message},
                    {"role": "assistant", "content": "No challenge loaded yet."},
                ]
                return history, "", hints_used, _hint_md(hints_used, 0)
            cs = ChallengeState(workspace_path)
            result = get_hint(
                user_message=message,
                history=history,
                hints_used=hints_used,
                submission_attempts=submit_count,
                challenge_info=cs.challenge_info(),
            )
            history = list(history or []) + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": result["response"]},
            ]
            new_hints = hints_used + (1 if result["gave_hint"] else 0)
            penalty   = PENALTY_TABLE[min(new_hints, len(PENALTY_TABLE) - 1)]
            return history, "", new_hints, _hint_md(new_hints, penalty)

        # ── Wiring ────────────────────────────────────────────────────────

        search_btn.click(fn=None, inputs=[search_box], outputs=[search_box], js=_SEARCH_JS)
        search_box.submit(fn=None, inputs=[search_box], outputs=[search_box], js=_SEARCH_JS)
        search_next.click(fn=None, inputs=[search_box], outputs=[search_box], js=_SEARCH_NEXT_JS)
        search_prev.click(fn=None, inputs=[search_box], outputs=[search_box], js=_SEARCH_PREV_JS)

        file_dropdown.change(
            on_file_select,
            inputs=[file_dropdown, workspace_state],
            outputs=[code_editor],
        )
        save_btn.click(on_save, inputs=[code_editor, workspace_state], outputs=[save_status])

        btn_tests.click(on_run_tests, inputs=[workspace_state], outputs=[left_tabs, test_output])
        run_tests_tab_btn.click(on_run_tests, inputs=[workspace_state], outputs=[left_tabs, test_output])

        btn_changes.click(on_show_changes, inputs=[workspace_state], outputs=[left_tabs, changes_diff_html])
        refresh_btn.click(on_show_changes, inputs=[workspace_state], outputs=[left_tabs, changes_diff_html])

        revert_btn.click(on_revert, inputs=[workspace_state], outputs=[code_editor, left_tabs])

        submit_btn.click(
            on_submit,
            inputs=[code_editor, hints_used_state, submission_count_state,
                    workspace_state, chatbot],
            outputs=[
                challenge_page, results_page,
                score_summary_html,
                results_your_diff_html,
                results_expected_diff_html,
                results_test_html,
                results_hints_html,
                submission_count_state,
            ],
        )
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


# ── Legacy entry point (used by challenge.py CLI path) ───────────────────────

def create_interface(workspace_path: str, student_name: str = "", timer_minutes: int = 0) -> gr.Blocks:
    """Build the challenge-only interface for a pre-existing workspace."""

    cs  = ChallengeState(workspace_path)
    log = SubmissionLog(cs.workspace)

    py_files     = cs.list_py_files()
    default_file = cs.target_file if cs.target_file in py_files else (py_files[0] if py_files else None)

    name_suffix = f" &nbsp;|&nbsp; {student_name.strip()}" if student_name.strip() else ""

    with gr.Blocks(title="Legacy Code Challenge", js=_make_js(timer_minutes)) as demo:

        hints_used_state       = gr.State(0)
        submission_count_state = gr.State(0)

        with gr.Row(elem_classes=["header-row"]):
            gr.Markdown(
                f"### 🐛 Legacy Code Challenge &nbsp;|&nbsp; Nesting Level {cs.nesting_level}{name_suffix}",
                elem_classes=["app-header"],
            )
            gr.HTML("", elem_id="challenge-timer")

        # ── Challenge page ────────────────────────────────────────────────
        with gr.Column(visible=True) as challenge_col:
            with gr.Row(elem_classes=["main-row"]):
                with gr.Column(scale=3, elem_classes=["left-col"]):
                    with gr.Tabs(selected=0, elem_classes=["left-tabs"]) as left_tabs:

                        with gr.Tab("📋 Challenge", id=0):
                            gr.Markdown(cs.readme())

                        with gr.Tab("💻 Code Editor", id=1):
                            file_dropdown = gr.Dropdown(
                                choices=py_files, value=default_file, label="File",
                                info="Select a file to view. Submit saves to the target file.",
                            )
                            with gr.Row():
                                btn_tests   = gr.Button("▶ Run Tests",        variant="secondary")
                                btn_changes = gr.Button("👁️ View My Changes", variant="secondary")
                            with gr.Row():
                                search_box  = gr.Textbox(
                                    placeholder="Search in file… (Enter = next match)", show_label=False, scale=5,
                                )
                                search_prev = gr.Button("◀", variant="secondary", scale=1, min_width=40)
                                search_btn  = gr.Button("🔍 Find", variant="secondary", scale=1)
                                search_next = gr.Button("▶", variant="secondary", scale=1, min_width=40)
                            code_editor = gr.Code(
                                value=cs.read_target(), language="python", label=cs.target_file,
                                interactive=True, lines=30, elem_id="code-editor",
                            )
                            save_status = gr.Markdown("")
                            save_btn = gr.Button("💾 Save", variant="secondary")

                        with gr.Tab("🧪 Test Results", id=2):
                            test_output = gr.HTML(
                                "<p style='color:#888;font-family:monospace;'>Click '▶ Run Tests' to run…</p>"
                            )
                            run_tests_tab_btn = gr.Button("▶ Run Tests", variant="secondary")

                        with gr.Tab("👁️ My Changes", id=3):
                            gr.Markdown("_Diff: your saved file vs the original buggy code you received_")
                            changes_diff_html = gr.HTML("")
                            with gr.Row():
                                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                                revert_btn  = gr.Button("↩️ Revert to Original", variant="secondary")

                with gr.Column(scale=2, elem_classes=["right-col"]):
                    hint_counter = gr.Markdown(_hint_md(0, 0))
                    chatbot = gr.Chatbot(
                        label="AI Assistant",
                        elem_id="ai-chatbot",
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask the AI assistant for a hint…",
                            show_label=False, scale=5,
                        )
                        send_btn   = gr.Button("Send", variant="secondary", scale=1)
                    gr.Markdown("---")
                    submit_btn = gr.Button("🚀 Submit Fix", variant="primary")

        # ── Results page ──────────────────────────────────────────────────
        with gr.Column(visible=False) as results_col:
            gr.Markdown("## 📊 Submission Results")

            score_summary_html = gr.HTML("")

            with gr.Tabs():
                with gr.Tab("🧪 Test Results"):
                    results_test_html = gr.HTML("")
                with gr.Tab("✏️ Your Changes"):
                    gr.Markdown("_Diff: your submitted code vs the original buggy code_")
                    results_your_diff_html = gr.HTML("")
                with gr.Tab("🔍 Expected Fix"):
                    gr.Markdown("_Diff: what the correct original code looks like vs the buggy version_")
                    results_expected_diff_html = gr.HTML("")
                with gr.Tab("💡 Hints Used"):
                    results_hints_html = gr.HTML("")

        # ── Callbacks ─────────────────────────────────────────────────────

        def on_file_select(rel_path):
            content = cs.read_py_file(rel_path)
            return gr.update(value=content, interactive=True, label=rel_path)

        def on_save(code):
            cs.write_target(code)
            return "✅ Saved"

        def on_run_tests():
            loading = '<p style="text-align:center;padding:20px;color:#888;">⏳ Running tests…</p>'
            yield (gr.update(selected=2), loading)
            try:
                from orchestrator.scoring import run_tests
                r   = run_tests(workspace_path)
                raw = r["output"] if r["output"] else "No output."
                yield (gr.update(selected=2), _colorise_test_output(raw))
            except Exception as exc:
                yield (gr.update(selected=2), f"<p style='color:#ef4444;font-family:monospace;'>❌ Error: {exc}</p>")

        def on_show_changes():
            current = cs.read_target()
            diff    = _diff_html(cs.sabotaged_code, current, "buggy (received)", "your saved code")
            return (gr.update(selected=3), diff)

        def on_revert():
            cs.reset_target()
            return cs.sabotaged_code, gr.update(selected=1)

        def on_submit(code, hints_used, submit_count, history):
            _loading = '<p style="text-align:center;padding:40px;color:#888;font-size:1.2em;">⏳ Running tests…</p>'
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                _loading, "", "", "", "",
                submit_count,
            )
            try:
                result = evaluate_submission(
                    workspace_path=workspace_path,
                    student_code=code,
                    original_code=cs.original_code,
                    bug_func_name=cs.bug_func_name,
                    hints_used=hints_used,
                )
                log.save(code, result, hints_used)
                new_count = submit_count + 1

                score_html    = _score_summary_html(result)
                your_diff     = _diff_html(cs.sabotaged_code, code, "buggy (received)", "your fix")
                expected_diff = _expected_fix_diff_html(cs)
                test_html     = _colorise_test_output(result["test_output"] or "No test output.")
                hints_html    = _hints_html(history or [])

                yield (
                    gr.update(), gr.update(),
                    score_html, your_diff, expected_diff, test_html, hints_html,
                    new_count,
                )
            except Exception as exc:
                err = f"<p style='color:#ef4444;padding:20px;font-family:monospace;'>❌ Error during evaluation:<br>{exc}</p>"
                yield (gr.update(), gr.update(), err, "", "", "", "", submit_count)

        def on_send(message, history, hints_used, submit_count):
            if not message.strip():
                penalty = PENALTY_TABLE[min(hints_used, len(PENALTY_TABLE) - 1)]
                return history, "", hints_used, _hint_md(hints_used, penalty)
            result = get_hint(
                user_message=message, history=history,
                hints_used=hints_used, submission_attempts=submit_count,
                challenge_info=cs.challenge_info(),
            )
            history = list(history or []) + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": result["response"]},
            ]
            new_hints = hints_used + (1 if result["gave_hint"] else 0)
            penalty   = PENALTY_TABLE[min(new_hints, len(PENALTY_TABLE) - 1)]
            return history, "", new_hints, _hint_md(new_hints, penalty)

        # ── Wiring ────────────────────────────────────────────────────────

        search_btn.click(fn=None, inputs=[search_box], outputs=[search_box], js=_SEARCH_JS)
        search_box.submit(fn=None, inputs=[search_box], outputs=[search_box], js=_SEARCH_JS)
        search_next.click(fn=None, inputs=[search_box], outputs=[search_box], js=_SEARCH_NEXT_JS)
        search_prev.click(fn=None, inputs=[search_box], outputs=[search_box], js=_SEARCH_PREV_JS)

        file_dropdown.change(on_file_select, inputs=[file_dropdown], outputs=[code_editor])
        save_btn.click(on_save, inputs=[code_editor], outputs=[save_status])

        btn_tests.click(on_run_tests, outputs=[left_tabs, test_output])
        run_tests_tab_btn.click(on_run_tests, outputs=[left_tabs, test_output])

        btn_changes.click(on_show_changes, outputs=[left_tabs, changes_diff_html])
        refresh_btn.click(on_show_changes, outputs=[left_tabs, changes_diff_html])

        revert_btn.click(on_revert, outputs=[code_editor, left_tabs])

        submit_btn.click(
            on_submit,
            inputs=[code_editor, hints_used_state, submission_count_state, chatbot],
            outputs=[
                challenge_col, results_col,
                score_summary_html,
                results_your_diff_html,
                results_expected_diff_html,
                results_test_html,
                results_hints_html,
                submission_count_state,
            ],
        )
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
