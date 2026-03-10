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
        self.bug_func_name:           str  = data.get("bug_func_name", "")
        self.original_bug_func_source: str  = data.get("original_bug_func_source", "")
        self.nesting_level:    int  = data.get("nesting_level", 3)
        self.refactoring_enabled: bool = data.get("refactoring_enabled", False)
        self.debug_mode:       bool = data.get("debug_mode", False)

        # sabotaged_files: {rel_posix_path: content} for every file the
        # saboteur touched.  Primary source: .challenge_snapshot/ directory
        # written by the deployer immediately after sabotage (always up-to-date).
        # Fallback: sabotaged_files dict in JSON, then single sabotaged_code field.
        snapshot_dir = self.workspace / ".challenge_snapshot"
        snap_files: dict[str, str] = {}
        if snapshot_dir.exists():
            for snap_path in snapshot_dir.iterdir():
                if snap_path.is_file():
                    # filename encodes the relative path: separators replaced by __
                    rel_posix = snap_path.name.replace("__", "/")
                    try:
                        snap_files[rel_posix] = snap_path.read_text(encoding="utf-8")
                    except Exception:
                        pass

        if snap_files:
            self.sabotaged_files: dict[str, str] = snap_files
        else:
            # Fall back to JSON fields
            stored = data.get("sabotaged_files", {})
            if not stored and self.sabotaged_code:
                try:
                    rel = Path(self.target_file).resolve().relative_to(
                        self.workspace.resolve()
                    ).as_posix()
                except ValueError:
                    rel = Path(self.target_file).name
                stored = {rel: self.sabotaged_code}
            self.sabotaged_files = stored

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
        return [
            f.relative_to(self.workspace).as_posix() for f in files
            if ".challenge_snapshot" not in f.parts
        ]

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
            "timestamp":       ts,
            "hints_used":      hints_used,
            "llm_score":       result.get("llm_score", result.get("total_score", 0)),
            "llm_explanation": result.get("llm_explanation", ""),
            "hint_penalty":    result["hint_penalty"],
            "total_score":     result["total_score"],
            "passed":          result["passed"],
            "total_tests":     result["total_tests"],
            "all_passed":      result["all_passed"],
            "student_code":    student_code,
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

    # Read the actual file on disk (may differ from state if extra transforms ran)
    target_file_rel = result.get("target_file", "")
    actual_sabotaged_code = result.get("sabotaged_code", "")
    workspace_path_obj = Path(workspace_path).resolve()
    if target_file_rel:
        target_path = Path(workspace_path) / target_file_rel
        if target_path.exists():
            actual_sabotaged_code = target_path.read_text(encoding="utf-8")

    # Build sabotaged_files: {rel_posix_path: content} for every file the
    # saboteur modified.  Currently always one file (target_file).
    sabotaged_files: dict[str, str] = {}
    if target_file_rel and actual_sabotaged_code:
        try:
            rel_posix = Path(target_file_rel).resolve().relative_to(workspace_path_obj).as_posix()
        except (ValueError, OSError):
            rel_posix = Path(target_file_rel).as_posix()
        sabotaged_files[rel_posix] = actual_sabotaged_code

    # Persist challenge_state.json
    challenge_state = {
        "github_url":          github_url,
        "workspace_path":      workspace_path,
        "target_file":         target_file_rel,
        "original_code":       result.get("original_code",    ""),
        "sabotaged_code":      actual_sabotaged_code,
        "sabotaged_files":     sabotaged_files,
        "function_name":       result.get("function_name",    ""),
        "bug_func_name":       result.get("bug_func_name",    ""),
        "bug_func_source":          result.get("bug_func_source",           ""),
        "original_bug_func_source": result.get("original_bug_func_source",  ""),
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
        f"""
    (function tryStartTimer() {{
        if (document.getElementById('timer-inner')) {{
            window.startTimer({timer_minutes});
        }} else {{
            setTimeout(tryStartTimer, 150);
        }}
    }})();"""
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
            const el = document.getElementById('timer-inner');
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
    };
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            const btn = document.getElementById('save-btn');
            if (btn) btn.click();
        }
    }, true);""" + auto_start + "\n}"


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

/* Challenge timer — pushed to the right corner of the header */
.header-row { display: flex !important; align-items: center !important;
              width: 100% !important; gap: 8px; }
.header-row > * { flex-shrink: 0 !important; }
.header-row > *:first-child { flex: 1 1 auto !important; min-width: 0; }
#challenge-timer { flex: 0 0 auto !important; font-family: monospace;
                   font-size: 1em; text-align: right; padding: 4px 8px;
                   min-width: 90px; white-space: nowrap; }
"""


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


def _workspace_diff_html(cs: "ChallengeState") -> str:
    """
    Show a unified diff for every file the student changed, compared to the
    state they received.

    For files the saboteur touched (cs.sabotaged_files): compare the locally
    stored received-state snapshot vs the current disk content.
    For every other modified file: compare git HEAD vs disk.

    This ensures only the student's changes are shown — not the sabotage noise.
    """
    import subprocess

    workspace        = cs.workspace.resolve()
    sabotaged_files  = cs.sabotaged_files   # {rel_posix: received_content}
    sections: list[str] = []

    # ── 1. Files modified by the saboteur ────────────────────────────────────
    for rel_posix, received_content in sabotaged_files.items():
        full_path = workspace / rel_posix
        if not full_path.exists():
            continue
        current = full_path.read_text(encoding="utf-8")
        if _normalize(current) == _normalize(received_content):
            continue
        diff = _diff_html(received_content, current, rel_posix, rel_posix)
        sections.append(
            f"<h4 style='color:#94a3b8;margin:10px 0 4px;font-family:monospace;'>"
            f"📄 {rel_posix}</h4>" + diff
        )

    # ── 2. Other modified files via git diff ──────────────────────────────────
    try:
        proc = subprocess.run(
            ["git", "diff", "HEAD", "--name-only"],
            cwd=str(workspace), capture_output=True, text=True,
            encoding="utf-8", timeout=10,
        )
        for changed_rel in proc.stdout.splitlines():
            changed_posix = changed_rel.replace("\\", "/")
            if changed_posix in sabotaged_files:
                continue          # already handled above
            if not changed_rel.endswith(".py"):
                continue
            orig_proc = subprocess.run(
                ["git", "show", f"HEAD:{changed_rel}"],
                cwd=str(workspace), capture_output=True, text=True,
                encoding="utf-8", timeout=10,
            )
            if orig_proc.returncode != 0:
                continue
            full_path = workspace / changed_rel
            if not full_path.exists():
                continue
            current = full_path.read_text(encoding="utf-8")
            if _normalize(current) == _normalize(orig_proc.stdout):
                continue
            diff = _diff_html(orig_proc.stdout, current, changed_posix, changed_posix)
            sections.append(
                f"<h4 style='color:#94a3b8;margin:10px 0 4px;font-family:monospace;'>"
                f"📄 {changed_posix}</h4>" + diff
            )
    except Exception:
        pass

    if not sections:
        return "<div style='color:#22c55e;padding:12px;'>No changes detected in workspace.</div>"

    return "".join(sections)


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


def _compute_expected_fixed_code(cs: "ChallengeState") -> str | None:
    """
    Compute the expected correct version of the sabotaged file by surgically
    replacing only the injected bug tokens (using the pre-transform pair).

    Strategy: character-level diff between orig_pre and sabot_pre surfaces the
    exact changed substrings (e.g. 'plural' vs 'plural[:-1]').  Those same
    substrings survive obfuscation, so we can find-and-replace them directly
    in the fully-obfuscated sabotaged_code.
    Returns None if the replacement cannot be determined.
    """
    import difflib
    import json as _json

    # orig_pre: correct pre-transform function.
    # Use stored value if available, otherwise extract live from original_code.
    orig_pre = (
        cs.original_bug_func_source
        or _extract_function_source(cs.original_code, cs.bug_func_name)
    )

    # sabot_pre: sabotaged (buggy) pre-transform function — always from JSON.
    sabot_pre = ""
    try:
        _data = _json.loads((cs.workspace / "challenge_state.json").read_text(encoding="utf-8"))
        sabot_pre = _data.get("bug_func_source", "")
    except Exception:
        pass

    if not (orig_pre and sabot_pre):
        return None

    # Use snapshot as the base (always matches what the student received on disk).
    # Fall back to cs.sabotaged_code only if no snapshot entry exists for the target.
    try:
        target_rel = cs.target_path.resolve().relative_to(cs.workspace.resolve()).as_posix()
    except ValueError:
        target_rel = ""
    received_code = cs.sabotaged_files.get(target_rel) or cs.sabotaged_code

    # Extract the sabotaged function's exact text from received_code.
    # All replacements are scoped to this substring so we never accidentally
    # patch code outside the target function.
    func_in_received = _extract_function_source(received_code, cs.bug_func_name)
    if not func_in_received:
        return None  # can't locate function in the file — give up

    # Character-level diff: find exactly what changed between correct and buggy.
    # We use context-aware replacement: for each changed fragment in sabot_pre,
    # extend left until we hit a non-identifier char so we get a unique search term
    # (e.g. 'plural[:-1]' instead of just '[:-1]').
    matcher = difflib.SequenceMatcher(None, orig_pre, sabot_pre, autojunk=False)
    opcodes = matcher.get_opcodes()

    # Work on just the function text, not the whole file.
    fixed_func = func_in_received
    changed = False

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue

        orig_frag  = orig_pre[i1:i2]   # correct fragment
        sabot_frag = sabot_pre[j1:j2]  # buggy fragment

        # Skip pure-whitespace/newline differences (formatting noise, not bugs).
        if not orig_frag.strip() and not sabot_frag.strip():
            continue

        if tag in ("replace", "insert"):
            # Build a context-extended search/replace pair so we don't
            # accidentally match a shorter fragment elsewhere in the function.
            # Walk left in both strings to the same word boundary.
            if not sabot_frag.strip():
                continue

            ctx_start_orig  = i1
            ctx_start_sabot = j1
            while ctx_start_orig > 0 and ctx_start_sabot > 0:
                co = orig_pre[ctx_start_orig - 1]
                cs_ = sabot_pre[ctx_start_sabot - 1]
                if co != cs_ or not (co.isalnum() or co == "_"):
                    break
                ctx_start_orig  -= 1
                ctx_start_sabot -= 1

            search_str  = sabot_pre[ctx_start_sabot:j2]   # buggy (with context)
            replace_str = orig_pre[ctx_start_orig:i2]      # correct (with context)

            if search_str and search_str in fixed_func:
                fixed_func = fixed_func.replace(search_str, replace_str, 1)
                changed = True
            elif sabot_frag and sabot_frag in fixed_func:
                fixed_func = fixed_func.replace(sabot_frag, orig_frag, 1)
                changed = True

        elif tag == "delete":
            # Something exists in orig but was removed in sabot (e.g. +1).
            # Find the surrounding context in sabot_pre and reinsert the deleted text.
            if not orig_frag.strip():
                continue

            # Take left context from sabot_pre and clip right context at first newline
            # (comments/indentation differ between pre-transform and obfuscated code).
            ctx_left  = sabot_pre[max(0, j1 - 30):j1]
            ctx_right_raw = sabot_pre[j2:j2 + 20]
            nl = ctx_right_raw.find("\n")
            ctx_right = ctx_right_raw[:nl] if nl != -1 else ctx_right_raw

            # Trim left context leftward until we find a match in fixed_func
            for trim in range(len(ctx_left)):
                search_str  = ctx_left[trim:] + ctx_right
                replace_str = ctx_left[trim:] + orig_frag + ctx_right
                if search_str and search_str in fixed_func:
                    fixed_func = fixed_func.replace(search_str, replace_str, 1)
                    changed = True
                    break

    if not changed or fixed_func == func_in_received:
        return None

    # Splice the patched function back into the full file.
    fixed = received_code.replace(func_in_received, fixed_func, 1)
    if fixed != received_code:
        return fixed

    return None


def _expected_fix_diff_html(cs: "ChallengeState", submitted_code: str = "") -> str:
    """
    Show two sections:
    1. What the expected fix looks like (buggy → expected).
    2. How the student's submission compares to the expected fix
       (expected → submitted): empty if perfect, otherwise shows extra/wrong changes.
    """
    func = cs.bug_func_name

    # Resolve the received (snapshot) content for the target file
    try:
        target_rel = cs.target_path.resolve().relative_to(cs.workspace.resolve()).as_posix()
    except ValueError:
        target_rel = ""
    received_code = cs.sabotaged_files.get(target_rel) or cs.sabotaged_code

    expected_fixed = _compute_expected_fixed_code(cs)

    # ── Fallback: function-level or full-file diff ────────────────────────────
    if expected_fixed is None:
        if func:
            sabot_func = _extract_function_source(received_code, func)
            orig_func  = _extract_function_source(cs.original_code, func)
            if sabot_func and orig_func:
                expected_fixed_approx = received_code.replace(sabot_func, orig_func, 1)
                if expected_fixed_approx != received_code:
                    expected_fixed = expected_fixed_approx
        if expected_fixed is None:
            return _diff_html(received_code, cs.original_code, target_rel, target_rel)

    # ── Section 1: expected fix ───────────────────────────────────────────────
    section1 = (
        "<h4 style='color:#94a3b8;margin:8px 0 4px;'>🎯 Expected fix</h4>"
        + _diff_html(received_code, expected_fixed, target_rel, target_rel)
    )

    if not submitted_code:
        return section1

    # ── Section 2: workspace-wide comparison ─────────────────────────────────
    # "Perfect fix" = target file matches expected AND no other files were changed.
    def _ast_equivalent(a: str, b: str) -> bool:
        try:
            import ast as _ast
            return _ast.dump(_ast.parse(a)) == _ast.dump(_ast.parse(b))
        except SyntaxError:
            return _normalize(a) == _normalize(b)

    target_file_ok = _ast_equivalent(expected_fixed, submitted_code)

    # Check whether the student touched any files outside the sabotaged set.
    # Known-modified files = snapshot keys + the challenge target file itself.
    known_changed = set(cs.sabotaged_files.keys()) | ({target_rel} if target_rel else set())
    import subprocess as _sp
    try:
        proc = _sp.run(
            ["git", "diff", "HEAD", "--name-only"],
            cwd=str(cs.workspace.resolve()), capture_output=True,
            text=True, encoding="utf-8", timeout=10,
        )
        extra_changed = [
            f.replace("\\", "/") for f in proc.stdout.splitlines()
            if f.replace("\\", "/") not in known_changed
        ]
    except Exception:
        extra_changed = []

    is_perfect = target_file_ok and not extra_changed

    if is_perfect:
        verdict = (
            "<div style='margin-top:12px;padding:10px 14px;background:#052e16;"
            "border:1px solid #22c55e;border-radius:6px;color:#22c55e;font-weight:600;'>"
            "✅ Perfect fix — you changed exactly the right lines and nothing more."
            "</div>"
        )
    else:
        # Show all changes across the whole workspace so the student can see
        # both incorrect target-file changes and changes to other files.
        workspace_diff = _workspace_diff_html(cs)
        verdict = (
            "<h4 style='color:#94a3b8;margin:16px 0 4px;'>⚠️ Your changes vs expected fix</h4>"
            "<p style='color:#f59e0b;font-size:0.85em;margin:0 0 6px;'>"
            "Your changes differ from the minimal expected fix. "
            "Red lines were expected but missing or changed; "
            "green lines are extra changes.</p>"
            + workspace_diff
        )

    return section1 + verdict


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

        # Gradio multimodal messages can have content as a list of parts
        if isinstance(content, list):
            content = " ".join(str(p) for p in content if p)
        content = str(content) if content is not None else ""

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
    total       = result["total_score"]
    llm_score   = result.get("llm_score", total)
    explanation = result.get("llm_explanation", "")
    penalty     = result["hint_penalty"]
    passed      = result["passed"]
    ttl         = result["total_tests"]
    all_ok      = result["all_passed"]

    color = "#22c55e" if all_ok else ("#f59e0b" if total >= 50 else "#ef4444")
    badge = "🎉 ALL TESTS PASS!" if all_ok else ("⚠️ Partial" if total > 0 else "❌ Failed")

    explanation_html = ""
    if explanation:
        import html as _html
        escaped = _html.escape(explanation)
        explanation_html = (
            f'<div style="margin-top:14px;padding:10px 14px;background:#0f172a;'
            f'border-left:3px solid #60a5fa;border-radius:4px;color:#cbd5e1;'
            f'font-size:0.9em;line-height:1.5;">'
            f'<strong style="color:#60a5fa;">🤖 AI Evaluation:</strong><br>{escaped}</div>'
        )

    return f"""
<div style="padding:20px;border-radius:12px;background:#1e1e1e;border:2px solid {color};">
  <div style="font-size:2.5em;font-weight:bold;color:{color};text-align:center;">{total}/100</div>
  <div style="text-align:center;color:{color};font-size:1.1em;margin-bottom:16px;">{badge}</div>
  <table style="width:100%;border-collapse:collapse;font-family:monospace;">
    <tr>
      <td style="padding:6px 12px;color:#d4d4d4;">🤖 AI Score</td>
      <td style="padding:6px 12px;color:#60a5fa;text-align:right;font-weight:bold;">{llm_score}/100</td>
      <td style="padding:6px 12px;color:#888;">({passed}/{ttl} tests passed)</td>
    </tr>
    <tr>
      <td style="padding:6px 12px;color:#d4d4d4;">💡 Hint Penalty</td>
      <td style="padding:6px 12px;color:#f87171;text-align:right;font-weight:bold;">−{penalty}</td>
      <td style="padding:6px 12px;color:#888;"></td>
    </tr>
  </table>
  {explanation_html}
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
                timer_html = gr.HTML('<span id="timer-inner" style="font-family:monospace"></span>', elem_id="challenge-timer")

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
                            code_editor = gr.Code(
                                value="",
                                language="python",
                                label="",
                                interactive=True,
                                lines=30,
                                elem_id="code-editor",
                            )
                            save_status = gr.Markdown("")
                            save_btn = gr.Button("💾 Save", variant="secondary", elem_id="save-btn")

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
                cs   = ChallengeState(workspace_path)
                diff = _workspace_diff_html(cs)
            return (gr.update(selected=3), diff)

        def on_revert(workspace_path):
            if not workspace_path:
                return gr.update(), gr.update(selected=1)
            cs = ChallengeState(workspace_path)
            cs.reset_target()
            return cs.sabotaged_code, gr.update(selected=1)

        def on_submit(hints_used, submit_count, workspace_path, history):
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
                # Always read from disk — catches changes made in local IDE (e.g. VS Code)
                submitted_code = cs.read_target()
                result = evaluate_submission(
                    workspace_path=workspace_path,
                    student_code=submitted_code,
                    original_code=cs.original_code,
                    bug_func_name=cs.bug_func_name,
                    hints_used=hints_used,
                    sabotaged_code=cs.sabotaged_code,
                    target_file=str(cs.target_path),
                )
                log.save(submitted_code, result, hints_used)
                new_count = submit_count + 1

                score_html    = _score_summary_html(result)
                your_diff     = _workspace_diff_html(cs)
                expected_diff = _expected_fix_diff_html(cs, submitted_code)
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
            inputs=[hints_used_state, submission_count_state,
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
            gr.HTML('<span id="timer-inner"></span>', elem_id="challenge-timer")

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
                            code_editor = gr.Code(
                                value=cs.read_target(), language="python", label=cs.target_file,
                                interactive=True, lines=30, elem_id="code-editor",
                            )
                            save_status = gr.Markdown("")
                            save_btn = gr.Button("💾 Save", variant="secondary", elem_id="save-btn")

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
            return (gr.update(selected=3), _workspace_diff_html(cs))

        def on_revert():
            cs.reset_target()
            return cs.sabotaged_code, gr.update(selected=1)

        def on_submit(hints_used, submit_count, history):
            _loading = '<p style="text-align:center;padding:40px;color:#888;font-size:1.2em;">⏳ Running tests…</p>'
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                _loading, "", "", "", "",
                submit_count,
            )
            try:
                # Always read from disk — catches changes made in local IDE (e.g. VS Code)
                submitted_code = cs.read_target()
                result = evaluate_submission(
                    workspace_path=workspace_path,
                    student_code=submitted_code,
                    original_code=cs.original_code,
                    bug_func_name=cs.bug_func_name,
                    hints_used=hints_used,
                    sabotaged_code=cs.sabotaged_code,
                    target_file=str(cs.target_path),
                )
                log.save(submitted_code, result, hints_used)
                new_count = submit_count + 1

                score_html    = _score_summary_html(result)
                your_diff     = _workspace_diff_html(cs)
                expected_diff = _expected_fix_diff_html(cs, submitted_code)
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


        file_dropdown.change(on_file_select, inputs=[file_dropdown], outputs=[code_editor])
        save_btn.click(on_save, inputs=[code_editor], outputs=[save_status])

        btn_tests.click(on_run_tests, outputs=[left_tabs, test_output])
        run_tests_tab_btn.click(on_run_tests, outputs=[left_tabs, test_output])

        btn_changes.click(on_show_changes, outputs=[left_tabs, changes_diff_html])
        refresh_btn.click(on_show_changes, outputs=[left_tabs, changes_diff_html])

        revert_btn.click(on_revert, outputs=[code_editor, left_tabs])

        submit_btn.click(
            on_submit,
            inputs=[hints_used_state, submission_count_state, chatbot],
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
