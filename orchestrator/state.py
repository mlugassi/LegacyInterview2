"""OrchestratorState — shared state for the Challenge Orchestrator LangGraph."""

from typing import TypedDict


class OrchestratorState(TypedDict):
    # ── Input (set by challenge.py CLI) ───────────────────────────────────────
    github_url: str
    difficulty_level: int
    num_bugs: int
    port: int
    share: bool

    # ── Filled by node_run_bug_generator ──────────────────────────────────────
    workspace_path: str       # absolute path to the cloned workspace directory
    target_file: str          # relative path of the sabotaged file (e.g. boltons\formatutils.py)
    original_code: str        # file content BEFORE sabotage (used for diff scoring)
    sabotaged_code: str       # file content AFTER sabotage (what the student receives)
    function_name: str        # public surface function reported as broken
    bug_func_name: str        # internal helper where the bug was actually injected
    bug_func_source: str      # exact source of the buggy helper function
    test_cases: list          # [{"args": str, "expected": str}, ...] verified cases
    bug_description: str      # architect-only description (never shown to student)

    # ── Filled by node_save_challenge_state ───────────────────────────────────
    challenge_state_path: str  # absolute path to the saved challenge_state.json

    # ── Filled by node_launch_student_gui ────────────────────────────────────
    launch_status: str         # "Launched on port N" or error message
