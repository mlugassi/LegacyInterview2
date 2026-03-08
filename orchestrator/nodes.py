"""
Challenge Orchestrator nodes:
  1. node_run_bug_generator   — invoke the existing architect LangGraph pipeline
  2. node_save_challenge_state — persist metadata to challenge_state.json
  3. node_launch_student_gui  — start the Gradio student interface
"""

import json
from pathlib import Path

from orchestrator.state import OrchestratorState


# ── Node 1: call the Bug Generator Agent ─────────────────────────────────────

def node_run_bug_generator(state: OrchestratorState) -> OrchestratorState:
    """Invoke the existing architect pipeline and collect its output."""
    from architect.graph import build_graph as build_architect_graph

    print("\n[orchestrator] ── Step 1/3: Running Bug Generator Agent ──────────────")

    architect_graph = build_architect_graph()

    initial_arch_state = {
        "github_url":       state["github_url"],
        "difficulty_level": state["difficulty_level"],
        "num_bugs":         state["num_bugs"],
        # remaining fields initialised to empty (architect nodes will populate them)
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
    }

    result = architect_graph.invoke(initial_arch_state)

    print(f"[orchestrator] Bug Generator complete. Workspace: {result.get('clone_path', '?')}")

    return {
        **state,
        "workspace_path": result.get("clone_path", ""),
        "target_file":    result.get("target_file", ""),
        "original_code":  result.get("original_code", ""),
        "sabotaged_code": result.get("sabotaged_code", ""),
        "function_name":  result.get("function_name", ""),
        "bug_func_name":  result.get("bug_func_name", ""),
        "bug_func_source": result.get("bug_func_source", ""),
        "test_cases":     result.get("test_cases", []),
        "bug_description": result.get("bug_description", ""),
    }


# ── Node 2: save challenge_state.json ────────────────────────────────────────

def node_save_challenge_state(state: OrchestratorState) -> OrchestratorState:
    """Write all architect metadata to <workspace>/challenge_state.json."""
    print("\n[orchestrator] ── Step 2/3: Saving challenge state ──────────────────")

    workspace = Path(state["workspace_path"])
    if not workspace.exists():
        print(f"[orchestrator] WARNING: workspace path does not exist: {workspace}")

    challenge_state = {
        "github_url":       state["github_url"],
        "workspace_path":   str(workspace),
        "target_file":      state["target_file"],
        "original_code":    state["original_code"],
        "sabotaged_code":   state["sabotaged_code"],
        "function_name":    state["function_name"],
        "bug_func_name":    state["bug_func_name"],
        "bug_func_source":  state["bug_func_source"],
        "test_cases":       state["test_cases"],
        "difficulty_level": state["difficulty_level"],
        "bug_description":  state["bug_description"],
    }

    state_path = workspace / "challenge_state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(challenge_state, f, indent=2, ensure_ascii=False)

    print(f"[orchestrator] Challenge state saved → {state_path}")
    return {**state, "challenge_state_path": str(state_path)}


# ── Node 3: launch the student Gradio GUI ────────────────────────────────────

def node_launch_student_gui(state: OrchestratorState) -> OrchestratorState:
    """Import student_interface and launch Gradio."""
    import student_interface

    port  = state.get("port", 7860)
    share = state.get("share", False)

    print(f"\n[orchestrator] ── Step 3/3: Launching student GUI ─────────────────")
    print(f"[orchestrator] Opening http://localhost:{port} ...")

    import gradio as gr
    interface = student_interface.create_interface(state["workspace_path"])
    interface.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=student_interface._CSS,
    )

    return {**state, "launch_status": f"Launched on port {port}"}
