import os

from architect.state import ArchitectState
from architect.repo_cloner import clone_repo
from architect.file_mapper import map_files
from architect.saboteur import (
    sabotage_init,
    inflate_hierarchy,
    apply_obfuscation_level_1,
    apply_obfuscation_level_2,
    verify_sabotage,
    add_misleading_comments,
)
from architect.challenge_deployer import deploy_challenge
from architect.readme_generator import create_readme


def node_clone_repo(state: ArchitectState) -> ArchitectState:
    return clone_repo(state)


def node_map_files(state: ArchitectState) -> ArchitectState:
    return map_files(state)


def node_sabotage_init(state: ArchitectState) -> ArchitectState:
    """Target Selection: pick the function and inject the AI-resistant bug."""
    return sabotage_init(state)


def node_inflate_hierarchy(state: ArchitectState) -> ArchitectState:
    """Hierarchy Inflation: inflate all functions in the call chain and add dummy code."""
    return inflate_hierarchy(state)


def node_obfuscation_level_1(state: ArchitectState) -> ArchitectState:
    """Semantic Stripping: rename vars to meaningless identifiers."""
    return apply_obfuscation_level_1(state)


def node_obfuscation_level_2(state: ArchitectState) -> ArchitectState:
    """Deep Nesting: spaghettification with complex control flow."""
    return apply_obfuscation_level_2(state)


def node_verify_sabotage(state: ArchitectState) -> ArchitectState:
    """Integrity Check: confirm bug still manifests, no crashes introduced."""
    return verify_sabotage(state)


def node_add_misleading_comments(state: ArchitectState) -> ArchitectState:
    """Add false bug hints in random locations to confuse AI and students."""
    return add_misleading_comments(state)


def node_overwrite_file(state: ArchitectState) -> ArchitectState:
    with open(state["target_file"], "w", encoding="utf-8") as f:
        f.write(state["sabotaged_code"])
    print(f"[overwrite] Sabotaged code written to: {state['target_file']}")
    return state


def node_deploy(state: ArchitectState) -> ArchitectState:
    state = deploy_challenge(state)
    state = create_readme(state)
    return state


def node_done(state: ArchitectState) -> ArchitectState:
    target_rel = os.path.relpath(state["target_file"], state["clone_path"])
    refactoring_status = "Enabled (obfuscation + spaghettification)" if state["refactoring_enabled"] else "Disabled (bugs + inflation only)"
    
    summary = (
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║          LEGACY CHALLENGE ARCHITECT — REPORT             ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
        f"  Repository    : {state['github_url']}\n"
        f"  Cloned to     : {state['clone_path']}\n"
        f"  Target file   : {target_rel}\n"
        f"  Nesting level : {state['nesting_level']} (call-chain depth)\n"
        f"  Refactoring   : {refactoring_status}\n"
        f"  Bugs injected : {state['num_bugs']}\n"
        f"  Function      : {state['function_name']}{state['test_args']}\n"
        f"  Expected      : {state['expected_output']}\n"
        f"  Actual (bug)  : {state['actual_output']}\n"
        f"  Description   : {state['bug_description']}\n"
        "\n"
        "  Files created:\n"
        "    • challenge_run.py (5 public tests)\n"
        "    • challenge_run_secret.py (5 secret tests)\n"
        "    • STUDENT_README.md\n"
        "    • detailed_explanation.txt (for instructor)\n"
        "\n"
        "  Environment is ready. Share the cloned folder with the student.\n"
    )
    print(summary)
    state["challenge_summary"] = summary
    return state
