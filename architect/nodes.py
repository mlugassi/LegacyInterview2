import os

from architect.state import ArchitectState
from architect.repo_cloner import clone_repo
from architect.file_mapper import map_files
from architect.saboteur import sabotage
from architect.challenge_deployer import deploy_challenge
from architect.readme_generator import create_readme


def node_clone_repo(state: ArchitectState) -> ArchitectState:
    return clone_repo(state)


def node_map_files(state: ArchitectState) -> ArchitectState:
    return map_files(state)


def node_sabotage(state: ArchitectState) -> ArchitectState:
    return sabotage(state)


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
    level_names = {1: "Level 1 — Messy Code", 2: "Level 2 — Spaghetti Logic", 3: "Level 3 — Sensitive Code"}
    target_rel = os.path.relpath(state["target_file"], state["clone_path"])
    summary = (
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║          LEGACY CHALLENGE ARCHITECT — REPORT             ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
        f"  Repository   : {state['github_url']}\n"
        f"  Cloned to    : {state['clone_path']}\n"
        f"  Target file  : {target_rel}\n"
        f"  Difficulty   : {level_names.get(state['difficulty_level'], str(state['difficulty_level']))}\n"
        f"  Function     : {state['function_name']}{state['test_args']}\n"
        f"  Expected     : {state['expected_output']}\n"
        f"  Actual (bug) : {state['actual_output']}\n"
        f"  Bug injected : {state['bug_description']}\n"
        "\n"
        "  Files created:\n"
        "    • challenge_run.py\n"
        "    • STUDENT_README.md\n"
        "\n"
        "  Environment is ready. Share the cloned folder with the student.\n"
    )
    print(summary)
    state["challenge_summary"] = summary
    return state
