from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from architect.state import ArchitectState
from architect.nodes import (
    node_clone_repo,
    node_map_files,
    node_sabotage_init,
    node_code_inflation,
    node_obfuscation_level_1,
    node_obfuscation_level_2,
    node_verify_sabotage,
    node_overwrite_file,
    node_deploy,
    node_done,
)


def _route_after_inflation(state: ArchitectState) -> str:
    """Level 1 -> obfuscation (cryptic names). Level 2/3 -> spaghettification (deep nesting)."""
    if state["difficulty_level"] == 1:
        return "obfuscation_level_1"
    return "obfuscation_level_2"


def _route_after_level_2(state: ArchitectState) -> str:
    """Level 3 chains into Level 1 obfuscation after spaghettification; Level 2 goes to verify."""
    if state["difficulty_level"] == 3:
        return "obfuscation_level_1"
    return "verify_sabotage"


def build_graph() -> StateGraph:
    builder = StateGraph(ArchitectState)

    # Phase 1: Environment & Discovery
    builder.add_node("clone_repo",          RunnableLambda(node_clone_repo))
    builder.add_node("map_files",           RunnableLambda(node_map_files))

    # Phase 2: Execution Nodes
    # Node 1 - Target Selection: pick the function and inject the bug
    builder.add_node("sabotage_init",       RunnableLambda(node_sabotage_init))
    # Node 2 - Anti-Analysis Bloating: inflate to 200+ lines with busy-work
    builder.add_node("code_inflation",      RunnableLambda(node_code_inflation))
    # Node 3 - Semantic Stripping (Level 1/3): rename vars to meaningless identifiers
    builder.add_node("obfuscation_level_1", RunnableLambda(node_obfuscation_level_1))
    # Node 4 - Deep Nesting (Level 2/3): implement bug at minimum 4-call depth
    builder.add_node("obfuscation_level_2", RunnableLambda(node_obfuscation_level_2))
    # Node 5 - Integrity Check: confirm bug still manifests, no crashes introduced
    builder.add_node("verify_sabotage",     RunnableLambda(node_verify_sabotage))

    # Phase 3: Deployment
    builder.add_node("overwrite_file",      RunnableLambda(node_overwrite_file))
    builder.add_node("deploy",              RunnableLambda(node_deploy))
    builder.add_node("done",                RunnableLambda(node_done))

    # Edges
    builder.set_entry_point("clone_repo")
    builder.add_edge("clone_repo",    "map_files")
    builder.add_edge("map_files",     "sabotage_init")
    builder.add_edge("sabotage_init", "code_inflation")

    # After code_inflation: Level 1 -> obfuscate names; Level 2/3 -> deep nesting first
    builder.add_conditional_edges(
        "code_inflation",
        _route_after_inflation,
        {
            "obfuscation_level_1": "obfuscation_level_1",
            "obfuscation_level_2": "obfuscation_level_2",
        },
    )

    # After deep nesting: Level 3 also renames everything; Level 2 goes straight to verify
    builder.add_conditional_edges(
        "obfuscation_level_2",
        _route_after_level_2,
        {
            "obfuscation_level_1": "obfuscation_level_1",
            "verify_sabotage":     "verify_sabotage",
        },
    )

    builder.add_edge("obfuscation_level_1", "verify_sabotage")
    builder.add_edge("verify_sabotage",     "overwrite_file")
    builder.add_edge("overwrite_file",      "deploy")
    builder.add_edge("deploy",              "done")
    builder.add_edge("done",                END)

    return builder.compile()
