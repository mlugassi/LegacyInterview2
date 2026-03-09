from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from architect.state import ArchitectState
from architect.nodes import (
    node_clone_repo,
    node_map_files,
    node_sabotage_init,
    node_inflate_hierarchy,
    node_obfuscation_level_1,
    node_obfuscation_level_2,
    node_verify_sabotage,
    node_overwrite_file,
    node_deploy,
    node_done,
)


def _route_after_inflation(state: ArchitectState) -> str:
    """If refactoring enabled, apply obfuscation; otherwise skip to verify."""
    if state["refactoring_enabled"]:
        return "obfuscation_level_1"
    return "verify_sabotage"


def build_graph() -> StateGraph:
    builder = StateGraph(ArchitectState)

    # Phase 1: Environment & Discovery
    builder.add_node("clone_repo",          RunnableLambda(node_clone_repo))
    builder.add_node("map_files",           RunnableLambda(node_map_files))

    # Phase 2: Sabotage
    builder.add_node("sabotage_init",       RunnableLambda(node_sabotage_init))
    builder.add_node("inflate_hierarchy",   RunnableLambda(node_inflate_hierarchy))

    # Phase 3: Optional Refactoring (if enabled)
    builder.add_node("obfuscation_level_1", RunnableLambda(node_obfuscation_level_1))
    builder.add_node("obfuscation_level_2", RunnableLambda(node_obfuscation_level_2))

    # Phase 4: Verification & Deployment
    builder.add_node("verify_sabotage",     RunnableLambda(node_verify_sabotage))
    builder.add_node("overwrite_file",      RunnableLambda(node_overwrite_file))
    builder.add_node("deploy",              RunnableLambda(node_deploy))
    builder.add_node("done",                RunnableLambda(node_done))

    # Edges
    builder.set_entry_point("clone_repo")
    builder.add_edge("clone_repo",    "map_files")
    builder.add_edge("map_files",     "sabotage_init")
    builder.add_edge("sabotage_init", "inflate_hierarchy")

    # After inflation: if refactoring → obfuscate, else → verify
    builder.add_conditional_edges(
        "inflate_hierarchy",
        _route_after_inflation,
        {
            "obfuscation_level_1": "obfuscation_level_1",
            "verify_sabotage":     "verify_sabotage",
        },
    )

    # Refactoring chain: obfuscate → spaghettify → verify
    builder.add_edge("obfuscation_level_1", "obfuscation_level_2")
    builder.add_edge("obfuscation_level_2", "verify_sabotage")

    builder.add_edge("verify_sabotage",     "overwrite_file")
    builder.add_edge("overwrite_file",      "deploy")
    builder.add_edge("deploy",              "done")
    builder.add_edge("done",                END)

    return builder.compile()
