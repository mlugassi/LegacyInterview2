from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from architect.state import ArchitectState
from architect.nodes import (
    node_clone_repo,
    node_map_files,
    node_sabotage,
    node_overwrite_file,
    node_deploy,
    node_done,
)


def build_graph() -> StateGraph:
    builder = StateGraph(ArchitectState)

    builder.add_node("clone_repo",     RunnableLambda(node_clone_repo))
    builder.add_node("map_files",      RunnableLambda(node_map_files))
    builder.add_node("sabotage",       RunnableLambda(node_sabotage))
    builder.add_node("overwrite_file", RunnableLambda(node_overwrite_file))
    builder.add_node("deploy",         RunnableLambda(node_deploy))
    builder.add_node("done",           RunnableLambda(node_done))

    builder.set_entry_point("clone_repo")
    builder.add_edge("clone_repo",     "map_files")
    builder.add_edge("map_files",      "sabotage")
    builder.add_edge("sabotage",       "overwrite_file")
    builder.add_edge("overwrite_file", "deploy")
    builder.add_edge("deploy",         "done")
    builder.add_edge("done",           END)

    return builder.compile()
