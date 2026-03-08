"""
Challenge Orchestrator — LangGraph pipeline.

Flow:
  START → node_run_bug_generator → node_save_challenge_state → node_launch_student_gui → END
"""

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from orchestrator.state import OrchestratorState
from orchestrator.nodes import (
    node_run_bug_generator,
    node_save_challenge_state,
    node_launch_student_gui,
)


def build_orchestrator() -> StateGraph:
    builder = StateGraph(OrchestratorState)

    builder.add_node("run_bug_generator",    RunnableLambda(node_run_bug_generator))
    builder.add_node("save_challenge_state", RunnableLambda(node_save_challenge_state))
    builder.add_node("launch_student_gui",   RunnableLambda(node_launch_student_gui))

    builder.set_entry_point("run_bug_generator")
    builder.add_edge("run_bug_generator",    "save_challenge_state")
    builder.add_edge("save_challenge_state", "launch_student_gui")
    builder.add_edge("launch_student_gui",   END)

    return builder.compile()
