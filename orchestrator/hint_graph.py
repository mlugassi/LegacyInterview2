"""
Hint Sub-Graph — LangGraph agent for progressive AI hints.

Allow-level policy (based on hints_used):
  0 hints  → level 0 : no location information, pure methodology guidance
  1–2 hints → level 1 : may mention the bug is not in the surface function
  3–4 hints → level 2 : may describe the general bug type (e.g. "off-by-one")
  5+ hints  → level 3 : may name the specific helper function that has the bug

The graph:
  START → node_check_hint_policy → node_generate_hint → END
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# ── State ─────────────────────────────────────────────────────────────────────

class HintState(TypedDict):
    messages: list           # [{"role": "user"|"assistant", "content": str}, ...]
    hints_used: int          # total AI hints consumed so far this session
    submission_attempts: int # number of times student has submitted
    challenge_info: dict     # keys: function_name, bug_func_name, target_file, difficulty_level
    allow_level: int         # 0–3, set by node_check_hint_policy
    response: str            # filled by node_generate_hint


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_check_hint_policy(state: HintState) -> HintState:
    """Compute the allow_level based on how many hints have already been given."""
    h = state["hints_used"]
    if h == 0:
        level = 0
    elif h <= 2:
        level = 1
    elif h <= 4:
        level = 2
    else:
        level = 3
    return {**state, "allow_level": level}


def node_generate_hint(state: HintState) -> HintState:
    """Call GPT-4o with a system prompt tuned to the current allow_level."""
    allow_level = state["allow_level"]
    info = state["challenge_info"]
    func_name = info.get("function_name", "the function")
    bug_func = info.get("bug_func_name", "")
    target_file = info.get("target_file", "the target file")

    # Build the hint-policy section of the system prompt
    if allow_level == 0:
        policy = (
            "HINT POLICY — LEVEL 0 (no location clues):\n"
            "  ❌ DO NOT mention where the bug is, which function is broken, or what type of bug it is.\n"
            "  ✅ You CAN explain general debugging strategies, Python language features, "
            "and how to read complex legacy code.\n"
            "  ✅ You CAN ask guiding questions to help the student think."
        )
    elif allow_level == 1:
        policy = (
            f"HINT POLICY — LEVEL 1 (general area hint):\n"
            f"  ⚠️  You MAY hint that the bug might not be in `{func_name}` directly, "
            f"but could be in a function that `{func_name}` calls internally.\n"
            f"  ❌ DO NOT name the specific helper function or describe the bug type.\n"
            f"  ✅ Suggest the student trace the full call chain starting from `{func_name}`."
        )
    elif allow_level == 2:
        policy = (
            "HINT POLICY — LEVEL 2 (bug type hint):\n"
            "  ⚠️  You MAY describe the general category of the bug "
            "(e.g. 'an incorrect variable substitution', 'an off-by-one boundary', "
            "'a wrong operator', 'a subtle precedence issue').\n"
            f"  ⚠️  You MAY confirm the bug lives inside a helper called by `{func_name}`, "
            "without naming it specifically.\n"
            "  ❌ Still DO NOT name the exact function or show the fix."
        )
    else:
        target = bug_func if bug_func else f"a helper function called by {func_name}"
        policy = (
            f"HINT POLICY — LEVEL 3 (function name revealed):\n"
            f"  ⚠️  The student has used many hints. You MAY now tell them the bug is "
            f"located in `{target}`.\n"
            "  ❌ Still DO NOT show the corrected code or tell them exactly what to change.\n"
            "  ✅ Guide them to inspect that function carefully."
        )

    system_prompt = f"""You are a helpful coding mentor for a legacy code debugging challenge.

CHALLENGE CONTEXT:
- The student is debugging `{func_name}` in `{target_file}`
- The fix must be 1–3 lines (minimal change)
- The code may have confusing variable names — that is intentional

{policy}

UNIVERSAL RULES (always apply):
1. NEVER write or suggest the corrected code
2. NEVER tell the student exactly which value, operator, or variable to change
3. Be encouraging and supportive
4. Keep responses concise — 2–3 short paragraphs maximum
5. Use guiding questions rather than direct answers when possible
6. If the student asks for the solution outright, politely redirect them"""

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Reconstruct the conversation for the LLM
    lc_messages: list = [SystemMessage(content=system_prompt)]
    for msg in state["messages"]:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    try:
        resp = llm.invoke(lc_messages)
        return {**state, "response": resp.content}
    except Exception as exc:
        return {**state, "response": f"Sorry, I had trouble generating a hint: {exc}"}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_hint_graph():
    builder = StateGraph(HintState)
    builder.add_node("check_policy", node_check_hint_policy)
    builder.add_node("generate_hint", node_generate_hint)
    builder.set_entry_point("check_policy")
    builder.add_edge("check_policy", "generate_hint")
    builder.add_edge("generate_hint", END)
    return builder.compile()


# ── Convenience function for the GUI ─────────────────────────────────────────

def get_hint(
    user_message: str,
    history: list,
    hints_used: int,
    submission_attempts: int,
    challenge_info: dict,
) -> str:
    """
    Get a hint for the student.

    Args:
        user_message:        The student's current question.
        history:             List of [user_msg, assistant_msg] pairs (Gradio chatbot format).
        hints_used:          Number of hints used so far (BEFORE this call).
        submission_attempts: Number of failed submissions.
        challenge_info:      Dict with function_name, bug_func_name, target_file, difficulty_level.

    Returns:
        The assistant's hint as a string.
    """
    # Convert Gradio history to our message list.
    # Gradio 6 uses dicts {"role": ..., "content": ...};
    # older versions used [user, assistant] pairs — handle both.
    messages: list = []
    for entry in history:
        if isinstance(entry, dict):
            role    = entry.get("role", "")
            content = entry.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        else:
            human, ai = entry
            if human:
                messages.append({"role": "user",      "content": human})
            if ai:
                messages.append({"role": "assistant", "content": ai})
    messages.append({"role": "user", "content": user_message})

    graph = build_hint_graph()
    result = graph.invoke({
        "messages": messages,
        "hints_used": hints_used,
        "submission_attempts": submission_attempts,
        "challenge_info": challenge_info,
        "allow_level": 0,
        "response": "",
    })
    return result["response"]
