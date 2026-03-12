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

# Cumulative penalty table — index = number of hints used (mirrors scoring.py)
_HINT_PENALTY = [0, 2, 6, 12, 20, 30]
MAX_HINTS = 5


# ── State ─────────────────────────────────────────────────────────────────────

class HintState(TypedDict):
    messages: list           # [{"role": "user"|"assistant", "content": str}, ...]
    hints_used: int          # total AI hints consumed so far this session
    submission_attempts: int # number of times student has submitted
    challenge_info: dict     # keys: function_name, bug_func_name, target_file, difficulty_level
    allow_level: int         # 0–3, set by node_check_hint_policy
    response: str            # filled by node_generate_hint
    gave_hint: bool          # True if the AI gave substantive debugging guidance
    hint_summary: str        # one-sentence summary of the hint (empty if gave_hint=False)


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
    hints_used = state["hints_used"]

    # Hard limit: no more hints once MAX_HINTS is reached
    if hints_used >= MAX_HINTS:
        return {**state,
                "response": (
                    f"You've reached the maximum of {MAX_HINTS} hints for this challenge. "
                    "I can no longer provide debugging hints, but I'm still happy to help "
                    "with questions about the UI, how to run tests, or how to navigate the code editor."
                ),
                "gave_hint": False}

    allow_level = state["allow_level"]
    info = state["challenge_info"]
    func_name = info.get("function_name", "the function")
    target_file = info.get("target_file", "the target file")

    bug_func_names: list = info.get("bug_func_names", [])
    if not bug_func_names:
        single = info.get("bug_func_name", "")
        bug_func_names = [single] if single else []

    sabot_sources: list = info.get("bug_func_sources_list", [])
    orig_sources:  list = info.get("original_bug_func_sources_list", [])
    original_code: str  = info.get("original_code", "")

    # Build a file overview from the original code (module docstring + first ~40 lines)
    if original_code:
        overview_lines = original_code.splitlines()[:60]
        file_overview = (
            "\n\nFILE OVERVIEW (original, unmodified file — use this to understand "
            "what the module does and answer general questions about it):\n"
            + "\n".join(overview_lines)
            + ("\n..." if len(original_code.splitlines()) > 60 else "")
        )
    else:
        file_overview = ""

    # Build the internal bug reference block (never shown verbatim to student)
    bug_blocks: list[str] = []
    for i, fn in enumerate(bug_func_names, 1):
        sabot = sabot_sources[i - 1] if i - 1 < len(sabot_sources) else ""
        orig  = orig_sources[i - 1]  if i - 1 < len(orig_sources)  else ""
        block = f"Bug #{i} — function `{fn}`:"
        if sabot:
            block += f"\n  SABOTAGED (what the student sees):\n{sabot}"
        if orig:
            block += f"\n  CORRECT (what it should look like):\n{orig}"
        bug_blocks.append(block)

    bug_reference = (
        "\n\nBUG REFERENCE (for your eyes only — use this to understand the bug and craft hints):\n"
        + "\n\n".join(bug_blocks)
        if bug_blocks else ""
    )

    # Build the hint-policy section of the system prompt.
    # ALL levels must use the BUG REFERENCE to craft a targeted hint — never give generic advice.
    primary_bug_func = bug_func_names[0] if bug_func_names else ""
    if allow_level == 0:
        policy = (
            f"HINT POLICY — LEVEL 0 (1st hint — subtle, behavior-level clue):\n"
            f"  Use the BUG REFERENCE to understand the actual bug. Then craft a hint that:\n"
            f"  ✅ Focuses on the OBSERVABLE BEHAVIOR that is wrong "
            f"(e.g. 'think about what the function should return when the input is X').\n"
            f"  ✅ Asks the student a targeted question about the specific behavior the bug affects "
            f"— without saying where in the code the problem is.\n"
            f"  ❌ DO NOT mention which function contains the bug.\n"
            f"  ❌ DO NOT name the bug category (no 'off-by-one', 'wrong operator', etc.).\n"
            f"  ❌ NEVER give generic debugging advice like 'trace the flow' or 'check edge cases' "
            f"— the hint must be specific to THIS code and THIS bug."
        )
    elif allow_level == 1:
        policy = (
            f"HINT POLICY — LEVEL 1 (2nd hint — structural clue):\n"
            f"  Use the BUG REFERENCE to understand the actual bug. Then craft a hint that:\n"
            f"  ✅ Tells the student the bug is NOT in `{func_name}` itself, but in a "
            f"helper function that `{func_name}` calls internally.\n"
            f"  ✅ Points the student toward the specific part of `{func_name}`'s behavior "
            f"that is broken, so they can trace which helper is responsible.\n"
            f"  ❌ DO NOT name the helper function.\n"
            f"  ❌ DO NOT describe the bug type yet."
        )
    elif allow_level == 2:
        bug_category_hint = ""
        if orig_sources and sabot_sources:
            bug_category_hint = (
                f"  ✅ Based on the BUG REFERENCE you can see the difference — describe the "
                f"general category of the bug "
                f"(e.g. 'an off-by-one boundary', 'a wrong comparison operator', "
                f"'a subtle variable mix-up', 'an incorrect arithmetic operation').\n"
            )
        policy = (
            f"HINT POLICY — LEVEL 2 (3rd hint — bug category revealed):\n"
            f"  Use the BUG REFERENCE to understand the actual bug. Then craft a hint that:\n"
            + bug_category_hint +
            f"  ✅ Confirms the bug is inside a helper called by `{func_name}` "
            f"(without naming it).\n"
            f"  ❌ Still DO NOT name the exact function."
        )
    else:
        target = primary_bug_func if primary_bug_func else f"a helper function called by {func_name}"
        policy = (
            f"HINT POLICY — LEVEL 3 (4th–5th hint — function name revealed):\n"
            f"  The student has used many hints. Now craft a hint that:\n"
            f"  ✅ Names the specific function: tell the student the bug is in `{target}`.\n"
            f"  ✅ Describes what to look for inside that function "
            f"(without showing the fix or naming the exact line).\n"
            f"  ❌ Still DO NOT show corrected code or say exactly what to change."
        )

    hints_remaining = MAX_HINTS - hints_used
    current_penalty = _HINT_PENALTY[min(hints_used, len(_HINT_PENALTY) - 1)]
    next_penalty    = _HINT_PENALTY[min(hints_used + 1, len(_HINT_PENALTY) - 1)]
    penalty_schedule = " → ".join(f"−{p}" for p in _HINT_PENALTY[1:])

    # ── Detect confirmation-then-response flow (must run before system_prompt is built) ──
    def _msg_text(msg) -> str:
        c = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(c, list):
            c = " ".join(str(p) for p in c if p)
        return str(c) if c else ""

    last_user_msg_raw  = _msg_text(state["messages"][-1]).strip() if state["messages"] else ""
    last_user_msg      = last_user_msg_raw.lower()

    # Detect the language from the user's message so the LLM can be instructed explicitly.
    def _detect_language(text: str) -> str:
        """Return a human-readable language name based on the script used."""
        for ch in text:
            cp = ord(ch)
            if 0x0590 <= cp <= 0x05FF:   # Hebrew block
                return "Hebrew"
            if 0x0600 <= cp <= 0x06FF:   # Arabic block
                return "Arabic"
            if 0x0400 <= cp <= 0x04FF:   # Cyrillic block
                return "Russian"
            if 0x4E00 <= cp <= 0x9FFF:   # CJK block
                return "Chinese"
        return "English"

    _user_language = _detect_language(last_user_msg_raw)

    # Detect explicit DECLINES — multilingual (English + Hebrew + other common).
    _declines = {
        # English
        "no", "nope", "nah", "not now", "nevermind", "never mind",
        "no thanks", "no thank you", "skip", "cancel", "forget it",
        "don't", "dont", "no hint", "stop",
        # Hebrew
        "לא", "לא תודה", "לא רוצה", "לא צריך", "לא עכשיו", "בטל", "עצור",
        # Other common
        "non", "nein", "нет", "لا",
    }
    _is_decline = (
        last_user_msg in _declines
        or last_user_msg.startswith("no ")
        or last_user_msg.startswith("don't")
        or last_user_msg.startswith("dont")
        or last_user_msg.startswith("nope")
        or last_user_msg.startswith("nah ")
        or last_user_msg.startswith("לא ")
    )

    _prev_was_confirmation = False
    for _m in reversed(state["messages"][:-1]):
        if isinstance(_m, dict) and _m.get("role") == "assistant":
            if "gave_hint" in _m:
                _prev_was_confirmation = not _m["gave_hint"]
            else:
                _cl = _msg_text(_m).lower()
                _prev_was_confirmation = any(kw in _cl for kw in
                    ["would you like", "proceed", "penalty to your score"])
            break

    # When the previous AI message was a confirmation question, the student's current
    # response IS their answer to it. Give the hint unless they explicitly declined.
    _student_accepted_hint = _prev_was_confirmation and not _is_decline

    _confirmed_hint_instruction = (
        "\n\n⚠️ CONFIRMED HINT: The student has just responded to your confirmation request "
        "and accepted the hint. Do NOT ask for confirmation again. "
        "Give the hint now and mark GAVE_HINT:YES."
        if _student_accepted_hint else ""
    )

    system_prompt = f"""You are a helpful coding mentor for a legacy code debugging challenge.

LANGUAGE RULE (mandatory — never violate):
The student's most recent message is written in {_user_language}.
You MUST respond entirely in {_user_language}. Do not use any other language, even partially.

CHALLENGE CONTEXT:
- There may be multiple bugs — use the BUG REFERENCE below to know exactly what they are
- The fix for each bug is 1–3 lines (minimal change)
- The code may have confusing variable names — that is intentional{file_overview}{bug_reference}

SCORING & PENALTY RULES (share this with the student if they ask):
- Hints cause a cumulative point deduction from their final score
- Penalty schedule (cumulative): {penalty_schedule} pts for hints 1–{MAX_HINTS}
- Current status: {hints_used} hint(s) used → current penalty is −{current_penalty} pts
- Hints remaining: {hints_remaining} (maximum is {MAX_HINTS} total)
- If they use the next hint, penalty becomes −{next_penalty} pts

CHALLENGE INTERFACE — you know how the UI works and can guide the student:
- **📋 Challenge tab** — shows the challenge README with mission, constraints, and scoring info.
- **💻 Code Editor tab** — browse and edit any file in the workspace. Use the file dropdown to
  switch files. Click 💾 Save to save changes. `challenge_run.py` is also visible here.
- **🧪 Test Results tab** — click ▶ Run Tests to execute the public test suite and see pass/fail output.
- **👁️ My Changes tab** — shows a diff of all edits made since the challenge started.
- **🚀 Submit Fix button** — submits the current saved code for final AI evaluation.
- The student can also edit files in their local IDE; changes are picked up automatically on submit.
- To run tests locally: `python challenge_run.py` from the workspace directory.

You can help the student with any of the above — navigating the UI, running tests, saving files,
understanding what a tab shows, or how to submit. These answers are free and do not count as hints.

{policy}

UNIVERSAL RULES (always apply):
1. NEVER write or suggest the corrected code — use the BUG REFERENCE only to understand the bug
2. NEVER tell the student exactly which value, operator, or variable to change
3. Be encouraging and supportive
4. Keep responses concise — 2–3 short paragraphs maximum
5. Use guiding questions rather than direct answers when possible
6. NO REPEATED HINTS — before writing a hint, review every previous assistant message in this
   conversation. If you already pointed the student in a certain direction (e.g. "trace the call
   chain", "look at boundary conditions"), do NOT repeat the same advice. Each new hint must
   add new, progressive information that moves the student one step closer to the bug. If you
   have nothing new to add within the current allow_level, say so honestly and encourage the
   student to try what was already suggested before asking for another hint.

HINT CONFIRMATION RULE (critical):
- Before giving any substantive debugging hint (GAVE_HINT:YES), you MUST first ask for confirmation —
  UNLESS the most recent assistant message in the conversation (the last AI turn above) was already
  a confirmation question for THIS same request.
- Confirmation message to use (word for word at the relevant language):
    "I can give you a hint, but keep in mind it will add −{next_penalty} pts penalty to your score
     (you've used {hints_used} hint(s) so far, {hints_remaining} remaining).
     Would you like me to proceed?"
  Do NOT give the actual hint in that response — mark it GAVE_HINT:NO.
- If the last AI message was already asking for confirmation AND the student's current message is
  an affirmation ("yes", "go ahead", "sure", "proceed", "ok", etc.), then give the hint directly
  without asking again.
- Each new hint request requires a fresh confirmation — the student saying "yes" once does NOT
  give blanket permission for all future hints.
- If the student asks about the UI, navigation, how to run tests, or other non-debugging questions,
  answer directly — no confirmation needed, these are free.

HINT TRACKING:
At the very end of your response, write the following lines (each on its own line):
1. Exactly one of:
     GAVE_HINT:YES  — when you provide substantive debugging guidance (including the response
                      where the student confirmed and you are now giving the actual hint).
     GAVE_HINT:NO   — ONLY when you:
                      (a) asked a confirmation question and did NOT yet give the hint, OR
                      (b) answered a UI/navigation/scoring question only, OR
                      (c) gave only encouragement with zero debugging content
2. If and only if GAVE_HINT:YES, also write on the next line:
     HINT_SUMMARY: <one short sentence (max 120 chars) summarising what aspect of the bug you hinted at>
   Do NOT write HINT_SUMMARY if GAVE_HINT:NO.{_confirmed_hint_instruction}"""

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
        raw = resp.content
        # Parse and strip GAVE_HINT and HINT_SUMMARY markers from the tail
        gave_hint    = False
        hint_summary = ""
        lines = raw.rstrip().split("\n")
        # Strip HINT_SUMMARY line (appears after GAVE_HINT)
        if lines and lines[-1].strip().startswith("HINT_SUMMARY:"):
            hint_summary = lines[-1].strip()[len("HINT_SUMMARY:"):].strip()
            lines = lines[:-1]
        # Strip GAVE_HINT line
        if lines and lines[-1].strip().startswith("GAVE_HINT:"):
            gave_hint = lines[-1].strip() == "GAVE_HINT:YES"
            lines = lines[:-1]
        raw = "\n".join(lines).rstrip()
        # Python-level guarantee: if the student accepted the hint (responded to
        # a confirmation and didn't decline), always count it regardless of LLM.
        if _student_accepted_hint:
            gave_hint = True
        return {**state, "response": raw, "gave_hint": gave_hint, "hint_summary": hint_summary}
    except Exception as exc:
        return {**state, "response": f"Sorry, I had trouble generating a hint: {exc}", "gave_hint": False, "hint_summary": ""}


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
) -> dict:
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
                msg: dict = {"role": role, "content": content}
                # Preserve gave_hint metadata so node_generate_hint can detect
                # whether the previous AI turn was a confirmation question.
                if "gave_hint" in entry:
                    msg["gave_hint"] = entry["gave_hint"]
                messages.append(msg)
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
        "gave_hint": False,
        "hint_summary": "",
    })
    return {
        "response":     result["response"],
        "gave_hint":    result["gave_hint"],
        "hint_summary": result.get("hint_summary", ""),
    }
