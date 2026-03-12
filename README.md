# Legacy Code Challenge System

An AI-powered system that transforms any public GitHub repository into a live debugging challenge. It clones a repo, injects realistic bugs into deeply-nested call chains, and presents students with a Gradio web interface — complete with a code editor, test runner, AI hint assistant, and LLM-based submission evaluator.

---

## Overview

The system takes a GitHub repository URL and:

1. **Clones** the repository
2. **Analyzes** code structure using AST to find functions at a target call-chain depth
3. **Injects bugs** (GPT-4o) into helper functions buried in the call chain
4. **Inflates** the call chain with dummy code to obscure the bugs
5. **Optionally obfuscates** code with variable renaming and misleading comments
6. **Generates test cases** (public + secret) and a student README
7. **Launches a Gradio web interface** where students debug, test, and submit

---

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

> For Intel corporate network, uncomment the proxy lines near the top of `challenge.py`.

---

## Quick Start

### GUI mode (setup page in browser)

```bash
python challenge.py
```

Opens at `http://localhost:7860`. Fill in the GitHub URL and settings in the browser, then click **Start Challenge**.

### CLI mode (skip setup, run immediately)

```bash
# Default settings
python challenge.py https://github.com/mahmoud/boltons.git

# Full example
python challenge.py https://github.com/mahmoud/boltons.git \
    --name "Alice Smith" \
    --nesting-level 5 \
    --num-bugs 3 \
    --refactoring \
    --debug \
    --timer 45 \
    --port 7860
```

**Parameters:**

| Flag | Default | Description |
|---|---|---|
| `--name` | `""` | Student name shown in the header |
| `--nesting-level N` | `3` | Call-chain depth (1–6). Higher = harder |
| `--num-bugs N` | `1` | Number of bugs to inject (1–5) |
| `--refactoring` | off | Obfuscate variables and add misleading comments |
| `--debug` | off | Show call-chain visualization and verbose logs |
| `--timer N` | `0` | Challenge countdown in minutes (0 = no timer) |
| `--port N` | `7860` | Gradio server port |
| `--share` | off | Create a public Gradio share link |

---

## Project Structure

```
LegacyInterview2/
├── challenge.py             # Unified entry point (GUI + CLI)
├── student_interface.py     # Gradio web interface
├── requirements.txt
├── .env                     # API keys (create this)
│
├── architect/               # Challenge generation pipeline
│   ├── graph.py             # LangGraph workflow
│   ├── state.py             # ArchitectState TypedDict
│   ├── repo_cloner.py       # GitHub cloning
│   ├── file_mapper.py       # AST call-graph analysis
│   ├── saboteur.py          # Bug injection & function inflation
│   ├── challenge_deployer.py# Test case generation
│   ├── readme_generator.py  # Student README generation
│   └── nodes.py             # Workflow node functions
│
├── orchestrator/            # Runtime evaluation
│   ├── graph.py             # Submission evaluation workflow
│   ├── hint_graph.py        # AI hint system (LangGraph)
│   ├── scoring.py           # Test runner & score calculation
│   ├── nodes.py             # Evaluation node functions
│   └── state.py             # OrchestratorState TypedDict
│
└── workspaces/              # Generated challenges (auto-created)
    └── <repo_name>/
        ├── <repo>/                    # Cloned repo with sabotaged files
        ├── challenge_run.py           # Public test runner
        ├── challenge_run_secret.py    # Secret test runner
        ├── STUDENT_README.md          # Challenge instructions
        ├── detailed_explanation.txt   # Instructor reference
        └── challenge_state.json       # Metadata & submission log
```

---

## Student Interface

### Pages

**Setup page** (GUI mode only)
- Enter GitHub URL, student name, nesting level, number of bugs
- Enable refactoring and/or debug mode
- Set an optional countdown timer
- Click **Start Challenge** — pipeline runs (~1–2 min)

**Challenge page**
- **📋 Challenge tab** — mission briefing, function signature, expected behavior, scoring rules
- **💻 Code Editor tab** — browse and edit any file; 💾 Save writes to disk
- **🧪 Test Results tab** — run the public test suite and see coloured pass/fail output
- **👁️ My Changes tab** — diff of all edits vs. the original sabotaged code; revert option
- **AI Assistant** (right panel) — ask for hints; see live hint counter and penalty
- **🚀 Submit Fix** — triggers LLM evaluation and moves to the Results page

**Results page**
- Score summary (LLM evaluation + test results + hint penalty)
- Side-by-side diff: your fix vs. expected fix
- Hints Used tab — one-line AI-generated summary per hint consumed

---

## Scoring System

Scoring combines automated tests with LLM code review.

### Hint penalty schedule (cumulative)

| Hints used | Penalty |
|---|---|
| 0 | −0 pts |
| 1 | −2 pts |
| 2 | −6 pts |
| 3 | −12 pts |
| 4 | −20 pts |
| 5 | −30 pts |

Maximum **5 hints** per session. The AI assistant asks for confirmation before each hint and clearly states the penalty cost.

### Final score

```
Final Score = max(0, LLM_Score − Hint_Penalty)
```

---

## AI Hint System

The hint assistant (`orchestrator/hint_graph.py`) is a LangGraph agent that:

- **Requires confirmation** before each hint and states the exact penalty
- Gives **progressive hints** based on how many have been used:
  - Hint 1 — observable behavior clue (no code location)
  - Hint 2 — tells the student the bug is in a helper, not the surface function
  - Hint 3 — reveals the bug category (off-by-one, wrong operator, etc.)
  - Hints 4–5 — names the specific function containing the bug
- **Never repeats** a hint already given in the session
- **Never shows** corrected code or exact line numbers
- Responds in the **student's language** (auto-detected from their message — Hebrew, Arabic, English, etc.)
- Appends a short `HINT_SUMMARY:` to each hint for display in the Results tab

---

## Nesting Level Guide

| Level | Description | Recommended for |
|---|---|---|
| 1–2 | Surface function calls one helper directly | Beginners |
| 3–4 | Multi-layer call chain with intermediate helpers | Intermediate |
| 5–6 | Deep hierarchy; bug buried 5–6 layers down | Advanced |

If the target depth doesn't naturally exist in the repo, the system auto-creates intermediate wrapper functions to reach it.

---

## Example Workflows

### Instructor

```bash
# Easy challenge
python challenge.py https://github.com/mahmoud/boltons.git --nesting-level 2

# Hard challenge with obfuscation, 3 bugs, 45-minute timer
python challenge.py https://github.com/mahmoud/boltons.git \
    --nesting-level 5 --num-bugs 3 --refactoring --timer 45

# Verify the generated challenge
cat workspaces/boltons/detailed_explanation.txt
python workspaces/boltons/challenge_run.py
```

### Student (CLI pre-launch)

```bash
python challenge.py https://github.com/mahmoud/boltons.git --name "Alice"
# Opens browser → read instructions → edit code → run tests → submit
```

---

## Recommended Repositories

| Repository | Why it works well |
|---|---|
| [mahmoud/boltons](https://github.com/mahmoud/boltons) | Pure Python utilities, rich call chains |
| [more-itertools/more-itertools](https://github.com/more-itertools/more-itertools) | Clean iterator logic |
| [jsonpickle/jsonpickle](https://github.com/jsonpickle/jsonpickle) | Good nesting depth |
| [python-attrs/attrs](https://github.com/python-attrs/attrs) | Well-structured helpers |

**Avoid:** repos with C extensions, web frameworks (Django/Flask), or heavy external dependencies.

---

## Security Notes

- Student code runs in the same Python environment — use Docker for production
- All submissions are logged to `workspaces/<repo>/challenge_state.json`

---

## Credits

Built with:
- **LangChain + LangGraph** — AI workflow orchestration
- **OpenAI GPT-4o** — bug injection, evaluation, and hint generation
- **Gradio** — web interface
- **GitPython** — repository management
- **Python AST** — code structure analysis

---

**Happy Debugging!**
