# Legacy Code Challenge System

An AI-powered system that turns any public GitHub repository into a live debugging challenge. The instructor runs one command; the student gets a Gradio web interface where they read the mission, edit the code, ask an AI assistant for hints, and submit their fix for automated scoring.

## 🎯 Overview

1. **Clone** a GitHub repository
2. **Analyse** the code with AST scoring to pick the best file/function
3. **Inject a bug** using GPT-4o with configurable difficulty
4. **Deploy** test cases and a student README into the workspace
5. **Launch** a Gradio web interface for the student

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

*(Intel corporate network only — uncomment the proxy lines in `challenge.py`)*

---

## 🚀 Quick Start

### Option A — GUI setup page (instructor fills in details in the browser)

```bash
python challenge.py
```

Opens a setup form at `http://localhost:7860`. Fill in the GitHub URL, difficulty level, number of bugs and student name, then click **Start Challenge**.

### Option B — CLI (skip setup page, go straight to the challenge)

```bash
python challenge.py <github_url> --name "Alice Smith" --level 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | *(required with URL)* | Student name shown in the header |
| `--level` | `1` | Difficulty: 1 = Messy Code, 2 = Spaghetti Logic, 3 = Sensitive Code |
| `--num-bugs` | `1` | Number of bugs to inject |
| `--port` | `7860` | Gradio port |
| `--share` | off | Generate a public Gradio share link |

**Example:**

```bash
python challenge.py https://github.com/mahmoud/boltons --name "Bob" --level 2 --num-bugs 1
```

---

## 🎓 Student Interface

The Gradio UI has four tabs on the left and an AI assistant panel on the right.

### Left tabs

| Tab | Description |
|-----|-------------|
| **📋 Challenge** | The student README — mission briefing, bug report, constraints |
| **💻 Code Editor** | Edit the target file. **💾 Save** writes to disk. Search bar with ◀ / 🔍 Find / ▶ navigation. |
| **🧪 Test Results** | Run the test suite; coloured PASS / FAIL / CRASH output. |
| **👁️ My Changes** | Unified diff of the saved file vs. the buggy original. **🔄 Refresh** re-reads the file; **↩️ Revert** restores the original. |

> **Tip:** Editing in your IDE and saving there is fully supported — the Gradio interface never auto-saves over your changes.

### Right panel

- **AI Assistant** — progressive hints (see below)
- **🚀 Submit Fix** — runs the test suite on the saved file, computes the score, and shows the Results page

### Results page

After submitting, a dedicated results page shows:

- Score card (total / 100)
- Test Results tab — per-case PASS / FAIL / CRASH
- Your Changes tab — diff of submitted code vs. buggy original
- Expected Fix tab — diff of buggy function vs. correct original (bug function only)
- Hints Used tab — full AI conversation log

---

## 🤖 AI Assistant & Hint Policy

The assistant uses a **progressive hint system** — it reveals more information the more hints the student has already used:

| Hints used | What the AI may say |
|-----------|---------------------|
| 0 | General debugging strategy only |
| 1–2 | May hint the bug is not in the surface function |
| 3–4 | May describe the bug category (off-by-one, wrong operator, …) |
| 5+ | May name the specific helper function |

The assistant **never** writes corrected code or names the exact line to change.

**Hint counting is smart**: only messages where the AI gave substantive debugging guidance increment the hint counter. Acknowledgements and encouragements do not count.

---

## 🎯 Scoring

| Component | Points | Details |
|-----------|--------|---------|
| Test score | 0 – 80 | Proportional to test cases passed |
| Diff score | 0 – 20 | Semantic similarity to the original correct code |
| Hint penalty | − 0 to 30 | −2 / −6 / −12 / −20 / −30 cumulative after 1–5 real hints |

**Total = max(0, test\_score + diff\_score − hint\_penalty)**

---

## 📁 Project Structure

```
LegacyInterview2/
├── challenge.py               # Unified entry point (GUI or CLI)
├── main.py                    # Architect-only entry point (no GUI)
├── student_interface.py       # Gradio web interface
├── requirements.txt
├── .env                       # OPENAI_API_KEY (create this)
├── architect/                 # Bug-generation pipeline
│   ├── graph.py               # LangGraph 6-node workflow
│   ├── state.py               # ArchitectState TypedDict
│   ├── repo_cloner.py         # git clone via GitPython
│   ├── file_mapper.py         # AST scoring → target file selection
│   ├── saboteur.py            # GPT-4o bug injection & obfuscation
│   ├── challenge_deployer.py  # Writes challenge_run.py
│   └── readme_generator.py   # Writes STUDENT_README.md
├── orchestrator/              # Scoring & hint sub-system
│   ├── scoring.py             # run_tests(), evaluate_submission()
│   └── hint_graph.py         # LangGraph hint sub-graph
└── workspaces/                # Auto-created; one folder per challenge
    └── <repo_name>/
        ├── <target_file>.py   # Sabotaged code (student edits this)
        ├── challenge_run.py   # Test runner
        ├── STUDENT_README.md  # Challenge briefing
        ├── challenge_state.json  # Metadata for scoring
        └── submissions/       # Per-submission JSON logs
```

---

## 🔧 Difficulty Levels

| Level | Name | Sabotage Strategy |
|-------|------|-------------------|
| 1 | Messy Code | Rename internals to `var1`/`temp_x`, add misleading comments, one off-by-one or wrong-operator bug |
| 2 | Spaghetti Logic | Level 1 + nested-if confusion, global variable misnaming |
| 3 | Sensitive Code | Level 1 + corrupt a magic number, comments describe the *correct* formula |

---

## 🔒 Security Notes

- Student code runs in the same Python process — use Docker for production
- The AI assistant is prompted to never give the solution; determined students might try creative phrasing
- Submissions are logged locally in `workspaces/<repo>/submissions/`

---

## 🛠️ Customisation

| What | Where |
|------|-------|
| Difficulty prompt wording | `architect/saboteur.py` — `_LEVEL_INSTRUCTIONS` |
| Scoring weights | `orchestrator/scoring.py` |
| Hint policy thresholds | `orchestrator/hint_graph.py` — `node_check_hint_policy` |
| Hint reveal levels | `orchestrator/hint_graph.py` — `node_generate_hint` system prompt |

---

## 🐛 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` in test runner | Repo has external deps not in the student env — pick a pure-Python repo |
| "No suitable file found" | Repo lacks functions with return values and arithmetic — try a different repo |
| All test cases fail after generation | GPT produced bad test args — delete the workspace and re-run |
| `OPENAI_API_KEY not set` | Create `.env` with your key or `export OPENAI_API_KEY=...` |
| Corporate proxy errors | Uncomment the proxy lines in `challenge.py` |

---

## 📝 Instructor Tips

1. **Pre-generate** one workspace per student before the session starts
2. Send students the direct CLI command with their name already filled in
3. Review `submissions/*.json` after the session — each has the full code snapshot and score breakdown
4. Use `--level 1` for newcomers, `--level 3` for experienced developers
5. Pure-Python utility libraries (e.g. `boltons`, `more-itertools`) make the best targets

---

Built with **LangGraph**, **LangChain**, **OpenAI GPT-4o**, **Gradio**, and **GitPython**.

**Happy Debugging! 🐛 ➜ ✨**
