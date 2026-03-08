"""
challenge.py — Unified Challenge entry point.

Usage:
    python challenge.py <github_url> [--level 1|2|3] [--num-bugs N] [--port 7860] [--share]

This runs the full pipeline:
  1. Bug Generator Agent (clones repo, injects bug, creates challenge_run.py + README)
  2. Saves challenge_state.json to the workspace
  3. Launches the student Gradio GUI at http://localhost:<port>
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# ── Resolve .env (LegacyInterview2/.env first, then parent AiAgent/.env) ──
_base = os.path.dirname(os.path.abspath(__file__))
_env_local = os.path.join(_base, ".env")
_env_fallback = os.path.join(_base, "..", "First Year", "Advanced Deep Neural Neworks",
                              "Home Exercises", "AiAgent", ".env")
if os.path.exists(_env_local):
    load_dotenv(dotenv_path=_env_local)
elif os.path.exists(_env_fallback):
    load_dotenv(dotenv_path=_env_fallback)
else:
    load_dotenv()

# ── Intel proxy (uncomment if needed on corporate network) ──
# os.environ["http_proxy"]  = "http://proxy-iil.intel.com:912"
# os.environ["https_proxy"] = "http://proxy-iil.intel.com:912"
# os.environ["no_proxy"]    = "localhost,127.0.0.1"

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

from orchestrator.graph import build_orchestrator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy Challenge — generate a bug challenge and launch the student GUI."
    )
    parser.add_argument("github_url", help="Public GitHub repository URL to clone and sabotage.")
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3], default=1,
        help="Difficulty level: 1=Messy Code, 2=Spaghetti Logic, 3=Sensitive Code (default: 1)"
    )
    parser.add_argument(
        "--num-bugs", type=int, default=1,
        help="Number of bugs to inject (default: 1)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port for the Gradio student GUI (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link"
    )
    args = parser.parse_args()

    print(f"\n[challenge] Starting — URL: {args.github_url}  Level: {args.level}  "
          f"Bugs: {args.num_bugs}  Port: {args.port}\n")

    graph = build_orchestrator()
    initial_state = {
        "github_url":       args.github_url,
        "difficulty_level": args.level,
        "num_bugs":         args.num_bugs,
        "port":             args.port,
        "share":            args.share,
        # remaining fields will be populated by graph nodes
        "workspace_path":       "",
        "target_file":          "",
        "original_code":        "",
        "sabotaged_code":       "",
        "function_name":        "",
        "bug_func_name":        "",
        "bug_func_source":      "",
        "test_cases":           [],
        "bug_description":      "",
        "challenge_state_path": "",
        "launch_status":        "",
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
