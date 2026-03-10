"""
challenge.py — Unified Challenge entry point.

Usage (GUI setup page):
    python challenge.py [--port 7860] [--share]

Usage (skip setup, run directly):
    python challenge.py <github_url> --name "Alice" [--nesting-level 3] [--num-bugs N]
                        [--refactoring] [--debug] [--timer N] [--port 7860] [--share]

If github_url is provided the setup page is skipped: the pipeline runs
immediately and the challenge interface opens directly.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# ── Resolve .env (LegacyInterview2/.env first, then parent AiAgent/.env) ──
_base = os.path.dirname(os.path.abspath(__file__))
_env_local    = os.path.join(_base, ".env")
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

import gradio as gr                    # noqa: E402
import student_interface               # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy Challenge — launch the student GUI (with optional CLI shortcuts)."
    )
    # Optional positional — if omitted, the GUI setup page is shown instead
    parser.add_argument(
        "github_url", nargs="?", default=None,
        help="GitHub repo URL. If given, skip the setup page and run immediately.",
    )
    parser.add_argument(
        "--name", type=str, default="",
        help="Student name shown in the header. Required when github_url is provided.",
    )
    parser.add_argument(
        "--nesting-level", type=int, choices=range(1, 7), default=3, metavar="1-6",
        help="Call-chain depth for bug hiding (default: 3). Only used when github_url is provided.",
    )
    parser.add_argument(
        "--num-bugs", type=int, default=1,
        help="Number of bugs to inject (default: 1). Only used when github_url is provided.",
    )
    parser.add_argument(
        "--refactoring", action="store_true",
        help="Enable refactoring/obfuscation transforms. Only used when github_url is provided.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode — verbose output, show bug locations. Only used when github_url is provided.",
    )
    parser.add_argument(
        "--timer", type=int, default=0,
        help="Countdown timer in minutes (0 = no timer). Only used when github_url is provided.",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port for the Gradio GUI (default: 7860).",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link.",
    )
    args = parser.parse_args()

    base_kwargs = dict(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Base(),
        css=student_interface._CSS,
    )

    if args.github_url:
        # ── Fast path: CLI args supplied → skip setup page ──────────────
        if not args.name:
            parser.error("--name is required when providing a GitHub URL (e.g. --name 'Alice Smith')")
        print(
            f"\n[challenge] CLI mode — URL: {args.github_url}  "
            f"Nesting: {args.nesting_level}  Bugs: {args.num_bugs}  "
            f"Refactoring: {args.refactoring}  Debug: {args.debug}  Name: {args.name}\n"
        )
        print("[challenge] Running pipeline…")
        workspace_path = student_interface._run_pipeline(
            args.github_url.strip(), args.nesting_level, args.num_bugs,
            refactoring_enabled=args.refactoring, debug_mode=args.debug,
        )
        print(f"[challenge] Workspace ready: {workspace_path}")
        print(f"[challenge] Opening http://localhost:{args.port}\n")
        demo = student_interface.create_interface(
            workspace_path, student_name=args.name, timer_minutes=args.timer
        )
        # JS is embedded in gr.Blocks(js=_make_js(timer)) inside create_interface
        demo.launch(**base_kwargs)

    else:
        # ── Normal path: open setup page in the browser ──────────────────
        print(f"\n[challenge] Opening setup page at http://localhost:{args.port}\n")
        demo = student_interface.create_full_interface()
        demo.launch(**base_kwargs, js=student_interface._JS)


if __name__ == "__main__":
    main()
