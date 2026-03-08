"""
challenge.py — Unified Challenge entry point.

Usage (GUI setup page):
    python challenge.py [--port 7860] [--share]

Usage (skip setup, run directly):
    python challenge.py <github_url> [--level 1|2|3] [--num-bugs N] [--name "Alice"] [--port 7860] [--share]

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
        "--level", type=int, choices=[1, 2, 3], default=1,
        help="Difficulty level (default: 1). Only used when github_url is provided.",
    )
    parser.add_argument(
        "--num-bugs", type=int, default=1,
        help="Number of bugs to inject (default: 1). Only used when github_url is provided.",
    )
    parser.add_argument(
        "--name", type=str, default="",
        help="Student name shown in the header. Only used when github_url is provided.",
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

    launch_kwargs = dict(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Base(),
        css=student_interface._CSS,
        js=student_interface._JS,
    )

    if args.github_url:
        # ── Fast path: CLI args supplied → skip setup page ──────────────
        print(
            f"\n[challenge] CLI mode — URL: {args.github_url}  "
            f"Level: {args.level}  Bugs: {args.num_bugs}\n"
        )
        print("[challenge] Running pipeline…")
        workspace_path = student_interface._run_pipeline(
            args.github_url.strip(), args.level, args.num_bugs
        )
        print(f"[challenge] Workspace ready: {workspace_path}")
        print(f"[challenge] Opening http://localhost:{args.port}\n")
        demo = student_interface.create_interface(workspace_path)
        # Patch header to include name if supplied
        # (create_interface sets its own header; name display is a nice-to-have)
        demo.launch(**launch_kwargs)

    else:
        # ── Normal path: open setup page in the browser ──────────────────
        print(f"\n[challenge] Opening setup page at http://localhost:{args.port}\n")
        demo = student_interface.create_full_interface()
        demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
