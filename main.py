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

from architect.graph import build_graph  # noqa: E402  (import after env setup)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy Challenge Architect — transform a GitHub repo into a student challenge."
    )
    parser.add_argument("github_url", help="Public GitHub repository URL to clone and sabotage.")
    parser.add_argument(
        "--nesting-level", type=int, default=3,
        help="Desired call-chain depth for bug placement (default: 3)"
    )
    parser.add_argument(
        "--num-bugs", type=int, default=1,
        help="Number of bugs to inject (default: 1)"
    )
    parser.add_argument(
        "--refactoring", action="store_true",
        help="Apply code obfuscation and spaghettification (default: False, only inject bugs + inflate)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode with verbose output and bug location comments (default: False)"
    )
    args = parser.parse_args()

    if args.debug:
        print(f"\n[architect] Starting — URL: {args.github_url}  Nesting: {args.nesting_level}  Bugs: {args.num_bugs}  Refactoring: {args.refactoring}  Debug: ON\n")
    else:
        print(f"\n[architect] Generating challenge from {args.github_url}...\n")

    graph = build_graph()
    initial_state = {
        "github_url": args.github_url,
        "nesting_level": args.nesting_level,
        "refactoring_enabled": args.refactoring,
        "debug_mode": args.debug,
        "num_bugs": args.num_bugs,
        # remaining fields will be populated by graph nodes
        "clone_path": "",
        "target_file": "",
        "original_code": "",
        "sabotaged_code": "",
        "function_name": "",
        "test_args": "",
        "expected_output": "",
        "actual_output": "",
        "bug_description": "",
        "detailed_explanation": "",
        "challenge_summary": "",
        "test_cases": [],
        "public_tests": [],
        "secret_tests": [],
        "candidate_files": [],
        "bug_func_name": "",
        "bug_func_source": "",
        "call_chain": [],
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
