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
        "--level", type=int, choices=[1, 2, 3], default=1,
        help="Difficulty level: 1=Messy Code, 2=Spaghetti Logic, 3=Sensitive Code (default: 1)"
    )
    args = parser.parse_args()

    print(f"\n[architect] Starting — URL: {args.github_url}  Level: {args.level}\n")

    graph = build_graph()
    initial_state = {
        "github_url": args.github_url,
        "difficulty_level": args.level,
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
        "challenge_summary": "",
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
