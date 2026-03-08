from typing import TypedDict


class ArchitectState(TypedDict):
    github_url: str
    clone_path: str        # absolute path of cloned repo on disk
    target_file: str       # absolute path to the chosen Python file
    original_code: str     # file content before sabotage
    sabotaged_code: str    # file content after sabotage
    difficulty_level: int  # 1, 2, or 3
    num_bugs: int          # number of bugs to inject (CLI --num-bugs, default 1)
    function_name: str     # the sabotaged function
    test_args: str         # first failing case args (for README / legacy use)
    expected_output: str   # correct return value for first failing case
    actual_output: str     # buggy return value for first failing case
    bug_description: str   # architect-only description of what was injected
    challenge_summary: str # final printout for the architect
    test_cases: list       # [{"args": str, "expected": str}, ...] verified cases
    candidate_files: list  # all scored file paths (sorted best-first) for fallback
    bug_func_name: str     # internal: name of the sabotaged helper function (passed between nodes)
    bug_func_source: str   # internal: exact buggy source of the helper (passed between nodes)
