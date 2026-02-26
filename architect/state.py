from typing import TypedDict


class ArchitectState(TypedDict):
    github_url: str
    clone_path: str        # absolute path of cloned repo on disk
    target_file: str       # absolute path to the chosen Python file
    original_code: str     # file content before sabotage
    sabotaged_code: str    # file content after sabotage
    difficulty_level: int  # 1, 2, or 3
    function_name: str     # the sabotaged function
    test_args: str         # example call as a string, e.g. "10, 5"
    expected_output: str   # correct return value as string
    actual_output: str     # buggy return value as string
    bug_description: str   # architect-only description of what was injected
    challenge_summary: str # final printout for the architect
