from typing import TypedDict


class ArchitectState(TypedDict):
    github_url: str
    clone_path: str        # absolute path of cloned repo on disk
    target_file: str       # absolute path to the chosen Python file
    original_code: str     # file content before sabotage
    sabotaged_code: str    # file content after sabotage
    nesting_level: int     # desired call-chain depth for bug placement
    refactoring_enabled: bool  # whether to apply obfuscation/spaghettification
    debug_mode: bool       # whether to show verbose output and bug location comments
    num_bugs: int          # number of bugs to inject (CLI --num-bugs, default 1)
    function_name: str     # the sabotaged function
    test_args: str         # first failing case args (for README / legacy use)
    expected_output: str   # correct return value for first failing case
    actual_output: str     # buggy return value for first failing case
    bug_description: str   # architect-only description of what was injected
    detailed_explanation: str  # detailed report: where bug is, what it does, how it's hidden
    challenge_summary: str # final printout for the architect
    test_cases: list       # [{"args": str, "expected": str}, ...] all test cases (10)
    public_tests: list     # [{"args": str, "expected": str}, ...] 5 public tests
    secret_tests: list     # [{"args": str, "expected": str}, ...] 5 secret tests
    candidate_files: list  # all scored file paths (sorted best-first) for fallback
    bug_func_name: str     # internal: name of the sabotaged helper function (passed between nodes)
    bug_func_source: str   # internal: exact buggy source of the helper (passed between nodes)
    call_chain: dict       # internal: dict of bug_func_name -> [surface_func, ..., bug_func] call chains
    bug_func_names: list                  # all sabotaged helper names (one per injected bug)
    bug_func_sources_list: list           # all sabotaged helper sources (one per injected bug)
    original_bug_func_sources_list: list  # all original helper sources (one per injected bug)
