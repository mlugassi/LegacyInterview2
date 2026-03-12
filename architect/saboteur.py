import ast
import difflib
import json
import os
import random
import textwrap

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from architect.state import ArchitectState

# Maximum call-chain depth to descend when picking bug targets
# Level 1 → depth 2 (surface → helper)
# Level 2 → depth 3 (surface → mid → helper)
# Level 3 → depth 4 (surface → mid → mid2 → helper)
_MAX_DEPTH_LEVEL = {1: 4, 2: 4, 3: 4}

# All levels share the same bug-injection instruction.
# Structural transformations (obfuscation, spaghettification) are applied as
# separate post-passes in sabotage() based on the difficulty level.

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: TEST GENERATION (happens BEFORE bug injection)
# ═══════════════════════════════════════════════════════════════════════════

_TEST_GENERATION_SYSTEM_PROMPT = """
You are an expert test creator for Python functions.

Your job: Generate 6-10 diverse test cases for a given function.

CRITICAL RULES FOR TEST ARGUMENTS:
1. ONLY positional arguments - NO keyword arguments, NO **kwargs
2. NO lambda functions, NO range(), NO generators, NO iterators
3. Use only primitives: int, float, str, list, dict, bool, None
4. Each "args" must be a STRING containing a valid Python tuple literal
5. For functions expecting iterables: use LISTS, not range() or generators

GOOD examples (args as STRINGS):
- "(5,)"
- "([1,2,3], 2)"
- "('hello', 'world')"
- "({'a': 1}, 'a')"
- "([], 0)"
- "([10, 20, 30, 40],)"  ← Use LIST for iterable, NOT range()

BAD examples (NEVER DO THIS):
- "([1,2,3], fill=0)"  ❌ NO kwargs!
- "(lambda x: x,)"  ❌ NO lambdas!
- "(range(10),)"  ❌ NO range! Use "([0,1,2,3,4,5,6,7,8,9],)" instead
- "(iter([1,2,3]),)"  ❌ NO iter()! Use "([1,2,3],)" directly

Return ONLY valid JSON (no markdown):
{
  "test_cases": [
    {"args": "(5,)"},
    {"args": "([1,2,3], 2)"},
    ...
  ]
}

CRITICAL: The "args" value must be a STRING, not actual Python syntax!
Use double quotes around the tuple literal string.

Focus on:
- Normal cases
- Edge cases (empty, None, zero, negative)
- Boundary values
- Different data types/sizes
"""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: BUG INJECTION (happens AFTER tests are generated)
# ═══════════════════════════════════════════════════════════════════════════

_BUG_INJECTION_SYSTEM_PROMPT = """
You are a bug injection specialist.

You will receive:
1. A Python function (ORIGINAL, working correctly)
2. Test cases with their CURRENT outputs on the original function

Your task: Inject ONE subtle bug so that AT LEAST HALF of the tests will produce DIFFERENT outputs.

CRITICAL REQUIREMENT:
- You MUST mentally trace through EACH test case on both the original and buggy versions
- Verify that at least 50% of tests will show DIFFERENT outputs
- If a test shows the SAME output on both versions, the bug is TOO SUBTLE or in the WRONG place

BEST BUG TYPES (choose ONE that WILL affect the test outputs you see):
1. Off-by-one in loop: range(len(lst)) → range(len(lst)-1)
2. Wrong operator: < → <=, + → -, and → or  
3. Wrong variable: total += value → total += old_value
4. Wrong initialization: count = 0 → count = 1
5. Wrong boundary: if x > 0 → if x >= 0

EXAMPLE - GOOD BUG SELECTION:
Test 1: input [1,2,3,4], output [2,3,4,5]  
Test 2: input [10,20,30], output [20,30,40]
→ Change `x + 1` to `x + 2` 
→ Test 1 will now output [3,4,5,6] ✓ DIFFERENT
→ Test 2 will now output [30,40,50] ✓ DIFFERENT

CRITICAL RULES:
- Change EXACTLY ONE thing (one operator, one number, one variable name)
- Bug must cause WRONG OUTPUT (not crashes!)
- Do NOT add comments
- Do NOT rename function or parameters
- Function must still RUN without exceptions
- At least HALF of the provided tests must produce DIFFERENT outputs

Return ONLY valid JSON (no markdown):
{
  "sabotaged_function_code": "<complete function def block>",
  "bug_description": "<one sentence: what you changed and why it fails>"
}
"""

_LEVEL_INSTRUCTIONS = {1: _BUG_INJECTION_SYSTEM_PROMPT,
                       2: _BUG_INJECTION_SYSTEM_PROMPT,
                       3: _BUG_INJECTION_SYSTEM_PROMPT}

_SYSTEM_PROMPT = """
You are the Legacy Challenge Architect. You will receive either ONE or TWO Python functions.

SCENARIO A — labeled "FUNCTION TO SABOTAGE":
  Sabotage that function directly. test_args are for calling that function.

SCENARIO B — labeled "SURFACE FUNCTION" and "HELPER FUNCTION TO SABOTAGE":
  Sabotage ONLY the HELPER FUNCTION (inject the bug there).
  - sabotaged_function_code must be the sabotaged HELPER FUNCTION def block only.
  - test_args must be valid arguments for calling the SURFACE FUNCTION (not the helper).
  - expected_output and actual_output are the results of calling the SURFACE FUNCTION with those args
    (correct vs. buggy). Mentally trace how the helper's bug propagates up to the surface result.

Reply with ONLY a valid JSON object (no markdown, no explanation), matching this exact schema:

{
  "sabotaged_function_code": "<the complete sabotaged function as a Python string — only the function def block>",
  "test_cases_public": [
    {"args": "<Python literal tuple of args, e.g. (10, 3) or ('hello',)>", "correct_output": "<correct return value for the ORIGINAL function>"},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."}
  ],
  "test_cases_secret": [
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."},
    {"args": "...", "correct_output": "..."}
  ],
  "bug_description": "<one sentence describing the FUNCTIONAL behavior that is now broken and why>"
}

CRITICAL TEST CASE REQUIREMENTS:

You must create EXACTLY 3 PUBLIC tests and EXACTLY 3 SECRET tests.
Each test must be specifically designed to expose YOUR specific bug!

PUBLIC TESTS (test_cases_public) - EXACTLY 3 TESTS:
- Test #1: Normal case that should FAIL because of your bug
- Test #2: Edge case that exposes your bug clearly
- Test #3: Another case that shows the bug pattern
- At least 2 out of 3 MUST fail on the buggy code
- These give the student hints about what's wrong
- Focus on obvious manifestations of your specific bug

SECRET TESTS (test_cases_secret) - EXACTLY 3 TESTS:
- Test #1: Boundary/edge case for your specific bug
- Test #2: Corner case that only fails if bug is NOT fixed
- Test #3: Tricky input that reveals incomplete fixes
- At least 2 out of 3 MUST fail on the buggy code
- These verify complete fix, not just partial workaround
- Examples: empty list, None, zero, negative, very large values

MANDATORY REQUIREMENT:
- If you injected "range(n-1)" bug → test with lists of different lengths
- If you injected "<=" bug → test with values exactly at the boundary
- If you injected wrong variable → test where that variable matters
- Your tests MUST actually catch YOUR SPECIFIC bug!
- DO NOT create generic tests that pass despite the bug!

TEST VALIDATION:
- At least 4 out of 6 total tests MUST fail on buggy code
- If bug is in loop condition → tests must trigger loop edge cases
- If bug is in comparison → tests must hit the comparison boundary
- NO TEST should crash with exceptions - only wrong output values

Critical rules:
- sabotaged_function_code must be ONLY the function definition (def ...: ...) — nothing before or after.
- The `def` line MUST keep the EXACT SAME function name as the original. Do NOT rename the function.
- Preserve the EXACT original indentation level of the function (copy it from the input).
- Do NOT modify any docstrings or string literals that already exist in the function.
- CRITICAL: Do NOT add ANY comments to the code! No # comments, no explanatory notes, nothing!
  Follow ONLY the steps listed in the SABOTAGE INSTRUCTIONS — do not add renaming or comments
  unless the instructions explicitly ask for them (they never will).
- The bug MUST produce a WRONG RETURN VALUE — NOT a crash or exception.
  The sabotaged function must still run without raising exceptions; it just returns the wrong result.
  If your bug causes a TypeError, AttributeError, or any other exception, it will be REJECTED.
  Change a condition, a boundary, an operator, or a value — not the structure so drastically it crashes.
- Each "args" must be a Python tuple literal using ONLY these primitives: int, float, str, list, dict, bool.
  NEVER use: lambda, range, callable, custom classes, or any non-serializable object.
  NEVER use keyword arguments — only positional values inside the tuple.
  Wrong: ('hello', sep=',') or (x=1, y=2)  |  Right: ('hello', ',') or (1, 2)
  Each "correct_output" is the return value of the ORIGINAL (unfixed) function — not the buggy one.
  Example args: (10, 3) or (['a','b','c'],) or ("hello world",) or ([1, 2, 3],) — nothing else.
- Do NOT include any text outside the JSON object.

====================
COMPLETE EXAMPLE - STUDY THIS BEFORE YOU START!
====================

Original function:
```python
def get_last_n(lst, n):
    \"\"\"Return the last n elements of list.\"\"\"
    if n <= 0:
        return []
    return lst[-n:]
```

BUGGY version (off-by-one in slice):
```python
def get_last_n(lst, n):
    \"\"\"Return the last n elements of list.\"\"\"
    if n <= 0:
        return []
    return lst[-(n-1):]  # BUG: slice index off by one!
```

Why this is a GOOD bug:
- Simple one-operator change: -n becomes -(n-1)
- Returns wrong value (too many elements)
- Doesn't crash
- Subtle - looks like it could be correct

====================
CRITICAL: HOW TO CREATE TESTS THAT CATCH YOUR BUG
====================

STEP 1: Create the bug FIRST
STEP 2: Mentally execute EACH test on BOTH versions (original and buggy)
STEP 3: Verify that at least 4 tests have DIFFERENT outputs between the two versions

MENTAL EXECUTION EXAMPLE:

Test: ([1,2,3,4,5], 2)
  Original code: lst[-2:] = [4,5]  OK
  Buggy code: lst[-(2-1):] = lst[-1:] = [5]  X DIFFERENT!
  
Test: ([10,20,30], 1) 
  Original: lst[-1:] = [30]  OK
  Buggy: lst[-(1-1):] = lst[0:] = [10,20,30]  X DIFFERENT!

Test: (['a','b','c','d'], 3)
  Original: lst[-3:] = ['b','c','d']  OK
  Buggy: lst[-2:] = ['c','d']  X DIFFERENT!

Result: 3/3 tests show different outputs = GOOD! The tests CATCH the bug.

IF YOUR TESTS DO NOT SHOW DIFFERENT OUTPUTS, THEY ARE WRONG!

Tests that EXPOSE this bug:
```json
{
  "test_cases_public": [
    {"args": "([1,2,3,4,5], 2)", "correct_output": "[4,5]"},
    {"args": "([10,20,30], 1)", "correct_output": "[30]"},
    {"args": "(['a','b','c','d'], 3)", "correct_output": "['b','c','d']"}
  ],
  "test_cases_secret": [
    {"args": "([1], 1)", "correct_output": "[1]"},
    {"args": "([5,6,7,8,9], 4)", "correct_output": "[6,7,8,9]"},
    {"args": "([], 2)", "correct_output": "[]"}
  ]
}
```

Why these tests WORK:
- Test ([1,2,3,4,5], 2): Original returns [4,5], Buggy returns [3,4,5] - FAILS OK
- Test ([10,20,30], 1): Original returns [30], Buggy returns [20,30] - FAILS OK
- Test (['a','b','c','d'], 3): Original returns ['b','c','d'], Buggy returns ['a','b','c','d'] - FAILS OK
- Test ([1], 1): Original returns [1], Buggy returns [1,1] or crashes - edge case OK
- Test ([5,6,7,8,9], 4): Original returns [6,7,8,9], Buggy returns [5,6,7,8,9] - FAILS OK

Result: 4+ out of 6 tests fail on buggy code!

====================
NOW DO THE SAME FOR YOUR FUNCTION!
====================
"""


def _extract_function_source(source: str, func_name: str) -> tuple[str, int, int]:
    """Return (func_source, start_line_0indexed, end_line_exclusive_0indexed)."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            start = node.lineno - 1          # 0-indexed
            end = node.end_lineno            # 0-indexed exclusive
            func_source = "".join(lines[start:end])
            return func_source, start, end
    raise ValueError(f"Function '{func_name}' not found in source.")


def _pick_best_function(source: str, exclude: set[str] | None = None) -> str:
    """Pick randomly from the top-3 MODULE-LEVEL functions that work with primitive types.
    exclude: function names already tried — deprioritised but used as last resort."""
    exclude = exclude or set()
    tree = ast.parse(source)
    scored: list[tuple[int, str]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") or node.name.startswith("_"):
            continue
        if node.name.startswith("test"):
            continue
        if node.end_lineno - node.lineno < 5:
            continue

        score = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, (int, float, str)):
                score += 2
            if isinstance(child, (ast.For, ast.While)):
                score += 3
            if isinstance(child, ast.If):
                score += 2
            if isinstance(child, ast.Return):
                score += 2
            if isinstance(child, ast.BinOp):
                score += 2
            if isinstance(child, ast.ListComp):
                score += 3
            if isinstance(child, ast.Compare):
                score += 1
            if isinstance(child, ast.Attribute):
                score -= 1  # obj.method() suggests class usage

        if score > 0:
            scored.append((score, node.name))

    if not scored:
        # Fallback: accept any public module-level function longer than 3 lines
        fallback_funcs = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
            and not node.name.startswith("test")
            and node.end_lineno - node.lineno >= 3
        ]
        if not fallback_funcs:
            raise ValueError("No suitable public module-level function found in the target file.")
        fresh = [f for f in fallback_funcs if f not in exclude] or fallback_funcs
        chosen = random.choice(fresh)
        print(f"[saboteur] Function candidates (fallback): {fallback_funcs} -> chose '{chosen}'")
        return chosen

    scored.sort(key=lambda x: x[0], reverse=True)
    # Prefer functions not yet tried; fall back to all if all tried
    fresh_scored = [(s, n) for s, n in scored if n not in exclude]
    pool = fresh_scored[:3] if fresh_scored else scored[:3]
    _, chosen = random.choice(pool)
    print(f"[saboteur] Function candidates: {[n for _, n in scored[:5]]} -> chose '{chosen}'"
          + (f" (excluded: {sorted(exclude)})" if exclude else ""))
    return chosen


def _find_called_module_functions(func_node: ast.FunctionDef, module_func_names: set) -> list:
    """Return names of module-level functions directly called inside func_node."""
    called = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in module_func_names:
                called.add(node.func.id)
    return list(called)


def _build_call_graph(module_func_nodes: dict) -> dict:
    """Return {name: set_of_module_level_functions_it_calls} for every function."""
    names = set(module_func_nodes.keys())
    return {
        name: set(_find_called_module_functions(node, names - {name}))
        for name, node in module_func_nodes.items()
    }


def _find_reachable(call_graph: dict, start: str, max_depth: int) -> dict:
    """
    Find all reachable functions from start with their MAXIMUM depth.
    Returns {func_name: max_depth} for every reachable function (excluding start).
    Uses DFS to explore all paths and keep the longest depth for each function.
    Only includes functions that exist as keys in call_graph.
    """
    visited: dict[str, int] = {}
    
    def dfs(current: str, depth: int, seen: set[str]):
        if depth >= max_depth:
            return
        
        for callee in call_graph.get(current, set()):
            if callee not in call_graph:
                continue
            
            # Update with maximum depth found
            if callee not in visited or visited[callee] < depth + 1:
                visited[callee] = depth + 1
            
            # Continue exploring if we haven't been through this exact path
            if callee not in seen:
                seen.add(callee)
                dfs(callee, depth + 1, seen)
                seen.remove(callee)
    
    dfs(start, 0, {start})
    return visited


def _find_call_path(call_graph: dict, start: str, target: str, max_depth: int) -> list[str]:
    """
    Find the LONGEST call path from start to target (up to max_depth).
    This ensures we get chains matching the requested nesting level.
    Uses DFS to explore all paths and returns the longest one found.
    """
    if start == target:
        return [start]
    
    longest_path: list[str] = []
    
    def dfs(current: str, path: list[str], visited: set[str]):
        nonlocal longest_path
        
        if len(path) > max_depth + 1:
            return
        
        if current == target:
            if len(path) > len(longest_path):
                longest_path = path.copy()
            return
        
        for callee in call_graph.get(current, set()):
            if callee not in visited:
                visited.add(callee)
                path.append(callee)
                dfs(callee, path, visited)
                path.pop()
                visited.remove(callee)
    
    dfs(start, [start], {start})
    
    # If no path found, return direct connection as fallback
    return longest_path if longest_path else [start, target]


def _pick_surface_function(source: str, max_depth: int,
                           exclude: set[str] | None = None) -> tuple[str, dict]:
    """
    Pick the surface function (entry point reported as broken to the student).
    Prefers functions that have the deepest reachable call chain, so bugs can
    be hidden many hops away from where the student starts investigating.
    exclude: set of function names already tried — skip them first, use them as last resort.
    Returns (surface_func_name, module_func_nodes).
    """
    exclude = exclude or set()
    tree = ast.parse(source)
    module_func_nodes = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
        and not node.name.startswith("test")
        and node.end_lineno - node.lineno >= 5
    }

    call_graph = _build_call_graph(module_func_nodes)

    # Score each function by the depth and count of its reachable targets
    # Heavily prefer functions that can reach targets at depths close to max_depth
    caller_candidates: list[tuple[int, str]] = []
    for name in module_func_nodes:
        reachable = _find_reachable(call_graph, name, max_depth)
        substantial = {
            h: d for h, d in reachable.items()
            if module_func_nodes[h].end_lineno - module_func_nodes[h].lineno >= 5
        }
        if not substantial:
            continue
        
        # Compute score with strong preference for deep chains
        score = 0
        max_reachable_depth = max(substantial.values()) if substantial else 0
        
        # Bonus for having functions at good depths (prefer depth >= max_depth - 1)
        for d in substantial.values():
            if d >= max_depth - 1:
                score += d * 100  # Very strong bonus for deep chains
            elif d >= max_depth - 2:
                score += d * 50   # Good bonus for moderately deep chains
            else:
                score += d * 10   # Small bonus for shallow chains
        
        # Additional bonus for maximum depth reached
        if max_reachable_depth >= max_depth - 1:
            score += 200
        
        # Function size contributes minimally
        score += module_func_nodes[name].end_lineno - module_func_nodes[name].lineno
        
        caller_candidates.append((score, name))

    if caller_candidates:
        # Prefer candidates not yet tried; fall back to all if all were tried
        fresh = [(s, n) for s, n in caller_candidates if n not in exclude]
        pool = fresh if fresh else caller_candidates
        # Randomly sample from the pool weighted by score (higher score = more likely)
        total = sum(s for s, _ in pool)
        r = random.uniform(0, total)
        cumulative = 0
        chosen = pool[-1][1]
        for score, name in pool:
            cumulative += score
            if r <= cumulative:
                chosen = name
                break
        all_names = [n for _, n in caller_candidates]
        print(f"[saboteur] Surface candidates (deep chains): {all_names} -> '{chosen}'"
              + (f" (excluded: {sorted(exclude)})" if exclude else ""))
        return chosen, module_func_nodes

    # Fallback: no function calls helpers — pick the most complex function directly
    print("[saboteur] No caller-surface found — falling back to direct mode.")
    all_funcs = [n for n in module_func_nodes if n not in exclude] or list(module_func_nodes)
    return _pick_best_function(source, exclude=exclude), module_func_nodes


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _try_exec(full_source: str, func_name: str, test_args_str: str,
              file_path: str | None = None) -> tuple[bool, str]:
    """
    Execute the full file source in an isolated namespace, call func_name(*args),
    and return (success, repr_of_result).  Falls back to (False, "") on any error.
    file_path: absolute path of the source file — its directory is added to sys.path
               so that sibling-module imports (e.g. 'import file2') can resolve.
    """
    import sys as _sys
    added_path: str | None = None
    if file_path:
        dir_to_add = os.path.dirname(os.path.abspath(file_path))
        if dir_to_add not in _sys.path:
            _sys.path.insert(0, dir_to_add)
            added_path = dir_to_add
    namespace: dict = {"__name__": "__main__", "__file__": file_path or "<exec>"}
    try:
        exec(compile(full_source, file_path or "<exec>", "exec"), namespace)  # runs all imports + defs
        func = namespace.get(func_name)
        if func is None:
            return False, ""
        args = eval(test_args_str, {"__builtins__": __builtins__})
        if not isinstance(args, tuple):
            args = (args,)
        result = func(*args)
        return True, repr(result)
    except Exception as e:
        return False, f"ERROR:{type(e).__name__}: {e}"
    finally:
        if added_path and added_path in _sys.path:
            _sys.path.remove(added_path)


# Only words that unambiguously signal "this is a bug" — common English words excluded
_REVEAL_WORDS = {
    "incorrect", "incorrectly", "wrong", "wrongly", "bug", "buggy",
    "sabotage", "injected", "intentional", "deliberately", "purposely",
    "off-by-one", "off by one",
}


def _has_revealing_comment(original_func: str, sabotaged_func: str) -> bool:
    """Return True if GPT added a comment that gives away the bug."""
    orig_stripped = {line.strip() for line in original_func.splitlines()}
    for line in sabotaged_func.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and stripped not in orig_stripped:
            lower = stripped.lower()
            if any(word in lower for word in _REVEAL_WORDS):
                return True
    return False


def _variables_were_renamed(original_func: str, sabotaged_func: str) -> bool:
    """Return True if at least one parameter or local variable was renamed."""
    def all_user_names(src: str) -> set:
        try:
            tree = ast.parse(textwrap.dedent(src))
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            # local variables
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            # parameters
            if isinstance(node, ast.arguments):
                for arg in node.args + node.posonlyargs + node.kwonlyargs:
                    names.add(arg.arg)
                if node.vararg:
                    names.add(node.vararg.arg)
                if node.kwarg:
                    names.add(node.kwarg.arg)
        return names

    orig = all_user_names(original_func)
    if not orig:
        return True  # nothing to rename; skip the check
    sabot = all_user_names(sabotaged_func)
    return bool(orig - sabot)


def _generate_tests_for_function(
    func_source: str,
    func_name: str,
    surface_func_name: str,
    surface_source: str,
    llm,
    indirect_mode: bool,
    debug_mode: bool = False,
) -> dict | None:
    """
    Generate 6-10 test cases for a function WITHOUT injecting any bugs.
    Tests are created for the ORIGINAL, working function.
    Returns dict with test_cases list, or None on failure.
    """
    
    if indirect_mode:
        user_content = (
            f"SURFACE FUNCTION (this is what the student will call):\n"
            f"```python\n{surface_source}\n```\n\n"
            f"HELPER FUNCTION (you're testing the SURFACE via this helper):\n"
            f"```python\n{func_source}\n```\n\n"
            f"Generate test args for calling {surface_func_name}."
        )
    else:
        user_content = (
            f"FUNCTION TO TEST:\n```python\n{func_source}\n```"
        )
    
    for attempt in range(1, 3):
        if debug_mode:
            print(f"[test_gen] Generating 6-10 tests for '{func_name}' (attempt {attempt})...")
        
        try:
            response = llm.invoke([
                SystemMessage(content=_TEST_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            
            if debug_mode:
                print(f"[test_gen] Raw GPT response preview: {response.content[:200]}...")
            
            data = _parse_response(response.content)
            
            test_cases = data.get("test_cases", [])
            
            if len(test_cases) >= 6:
                data["test_cases"] = test_cases[:10]  # Keep at most 10
                if debug_mode:
                    print(f"[test_gen] OK: Generated {len(data['test_cases'])} tests")
                return data
            
            if debug_mode:
                print(f"[test_gen] Not enough tests: got {len(test_cases)}, need 6+")
        
        except json.JSONDecodeError as e:
            if debug_mode:
                print(f"[test_gen] JSON parse error: {e}")
                print(f"[test_gen] Full response: {response.content}")
            continue
        except Exception as e:
            if debug_mode:
                print(f"[test_gen] Error: {e}")
            continue
    
    return None


def _inject_bug_into_function(
    func_source: str,
    func_name: str,
    llm,
    attempted_bugs: list[str] | None = None,
    test_cases: list[dict] | None = None,
    original_results: dict | None = None,
    debug_mode: bool = False,
) -> dict | None:
    """
    Inject ONE simple bug into a function.
    GPT sees test cases and their expected outputs to create a bug that WILL be detectable.
    Returns dict with sabotaged_function_code and bug_description, or None on failure.
    """
    forbidden_bugs = ""
    if attempted_bugs:
        listed = "\n".join(f"  - {b}" for b in attempted_bugs)
        forbidden_bugs = f"\n\nFORBIDDEN (already tried):\n{listed}\nChoose a DIFFERENT bug type!"
    
    # Build test cases section with current outputs from ORIGINAL function
    test_info = ""
    if test_cases and original_results:
        test_info = "\n\nTEST CASES (all currently PASSING on the ORIGINAL function):\n"
        for i, test in enumerate(test_cases, 1):
            args = test.get("args", "") or test.get("test_args", "")
            if args in original_results:
                success, result = original_results[args]
                if success:
                    test_info += f"{i}. Input: {args}\n   Current Output (before bug): {result}\n\n"
        test_info += "\nYour bug must cause AT LEAST HALF of these tests to produce DIFFERENT outputs.\n"
    
    user_content = f"""
FUNCTION TO INJECT BUG INTO:

```python
{func_source}
```
{test_info}{forbidden_bugs}
"""
    
    for attempt in range(1, 3):
        if debug_mode:
            print(f"[bug_inject] Injecting bug into '{func_name}' (attempt {attempt})...")
        
        try:
            response = llm.invoke([
                SystemMessage(content=_BUG_INJECTION_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            
            content = response.content.strip()
            # Remove markdown if present
            if content.startswith("```"):
                parts = content.split("```")
                content = parts[1] if len(parts) > 1 else content
                if content.startswith("json"):
                    content = content[4:].strip()
            
            data = json.loads(content)
            buggy_code = data.get("sabotaged_function_code", "")
            
            # Validate syntax
            try:
                ast.parse(buggy_code)
                if debug_mode:
                    print(f"[bug_inject] OK: {data.get('bug_description', 'N/A')}")
                return data
            except SyntaxError:
                if debug_mode:
                    print(f"[bug_inject] Syntax error in buggy code")
                continue
        
        except Exception as e:
            if debug_mode:
                print(f"[bug_inject] Error: {e}")
            continue
    
    return None



def _format_bug_diff(
    func_name: str,
    file_start_line: int,
    file_end_line: int,
    original_source: str,
    sabotaged_source: str,
) -> str:
    """
    Return a compact architect-only diff showing exactly which file lines changed.
    file_start_line is 0-indexed; output uses 1-indexed line numbers.
    """
    orig_lines  = original_source.splitlines()
    sabot_lines = sabotaged_source.splitlines()

    header = (
        f"\n  ┌─ BUG INJECTED INTO: {func_name} "
        f"(file lines {file_start_line + 1}–{file_end_line} - BEFORE inflation)\n"
        f"  │\n"
        f"  │  NOTE: Line numbers shown are BEFORE function inflation.\n"
        f"  │        In the final file, functions will be expanded (50+ lines each).\n"
        f"  │"
    )
    diff_lines = []
    matcher = difflib.SequenceMatcher(None, orig_lines, sabot_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for k, line in enumerate(orig_lines[i1:i2]):
            file_no = file_start_line + i1 + k + 1  # 1-indexed
            diff_lines.append(f"  │  -{file_no:4d}: {line.rstrip()}")
        for line in sabot_lines[j1:j2]:
            diff_lines.append(f"  │  +     : {line.rstrip()}")

    if not diff_lines:
        diff_lines.append("  │  (no line-level changes detected)")

    return header + "\n".join([""] + diff_lines) + "\n  └─"


def _execute_tests_on_source(
    source: str,
    func_name: str, 
    test_data: dict,
    file_path: str | None = None,
    debug_mode: bool = False
) -> dict:
    """
    Execute all tests on the given source code.
    Returns dict mapping test_args_str -> (success, result_str)
    """
    results = {}
    all_tests = test_data.get("test_cases", [])
    
    for test in all_tests:
        test_args_str = test.get("args", "") or test.get("test_args", "")
        if not test_args_str:
            continue
        success, result = _try_exec(source, func_name, test_args_str, file_path)
        results[test_args_str] = (success, result)
        
        if debug_mode:
            status = "OK" if success else "FAIL"
            print(f"[saboteur]   Test {test_args_str} -> {status}: {result[:50]}")
    
    return results


def _sabotage_one_helper(
    bug_func_name: str,
    current_source: str,
    surface_func_name: str,
    surface_source: str,
    instructions: str,
    llm,
    indirect_mode: bool,
    call_chain: list[str] | None = None,
    previous_bugs: list[str] | None = None,
    target_nesting: int = 3,
    file_path: str | None = None,
    debug_mode: bool = False,
) -> tuple[str, dict] | tuple[None, None]:
    """
    TWO-PHASE APPROACH:
    1. Generate 6-10 tests on the WORKING code
    2. Inject bug into function (GPT sees tests and must fail at least half)
    3. Validate that bug is detectable by at least 1 test
    
    Returns (buggy_source, data_dict) or (None, None) on failure.
    """
    func_source, start_line, end_line = _extract_function_source(current_source, bug_func_name)

    # =====================================================================
    # PHASE 1: GENERATE 6-10 TESTS ON WORKING CODE
    # =====================================================================
    if debug_mode:
        print(f"[saboteur] PHASE 1: Generating 6-10 tests for '{bug_func_name}'...")
    
    test_data = _generate_tests_for_function(
        func_source=func_source,
        func_name=bug_func_name,
        surface_func_name=surface_func_name,
        surface_source=surface_source,
        llm=llm,
        indirect_mode=indirect_mode,
        debug_mode=debug_mode
    )
    
    if test_data is None:
        if debug_mode:
            print(f"[saboteur] PHASE 1 FAILED: Could not generate tests")
        return None, None
    
    test_cases = test_data.get("test_cases", [])
    if len(test_cases) < 6:
        if debug_mode:
            print(f"[saboteur] PHASE 1 FAILED: Need at least 6 tests, got {len(test_cases)}")
        return None, None
    
    if debug_mode:
        print(f"[saboteur] PHASE 1 SUCCESS: Generated {len(test_cases)} tests")
    
    # Execute tests on ORIGINAL code to capture correct outputs
    if debug_mode:
        print(f"[saboteur] Executing tests on ORIGINAL code...")
    original_results = _execute_tests_on_source(
        current_source, 
        surface_func_name, 
        test_data,
        file_path=file_path,
        debug_mode=debug_mode
    )
    
    # =====================================================================
    # PHASE 2: INJECT BUG
    # =====================================================================
    attempted_bugs: list[str] = []
    
    for attempt in range(1, 4):
        if debug_mode:
            print(f"[saboteur] PHASE 2: Bug injection attempt {attempt}/3...")
        
        # Inject bug - GPT sees tests to make bug detectable
        buggy_func_dict = _inject_bug_into_function(
            func_source=func_source,
            func_name=bug_func_name,
            llm=llm,
            attempted_bugs=attempted_bugs,
            test_cases=test_cases,
            original_results=original_results,
            debug_mode=debug_mode
        )
        
        if buggy_func_dict is None:
            if debug_mode:
                print(f"[saboteur] Bug injection failed on attempt {attempt}")
            continue
        
        sabotaged_func = buggy_func_dict["sabotaged_function_code"]
        bug_description = buggy_func_dict.get("bug_description", "")
        
        if bug_description:
            attempted_bugs.append(bug_description)
        
        # Splice buggy function back into source
        lines = current_source.splitlines(keepends=True)
        candidate_source = "".join(lines[:start_line]) + sabotaged_func + "\n" + "".join(lines[end_line:])
        
        # Validate syntax
        try:
            new_tree = ast.parse(candidate_source)
        except SyntaxError as e:
            if debug_mode:
                print(f"[saboteur] Syntax error in buggy code: {e}")
            continue
        
        # Verify function name preserved
        new_names = {
            n.name for n in new_tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if bug_func_name not in new_names:
            if debug_mode:
                print(f"[saboteur] Function '{bug_func_name}' was renamed — retrying")
            continue
        
        # Reject if comments reveal the bug
        if _has_revealing_comment(func_source, sabotaged_func):
            if debug_mode:
                print(f"[saboteur] Comment reveals bug — retrying")
            continue
        
        # =====================================================================
        # PHASE 3: VALIDATE BUG IS DETECTABLE
        # =====================================================================
        if debug_mode:
            print(f"[saboteur] PHASE 3: Testing if bug is detectable...")
        
        buggy_results = _execute_tests_on_source(
            candidate_source,
            surface_func_name,
            test_data,
            file_path=file_path,
            debug_mode=debug_mode
        )
        
        # Count tests that detect the bug (different result)
        detecting_tests = []
        
        for test_args_str in original_results:
            if test_args_str not in buggy_results:
                continue
            
            orig_success, orig_result = original_results[test_args_str]
            buggy_success, buggy_result = buggy_results[test_args_str]
            
            # Bug detected if: success status changed OR result value changed
            if orig_success != buggy_success or orig_result != buggy_result:
                detecting_tests.append({
                    "args": test_args_str, 
                    "expected": orig_result
                })
                
                if debug_mode:
                    print(f"[saboteur]   Test DETECTS bug: {test_args_str}")
        
        total_tests = len(test_cases)
        detecting_count = len(detecting_tests)
        
        if debug_mode:
            print(f"[saboteur] Bug detected by {detecting_count}/{total_tests} tests")
        
        # SIMPLE VALIDATION: At least SOME tests must detect the bug
        if detecting_count >= 1:
            if debug_mode:
                print(f"[saboteur] SUCCESS! Bug is detectable and we have {total_tests} tests total")
            
            # Return all tests - GPT and student interface will handle them
            result_data = {
                "test_cases": test_cases,  # All tests (detecting + passing)
                "detecting_tests": detecting_tests,  # Which tests detect the bug
                "bug_description": bug_description,
                "_debug_func_name": bug_func_name,
                "_debug_start_line": start_line,
                "_debug_end_line": end_line,
                "_debug_func_source": func_source,  # Original function
                "_debug_sabot_func": sabotaged_func,  # Buggy function
                "_debug_tests_summary": f"{detecting_count}/{total_tests} tests detect bug"
            }
            
            return candidate_source, result_data
        else:
            if debug_mode:
                print(f"[saboteur] REJECT: Bug not detected by any tests")
    
    if debug_mode:
        print(f"[saboteur] All 3 attempts exhausted — no suitable bug found")
    return None, None


def _strip_markdown_code(text: str) -> str:
    """Strip ```python ... ``` or ``` ... ``` fences from a GPT response."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("python"):
            text = text[6:]
        text = text.strip()
    return text


def _chain_for_obfuscation(source: str, surface_func: str, max_depth: int = 4) -> set:
    """BFS call-chain discovery from surface_func through all module-level functions."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {surface_func}
    defined = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    chain: set = set()
    frontier = {surface_func}
    for _ in range(max_depth):
        next_f: set = set()
        for fname in frontier:
            if fname in chain or fname not in defined:
                continue
            chain.add(fname)
            for n in ast.walk(defined[fname]):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    if n.func.id in defined and n.func.id not in chain:
                        next_f.add(n.func.id)
        frontier = next_f
        if not frontier:
            break
    return chain


def _extract_chain_snippet(source: str, chain: set) -> tuple:
    """
    Build a mini source: header imports + only the chain functions.
    Returns (mini_source, last_end_lineno_in_original).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, len(source.splitlines())
    lines = source.splitlines(keepends=True)
    header: list = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)) and node.lineno <= 80:
            header.extend(lines[node.lineno - 1:node.end_lineno])
    blocks: list = []
    last_end = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in chain:
            blocks.append("".join(lines[node.lineno - 1:node.end_lineno]))
            last_end = max(last_end, node.end_lineno)
    mini = "".join(header) + "\n\n" + "\n\n".join(blocks)
    return mini, last_end


def _splice_transforms_back(original: str, transformed_snippet: str, orig_chain: set) -> str:
    """
    Splice transformed functions back into the original file.
    - Functions in orig_chain (same name) -> replace in-place
    - New functions (wrappers/ghosts) -> insert after the last replaced block
    Returns merged source, or original on any parse error.
    """
    try:
        orig_tree = ast.parse(original)
        new_tree  = ast.parse(transformed_snippet)
    except SyntaxError:
        return original

    orig_lines = original.splitlines(keepends=True)
    new_lines  = transformed_snippet.splitlines(keepends=True)

    new_funcs: dict = {}
    for node in new_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new_funcs[node.name] = "".join(new_lines[node.lineno - 1:node.end_lineno])

    orig_pos: dict = {}
    for node in orig_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            orig_pos[node.name] = (node.lineno - 1, node.end_lineno)

    replacements: list = []
    insertions:   list = []
    last_replace_end = -1

    for name, src in new_funcs.items():
        if name in orig_pos:
            start, end = orig_pos[name]
            replacements.append((start, end, src))
            last_replace_end = max(last_replace_end, end)
        else:
            insertions.append(src)

    if not replacements:
        return original

    replacements.sort(key=lambda x: x[0], reverse=True)
    result = list(orig_lines)
    size_delta = 0
    for start, end, new_src in replacements:
        new_src_lines = (new_src.rstrip("\n") + "\n").splitlines(keepends=True)
        old_size = end - start
        result[start:end] = new_src_lines
        size_delta += len(new_src_lines) - old_size

    if insertions and last_replace_end >= 0:
        adjusted = last_replace_end + size_delta
        insert_block = "\n\n" + "\n\n".join(s.rstrip("\n") for s in insertions) + "\n\n"
        result.insert(adjusted, insert_block)

    return "".join(result)


# Horizontal & Vertical Expansion rules applied to EVERY function created or modified.
_EXPANSION_RULES = """
MANDATORY EXPANSION RULES — apply to EVERY function you write or modify:

  !! ZERO TOLERANCE ANTI-PATTERN — NEVER produce this: !!

    def _process_segment_core(data_ref, sep=',', prefix=''):
        \"\"\"Core processing layer.\"\"\"
        return _process_segment_impl(data_ref, sep, prefix)    # <-- VIOLATION

  A function whose body is ONLY a return-delegation call (1-3 lines total) is a
  HARD FAILURE.  It will be automatically rejected.  Every wrapper you write —
  even a "thin" pass-through — MUST look like the CORRECT PATTERN below.

  !! CORRECT PATTERN — every wrapper MUST look like this (50+ lines minimum): !!

    def _process_segment_core(data_ref, sep=',', prefix='', flag=False):
        \"\"\"Core processing layer.\"\"\"
        import sys as _sys_ref
        global _state_flux
        _buffer_offset = len(str(data_ref)) * 1
        _tmp_flag = not False
        _dead_counter = 0
        _entropy_val = _buffer_offset - _buffer_offset
        _noise_a = sum(i for i in range(min(_buffer_offset, 5)))
        _noise_b = _noise_a - _noise_a
        _checksum_pre = (_buffer_offset % 7) - (_buffer_offset % 7)
        _ref_copy = data_ref
        _discarded_a = str(_ref_copy)[::-1][::-1]
        if _tmp_flag and _buffer_offset >= 0:
            _dead_list = [x * 0 for x in range(6)]
            _dead_result = sum(_dead_list)
            if not (flag is None and False):
                _checksum = _dead_result % 3
                if _checksum >= 0:                     # always evaluates true
                    _tmp_marker = bool(data_ref is not None)
                    if _tmp_marker == True:             # noqa
                        _state_flux = _dead_result
                        _entropy_b = len(_discarded_a) - len(_discarded_a)
                        # do not touch
                        if _entropy_b == 0:
                            _noise_c = [i - i for i in range(4)]
                            _ = sum(_noise_c)
                            # logic starts here
                            _sep_copy = sep if sep is not None else sep
                            _pre_copy = prefix if not (prefix is None and False) else prefix
                            _dead_counter += 0
                            try:
                                _intermediate = _process_segment_impl(
                                    _ref_copy, _sep_copy, _pre_copy, flag
                                )
                                _post_check = _intermediate is not None
                                if _post_check == True:  # noqa
                                    _dead_counter += 0
                                    _noise_d = _buffer_offset * 0
                                    return _intermediate
                            except Exception as _e_flux:
                                raise
        return _process_segment_impl(data_ref, sep, prefix, flag)

  EVERY function you write MUST follow this pattern: dense busy-work BEFORE and
  AFTER the real call, nested conditionals that always evaluate true, dead variables,
  global declarations, and the actual delegation buried in the middle.

  EXPANSION 1 — EXTREME FUNCTION BLOATING (MINIMUM 50 LINES, NO EXCEPTIONS):
    Every single function — wrapper, helper, ghost, or original — MUST be at least
    50 lines long. If you are about to write a function shorter than 50 lines, STOP
    and add more padding until it reaches 50+.
    Required padding techniques (use ALL of them):
    - At least 8-10 dummy local variable assignments at the top: _buf, _flag, _counter, _entropy
    - At least 2 loop-based dummy computations: `_n = [x*0 for x in range(5)]`
    - At least 3 levels of always-true nested if/else wrapping the real call
    - `global _state_flux` declaration inside the function
    - `import <module>` statement inside the function body
    - At least 3 trash comments: `# do not touch`, `# logic starts here`, `# ??`
    - A try/except block wrapping the actual delegation call

  EXPANSION 2 — CALL-STACK OVERLOAD:
    Inside each function, inject multiple calls to other newly created helper functions.
    Build a web of dependencies: funcA calls funcB, funcC, funcD; each of those calls
    3 more. The call graph must be deep and wide so tracing execution is exhausting.
    Every helper in the web must also follow EXPANSION 1 (bloated to 50+ lines).

  EXPANSION 3 — "WHERE IS THE LOGIC?" CHALLENGE:
    Surround every real/meaningful call with 5-10 fake "busy-work" calls whose names
    suggest they are critical infrastructure but actually do nothing:
        audit_buffer_state()   verify_integrity_checksum()   sync_internal_stack()
        flush_pipeline_cache()  reindex_lookup_table()   hydrate_context_frame()
    A debugger stepping through must wade past all these calls to find the one that matters.

  EXPANSION 4 — DEEP NESTING & INDENTATION:
    Push the actual core logic 5-7 levels deep using nested if/else and try/except blocks.
    Each nesting level should have a plausible-looking (but ultimately irrelevant) condition.
    The real work must be buried at the deepest indentation level, far to the right of screen.

  EXPANSION 5 — BUG BURIAL:
    The injected bug (or the core logic that contains it) MUST NOT appear at the top or
    bottom of any long function. Bury it around line 40-60 of the function body, inside
    a deeply nested block, preceded and followed by dense "busy-work" padding code.
    This ensures the bug is invisible at first glance and exhausts token budgets.
"""


def _obfuscate_full_file(source: str, surface_func_name: str, llm, level: int = 1,
                         verified_cases: list | None = None, file_path: str | None = None,
                         bug_func_name: str | None = None, buggy_func_source: str | None = None) -> str:
    """
    Level 1 post-pass: legacy obfuscation — cryptic names, deprecated patterns, global vars,
    mixed styling, removed comments.  When level==3, also injects red-herring decoys.
    Only the file where the bug was injected is transformed; no other files are touched.
    After transformation, verifies that surface_func_name still exists and the bug still
    manifests on at least one verified_case. Falls back to original source on any failure.
    """
    # Extract only the chain functions — avoids sending the whole file to GPT
    chain = _chain_for_obfuscation(source, surface_func_name)
    mini_source, _ = _extract_chain_snippet(source, chain)

    red_herring_rule = ""
    if level == 3:
        red_herring_rule = (
            "  RULE 8 \u2014 RED HERRINGS (Level 3 only):\n"
            "    Inject 2-3 convincing decoy bugs that look real but have NO effect on output.\n"
            "    Examples:\n"
            "    - A dead if-branch (condition always False) with suspicious-looking code\n"
            "    - A local variable overwritten immediately before use (looks like wrong value)\n"
            "    - A commented-out line hinting at a past fix: `# old fix: x -= 1`\n"
            "    - Suspicious math `result = result * 1.0000001` that looks wrong but is harmless\n"
            "    These MUST NOT change the observable output \u2014 pure misdirection only.\n\n"
        )

    bug_preservation_rule = ""
    if bug_func_name and buggy_func_source:
        bug_preservation_rule = (
            f"  !!! CRITICAL BUG PRESERVATION — read this before doing ANYTHING !!!\n"
            f"  The function `{bug_func_name}` contains an intentional, hidden bug.\n"
            f"  When you wrap or restructure this function, you MUST place its logic\n"
            f"  EXACTLY as shown below into the innermost wrapper — not paraphrased,\n"
            f"  not rewritten, not 'cleaned up'. Copy every operator, every condition,\n"
            f"  every loop bound VERBATIM. You may rename parameters/variables for\n"
            f"  obfuscation, but the LOGICAL STRUCTURE must be byte-for-byte identical.\n"
            f"  The original buggy function body:\n\n"
            f"{buggy_func_source}\n\n"
            f"  Preserve this logic EXACTLY inside the deepest wrapper.\n\n"
        )

    prompt = (
        f"Refactor ONLY the following Python file into LEVEL 1 — Obfuscated Legacy code.\n"
        f"SCOPE: Modify ONLY this file. Do NOT touch any other file in the project.\n"
        f"All logic and any existing bugs MUST be preserved EXACTLY — same inputs, same outputs.\n\n"
        f"=== LEVEL 1 RULES — apply ALL of them aggressively ===\n\n"
        f"  RULE 1 — MANDATORY SHADOW WRAPPING (at least 4 calls deep):\n"
        f"    Every functional logic path must pass through at least 4 function calls.\n"
        f"    If the original logic is short, create 'Shadow Wrapper' functions around it that\n"
        f"    perform redundant checks or dummy transformations before delegating:\n"
        f"      e.g.  x = (x + 0) * 1   or   val = val if val is not None else val\n"
        f"    Structure every function so its real work is 4 calls deep from the entry point.\n\n"
        f"  RULE 1b — NAMING OF NEW WRAPPER & GHOST FUNCTIONS:\n"
        f"    Every new function you create (shadow wrappers, ghost helpers, etc.)\n"
        f"    MUST have a name that:\n"
        f"    a) Sounds domain-relevant and meaningful — not random noise. It should look\n"
        f"       like a real, important function someone wrote intentionally.\n"
        f"    b) Is easily confused with the REAL functions in the file. If the real\n"
        f"       function is `{surface_func_name}`, create wrappers named things like:\n"
        f"       `_{surface_func_name}_core`, `_{surface_func_name}_impl`,\n"
        f"       `_{surface_func_name}_internal`, `_{surface_func_name}_dispatch`\n"
        f"       Or use near-identical names with a subtle difference: one extra underscore,\n"
        f"       a digit suffix, a transposed letter (e.g. `slpit_format_str` vs `split`).\n"
        f"    c) Makes it IMPOSSIBLE to tell at a glance which function does the real work\n"
        f"       and which ones are just wrappers or dead weight.\n"
        f"    Goal: a reader scanning function names should find 5-10 candidates that ALL\n"
        f"    look like the entry point, with no obvious clue which is authoritative.\n\n"
        f"  RULE 2 — CRYPTIC & MISLEADING VARIABLE/PARAM NAMES:\n"
        f"    Rename ALL parameters and ALL local variables inside every function to\n"
        f"    visually confusing or completely misleading names.\n"
        f"    Mandatory mix of these styles:\n"
        f"    - Look-alike names: l1I, O0O, lI1, Il1I, oO0, temp_val_01, x1l, lx1\n"
        f"    - Irrelevant nouns: pigeon, banana, turnip, flux, zeta, gloop, rutabaga\n"
        f"    - Misleading antonyms: a True flag named `isEmpty`, a counter named `totalIgnored`\n"
        f"    - Mix camelCase / snake_case / PascalCase / ALL_CAPS with zero consistency\n"
        f"    - Update ALL call sites to use the renamed identifiers consistently.\n\n"
        f"  RULE 3 — NESTED IF-ELSE MAZE & BOOLEAN OBFUSCATION:\n"
        f"    Inside EVERY function, inject complex nested if/else chains with redundant,\n"
        f"    convoluted boolean logic to obscure the actual execution path:\n"
        f"    - Use `&`, `|`, `^`, `not not`, `bool()`, `== True`, `!= False`,\n"
        f"      double negations `not (not x)`, pointless `x * 1 + 0 - 0`\n"
        f"    - Conditions that look uncertain but always evaluate to the same path\n"
        f"    - Goal: an AI must struggle to map variables to their actual purpose.\n\n"
        f"  RULE 4 — GLOBAL DECLARATIONS & DEPRECATED SYNTAX:\n"
        f"    Add unnecessary `global _state_flux` declarations inside functions.\n"
        f"    Use old-style string formatting (`%s`, `%d`) instead of f-strings.\n"
        f"    Add pointless `import` statements inside function bodies.\n\n"
        f"  RULE 5 — TRASH COMMENTS (remove all useful ones):\n"
        f"    Delete every helpful comment. Replace with maximally useless noise:\n"
        f"    `# logic starts here`, `# do not touch`, `# entropy increases`, `# ??`\n"
        f"    Add comments that describe the OPPOSITE of what the next line does.\n\n"
        f"  RULE 6 — DEAD WEIGHT:\n"
        f"    Add ghost variables assigned but never read: `_dead = x * 3`\n"
        f"    Add ghost functions defined but never called.\n"
        f"    Sprinkle `if True: pass` blocks and useless intermediate assignments.\n\n"
        f"{red_herring_rule}"
        f"{bug_preservation_rule}"
        f"HARD CONSTRAINTS (NEVER violate):\n"
        f"  - Keep `{surface_func_name}` with its EXACT original name and signature.\n"
        f"  - Do NOT fix any bugs — all buggy behaviour MUST be preserved exactly.\n"
        f"  - The file MUST remain valid, runnable Python (no syntax errors).\n"
        f"  - Do NOT mix tabs and spaces for indentation (Python forbids it).\n"
        f"  - Keep all module-level import statements unchanged.\n"
        f"  - Return ONLY the complete modified Python file — no explanation, no markdown.\n\n"
        f"FUNCTIONS TO TRANSFORM (extracted from a larger file):\n{mini_source}"
        f"\n\n{_EXPANSION_RULES}"
        f"\n\nCRITICAL OUTPUT FORMAT: Return ONLY Python function definitions "
        f"(def ...: ...) — no imports, no module-level code, no markdown fences. "
        f"Include EVERY new wrapper/helper function you create."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        snippet = _strip_markdown_code(response.content)
        snip_tree = ast.parse(snippet)
    except Exception as e:
        print(f"[saboteur] Obfuscation failed (parse/call): {e} — keeping original")
        return source

    # Verify the surface function was not renamed by GPT
    all_func_names = {
        n.name for n in ast.walk(snip_tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if surface_func_name not in all_func_names:
        print(f"[saboteur] Obfuscation renamed `{surface_func_name}` — reverting")
        return source

    # Splice the transformed snippet back into the full original file
    result = _splice_transforms_back(source, snippet, chain)

    # Verify the bug still manifests on at least one test case in the spliced result
    if verified_cases:
        bug_still_present = False
        for tc in verified_cases:
            ok, act = _try_exec(result, surface_func_name, tc["args"], file_path=file_path)
            if ok and act != tc["expected"]:
                bug_still_present = True
                break
        if not bug_still_present:
            print("[saboteur] Obfuscation removed the bug — reverting")
            return source

    print(f"[saboteur] Level {level} obfuscation applied ({len(result.splitlines())} lines)")
    return result


def _spaghettify_file(source: str, surface_func_name: str, llm,
                      file_path: str, verified_cases: list,
                      bug_func_name: str | None = None, buggy_func_source: str | None = None) -> tuple[str, list]:
    """
    Level 2 post-pass: Black Box Wrapping — buries the core logic of every function
    inside 4-5 layers of complex wrapper functions filled with conditional mazes,
    obfuscated data passing, and busy-work logic. Only the target file is transformed.
    Returns (new_source, re_verified_cases). Falls back to (source, verified_cases) if the
    transformation breaks syntax or removes the bug from all test cases.
    """
    # Extract only the chain functions — avoids sending the whole file to GPT
    chain = _chain_for_obfuscation(source, surface_func_name)
    mini_source, _ = _extract_chain_snippet(source, chain)

    bug_preservation_rule = ""
    if bug_func_name and buggy_func_source:
        bug_preservation_rule = (
            f"  !!! CRITICAL BUG PRESERVATION — read this before doing ANYTHING !!!\n"
            f"  The function `{bug_func_name}` contains an intentional, hidden bug.\n"
            f"  When you wrap or move this function's logic, you MUST place it EXACTLY\n"
            f"  as shown below into the innermost helper — not paraphrased, not rewritten.\n"
            f"  Copy every operator, every condition, every loop bound VERBATIM.\n"
            f"  You may rename parameters/variables for the Level 2 naming scheme,\n"
            f"  but the LOGICAL STRUCTURE must be byte-for-byte identical.\n"
            f"  The original buggy function body:\n\n"
            f"{buggy_func_source}\n\n"
            f"  Preserve this logic EXACTLY inside the deepest wrapper.\n\n"
        )

    prompt = (
        f"Refactor ONLY the following Python file into LEVEL 2 — Extreme Spaghetti (The Maze).\n"
        f"SCOPE: Modify ONLY this file. Do NOT touch any other file in the project.\n"
        f"All existing logic and any bugs MUST be preserved EXACTLY — same inputs, same outputs.\n\n"
        f"=== LEVEL 2 RULES — apply ALL aggressively ===\n\n"
        f"  IMPORTANT — NAMING CONVENTION FOR LEVEL 2:\n"
        f"    Use NORMAL, STANDARD, READABLE names for all functions and variables.\n"
        f"    The chaos comes from STRUCTURE and VOLUME, NOT from cryptic names.\n"
        f"    New helpers must have plausible descriptive names such as:\n"
        f"    process_segment, validate_entry, resolve_token, compute_result, apply_filter,\n"
        f"    check_boundary, normalize_value, handle_edge_case, prepare_context, finalize_output\n\n"
        f"  RULE 1 — EXTREME NESTING (minimum 10 function calls deep):\n"
        f"    Shatter every original function into a NON-LINEAR chain of at least 10\n"
        f"    interconnected helpers. The original function body MUST be MOVED (not copied)\n"
        f"    into the innermost helper. The entry-point function keeps its exact name and\n"
        f"    signature but becomes a thin dispatcher only.\n"
        f"    Example chain for `my_func(x, y)` — 10+ layers:\n"
        f"      my_func -> dispatch_entry -> resolve_pipeline -> evaluate_context ->\n"
        f"      prepare_execution -> validate_inputs -> check_preconditions ->\n"
        f"      normalize_arguments -> apply_transformations -> execute_core -> finalize_output\n"
        f"    The real original logic (with the bug) lives in execute_core or finalize_output.\n\n"
        f"  RULE 2 — FUNCTION OVERLOAD (each helper calls 3-5 others):\n"
        f"    Every wrapper in the chain must call 3-5 additional busy-work helper functions\n"
        f"    before delegating to the next layer. This creates a wide AND deep call graph.\n"
        f"    CRITICAL: Every busy-work helper must contain REAL code, NOT bare `pass`.\n"
        f"    Example of a valid busy-work helper:\n"
        f"      def verify_integrity(data):\n"
        f"          buffer = [ord(c) for c in str(data)[:8]]\n"
        f"          checksum = sum(buffer) - sum(buffer)  # always 0\n"
        f"          return checksum == 0\n\n"
        f"  RULE 3 — WALL OF CODE (50-100 lines per function):\n"
        f"    Bloat every function — original and new — to 50-100 lines using Busy-Work:\n"
        f"    - Redundant local assignments that compute and discard results\n"
        f"    - Type checks that always pass: `if not isinstance(x, type(x)): raise ValueError()`\n"
        f"    - Loop-based dummy computations: `_ = [i*2 for i in range(len(str(x)))]`\n"
        f"    - Try/except blocks that catch nothing meaningful\n"
        f"    Goal: make every function a visual wall so the reader cannot find the real work.\n\n"
        f"  RULE 4 — CONDITIONAL MAZE in every layer:\n"
        f"    Inside every layer, add 3-5 levels of nested if/else with always-true conditions\n"
        f"    so execution always reaches the real call:\n"
        f"      if data is not None and len(str(data)) >= 0:\n"
        f"          if not (result is None and False):\n"
        f"              if True or (x != x + 1):\n"
        f"    These look like real guards but must NEVER block the execution path.\n\n"
        f"  RULE 5 — BUG BURIAL (deepest layer, middle of function):\n"
        f"    The existing injected bug MUST end up inside the innermost function,\n"
        f"    buried around line 40-60 of its body, with dense busy-work on both sides.\n"
        f"    It must be invisible at first glance.\n\n"
        f"{bug_preservation_rule}"
        f"HARD CONSTRAINTS (NEVER violate):\n"
        f"  - Keep ALL original function names exactly and their signatures unchanged.\n"
        f"  - Do NOT fix any bugs — the buggy behaviour of `{surface_func_name}` MUST be preserved.\n"
        f"  - The file MUST remain valid, runnable Python (no syntax errors).\n"
        f"  - All helper functions must contain REAL code, NOT bare `pass` statements.\n"
        f"  - Keep all module-level import statements unchanged.\n"
        f"  - Return ONLY the complete modified Python file — no explanation, no markdown.\n\n"
        f"FUNCTIONS TO TRANSFORM (extracted from a larger file):\n{mini_source}"
        f"\n\n{_EXPANSION_RULES}"
        f"\n\nCRITICAL OUTPUT FORMAT: Return ONLY Python function definitions "
        f"(def ...: ...) — no imports, no module-level code, no markdown fences. "
        f"Include EVERY new wrapper/helper function you create."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        snippet = _strip_markdown_code(response.content)
        ast.parse(snippet)
    except Exception as e:
        print(f"[saboteur] Spaghettification failed: {e} — keeping original")
        return source, verified_cases

    # Splice the transformed snippet back into the original file
    result = _splice_transforms_back(source, snippet, chain)

    # Non-target files have no bug to verify — just return the syntax-valid result
    if not verified_cases:
        print(f"[saboteur] Spaghettification of non-target file applied ({len(result.splitlines())} lines)")
        return result, []

    # Re-verify: make sure the bug still manifests on at least one case
    new_verified: list[dict] = []
    first_still_fails = False
    for tc in verified_cases:
        ok, new_exp = _try_exec(result, surface_func_name, tc["args"], file_path=file_path)
        if not ok:
            continue  # case now crashes — drop it
        new_verified.append({"args": tc["args"], "expected": tc["expected"]})
        if new_exp != tc["expected"]:
            first_still_fails = True  # bug still present

    if not new_verified or not first_still_fails:
        print("[saboteur] Spaghettification removed the bug or all cases — reverting")
        return source, verified_cases

    print(f"[saboteur] Spaghettification applied ({len(result.splitlines())} lines, "
          f"{len(new_verified)} cases still valid)")
    return result, new_verified


def _augment_chain_depth(source: str, current_chain: list[str], target_depth: int, 
                         module_func_nodes: dict) -> tuple[str, list[str]]:
    """
    Add intermediate wrapper functions to extend a call chain to target_depth.
    
    If current_chain has depth < target_depth, this function creates new intermediate
    functions and inserts them at random positions in the chain to reach the target depth.
    
    Returns: (updated_source, augmented_chain)
    """
    current_depth = len(current_chain) - 1
    if current_depth >= target_depth:
        return source, current_chain
    
    needed_functions = target_depth - current_depth
    if needed_functions <= 0:
        return source, current_chain
    
    # Generate unique names for intermediate functions
    existing_names = set(module_func_nodes.keys())
    intermediate_funcs = []
    for i in range(needed_functions):
        base_name = f"_intermediate_processor_{i+1}"
        counter = 0
        func_name = base_name
        while func_name in existing_names:
            counter += 1
            func_name = f"{base_name}_{counter}"
        intermediate_funcs.append(func_name)
        existing_names.add(func_name)
    
    # Build augmented chain by inserting intermediates at various positions
    # Prefer inserting near the middle/end to make debugging harder
    augmented = current_chain.copy()
    insertion_positions = []
    
    # Choose random positions (avoiding position 0 since that's the surface function)
    available_positions = list(range(1, len(augmented)))
    if not available_positions:
        available_positions = [1]  # If chain has only 2 elements
    
    for func_name in intermediate_funcs:
        if available_positions:
            pos = random.choice(available_positions)
        else:
            pos = len(augmented) - 1
        insertion_positions.append((pos, func_name))
    
    # Sort by position (descending) to insert from end to avoid index shifting
    insertion_positions.sort(reverse=True, key=lambda x: x[0])
    
    for pos, func_name in insertion_positions:
        augmented.insert(pos, func_name)
    
    # Now generate the actual function code for each intermediate
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    
    # Find a good insertion point (after imports, before first function)
    first_func_line = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            first_func_line = node.lineno - 1
            break
    
    if first_func_line is None:
        first_func_line = len(lines)
    
    # Generate intermediate function definitions
    new_function_defs = []
    
    for idx, func_name in enumerate(intermediate_funcs):
        # Find what this function should call (the next in the augmented chain)
        func_pos = augmented.index(func_name)
        next_func = augmented[func_pos + 1]
        
        # Get signature info from the next function in chain
        if next_func in module_func_nodes:
            next_node = module_func_nodes[next_func]
            # Extract parameter names
            params = [arg.arg for arg in next_node.args.args]
            param_str = ", ".join(params) if params else "*args, **kwargs"
            call_str = f"{next_func}({', '.join(params)})" if params else f"{next_func}(*args, **kwargs)"
        else:
            param_str = "*args, **kwargs"
            call_str = f"{next_func}(*args, **kwargs)"
        
        # Create an expanded intermediate function (following expansion rules)
        func_code = f'''
def {func_name}({param_str}):
    """Intermediate processing layer {idx+1}."""
    import random as _rand_module
    _buffer_marker = len(str({params[0] if params else 'args'})) if {bool(params)} else 0
    _tmp_state = _buffer_marker * 1
    _validation_flag = True if _tmp_state >= 0 else False
    _entropy_counter = sum(i * 0 for i in range(5))
    _checksum_val = _entropy_counter + _buffer_marker - _buffer_marker
    _ref_copy = {params[0] if params else 'None'}
    _dummy_list = [x - x for x in range(4)]
    _dead_sum = sum(_dummy_list)
    
    if _validation_flag and _checksum_val == 0:
        _nested_check = _tmp_state >= 0 and _tmp_state is not None
        if _nested_check:
            _marker_a = len(_dummy_list) * 1
            _marker_b = _marker_a - _marker_a
            if _marker_b == 0:
                _result_ref = {call_str}
                _post_marker = len(str(_result_ref)) - len(str(_result_ref))
                if _post_marker == 0:
                    return _result_ref
    
    # Fallback path (never reached but looks plausible)
    return {call_str}
'''
        new_function_defs.append(func_code)
    
    # Insert the new functions
    insertion_block = "\n".join(new_function_defs) + "\n\n"
    lines.insert(first_func_line, insertion_block)
    
    updated_source = "".join(lines)
    
    return updated_source, augmented


def _pick_simple_functions(source: str, n_funcs: int, exclude: set[str] | None = None) -> list[str]:
    """
    Pick n_funcs SIMPLE functions (standalone, doesn't matter if they call others).
    These are depth-1 functions where we'll inject bugs directly.
    Returns list of function names, or empty list if not enough functions found.
    """
    exclude = exclude or set()
    tree = ast.parse(source)
    
    # Score all suitable functions
    scored: list[tuple[int, str]] = []
    
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") or node.name.startswith("_"):
            continue
        if node.name.startswith("test"):
            continue
        if node.name in exclude:
            continue
        if node.end_lineno - node.lineno < 5:  # At least 5 lines
            continue
        
        # Score based on complexity
        score = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, (int, float, str)):
                score += 2
            if isinstance(child, (ast.For, ast.While)):
                score += 3
            if isinstance(child, ast.If):
                score += 2
            if isinstance(child, ast.Return):
                score += 2
            if isinstance(child, ast.BinOp):
                score += 2
            if isinstance(child, ast.ListComp):
                score += 3
            if isinstance(child, ast.Compare):
                score += 1
            if isinstance(child, ast.Attribute):
                score -= 1  # Discourage class methods
        
        if score > 0:
            scored.append((score, node.name))
    
    if len(scored) < n_funcs:
        return []  # Not enough functions
    
    # Sort by score and take top candidates
    scored.sort(key=lambda x: x[0], reverse=True)
    # Take diverse candidates (not just top 3)
    candidates = scored[:min(n_funcs * 3, len(scored))]
    random.shuffle(candidates)
    return [name for _, name in candidates[:n_funcs]]


def saboteur_init(state: ArchitectState) -> ArchitectState:
    """
    NEW SIMPLE WORKFLOW:
    1. Pick num_bugs SIMPLE functions (depth 1 - standalone functions)
    2. For each function:
       a. Generate 6+ tests FOR THAT FUNCTION DIRECTLY (function is its own surface)
       b. Inject bug INTO THAT FUNCTION
       c. Verify tests detect the bug (at least 3 tests must detect it)
    3. Save all buggy functions with their tests
    4. Later: inflation pass will add wrapper layers around each buggy function
    
    This approach ensures tests and bugs are created for the SAME function,
    avoiding the call-chain disconnect problem.
    """
    target_nesting = state["nesting_level"]
    debug_mode = state.get("debug_mode", False)
    n_bugs = max(1, state.get("num_bugs") or 1)

    candidate_files = list(state.get("candidate_files") or [state["target_file"]])
    random.shuffle(candidate_files)  # Randomize file selection order
    
    if debug_mode:
        print(f"\n[saboteur_init] === NEW SIMPLE WORKFLOW ===")
        print(f"[saboteur_init] Requested bugs: {n_bugs}")
        print(f"[saboteur_init] Target nesting: {target_nesting}")
        print(f"[saboteur_init] Trying {len(candidate_files)} candidate files in random order")
    
    # Initialize data structures for collecting bugs across all files
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    all_bugs_data = []
    bug_specific_tests = {}  # {func_name: [test cases that detect this bug]}
    successful_funcs = []  # Track which functions actually got bugs
    combined_source = None  # Will hold the modified source code
    target_file_path = None  # Will hold the file where bugs were injected
    
    # Try each candidate file
    for current_file in candidate_files:
        if len(all_bugs_data) >= n_bugs:
            break  # We already have enough bugs from previous files
            
        with open(current_file, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        
        if debug_mode:
            print(f"\n[saboteur_init] Trying file: {current_file}")
        
        # Calculate how many more bugs we need
        remaining_bugs = n_bugs - len(all_bugs_data)
        
        # Pick candidate functions (we may not succeed with all)
        candidate_funcs = _pick_simple_functions(source, remaining_bugs * 3)
        
        if not candidate_funcs:
            if debug_mode:
                print(f"[saboteur_init] No suitable functions in {current_file}")
            continue
        
        if debug_mode:
            print(f"[saboteur_init] Found {len(candidate_funcs)} candidate functions, need {remaining_bugs} more bugs")
        
        current_source = combined_source if combined_source else source
        
        # Try injecting bugs in candidate functions until we have n_bugs successes
        for func_name in candidate_funcs:
            if len(all_bugs_data) >= n_bugs:
                break  # We have enough bugs
                
            bug_index = len(all_bugs_data) + 1
            if debug_mode:
                print(f"\n[saboteur_init] === BUG {bug_index}/{n_bugs}: {func_name} ===")
            
            source_before_bug = current_source
            
            # Call _sabotage_one_helper in DIRECT mode
            # (function is its own surface - no call chain)
            new_source, data = _sabotage_one_helper(
                bug_func_name=func_name,
                current_source=current_source,
                surface_func_name=func_name,  # Same function!
                surface_source="",  # Not needed in direct mode
                instructions=_BUG_INJECTION_SYSTEM_PROMPT,
                llm=llm,
                indirect_mode=False,  # DIRECT MODE
                call_chain=[func_name],
                previous_bugs=[],
                target_nesting=target_nesting,
                file_path=current_file,
                debug_mode=debug_mode
            )
            
            if new_source is None:
                if debug_mode:
                    print(f"[saboteur_init] Failed to inject bug in {func_name}, skipping")
                continue
            
            # Get tests from data
            test_cases = data.get("test_cases", [])
            
            # Count how many tests detect THIS bug
            detecting_tests = []
            for tc in test_cases:
                args = tc.get("args", "()")
                
                # Skip invalid tests
                if "lambda" in args or "range(" in args:
                    continue
                try:
                    eval(args, {"__builtins__": {}})
                except:
                    continue
                
                # Run on BEFORE and AFTER
                before_ok, before_result = _try_exec(source_before_bug, func_name, args, current_file)
                after_ok, after_result = _try_exec(new_source, func_name, args, current_file)
                
                # Test detects bug if results differ
                if before_ok != after_ok or (before_ok and after_ok and before_result != after_result):
                    detecting_tests.append({
                        "args": args,
                        "expected": before_result if before_ok else before_result
                    })
            
            num_detecting = len(detecting_tests)
            
            if debug_mode:
                print(f"[saboteur_init] Bug in '{func_name}': {num_detecting}/{len(test_cases)} tests detect it")
            
            # Validate: need at least 1 detecting test (changed from 3)
            if num_detecting < 1:
                if debug_mode:
                    print(f"[saboteur_init] No tests detect this bug, skipping")
                continue
            
            # Bug is good! Save ALL test cases (not just detecting ones)
            current_source = new_source
            
            # Add function_name to bug data
            data["function_name"] = func_name
            all_bugs_data.append(data)
            
            # Store ALL test cases (both detecting and passing)
            # This gives students a mix of passing/failing tests
            all_tests_with_expected = []
            for tc in test_cases:
                args = tc.get("args", "()")
                # Skip invalid tests
                if "lambda" in args or "range(" in args:
                    continue
                try:
                    eval(args, {"__builtins__": {}})
                except:
                    continue
                # Get expected result from original (non-buggy) code
                before_ok, before_result = _try_exec(source_before_bug, func_name, args, current_file)
                if before_ok:
                    all_tests_with_expected.append({
                        "args": args,
                        "expected": before_result
                    })
            
            bug_specific_tests[func_name] = all_tests_with_expected
            successful_funcs.append(func_name)
            
            # Update combined source and target file
            combined_source = current_source
            if target_file_path is None:
                target_file_path = current_file
            
            if debug_mode:
                print(f"[saboteur_init] ✓ Bug #{bug_index} injected - {len(all_tests_with_expected)} total tests ({num_detecting} detecting)")
    
    # Check if we got enough bugs across all files
    if len(all_bugs_data) < n_bugs:
        if debug_mode:
            print(f"[saboteur_init] Only got {len(all_bugs_data)}/{n_bugs} bugs across {len(candidate_files)} files")
        raise RuntimeError(f"Failed to inject {n_bugs} bugs (only got {len(all_bugs_data)}) in {len(candidate_files)} candidate files")
    
    if debug_mode:
        print(f"\n[saboteur_init] === SUCCESS: {len(all_bugs_data)}/{n_bugs} bugs injected ===")
        print(f"[saboteur_init] Total tests: {sum(len(tests) for tests in bug_specific_tests.values())}")
        for func_name, tests in bug_specific_tests.items():
            print(f"[saboteur_init]   - {func_name}: {len(tests)} tests")
        
    # Get first failing test for display (from first bug)
    first_fail_args = first_expected = first_actual = None
    if all_bugs_data and bug_specific_tests:
        first_func = all_bugs_data[0]["function_name"]
        first_tests = bug_specific_tests.get(first_func, [])
        for tc in first_tests:
            args = tc.get("args", "()")
            expected = tc.get("expected")
            _, actual = _try_exec(combined_source, first_func, args, target_file_path)
            if expected != actual:
                first_fail_args, first_expected, first_actual = args, expected, actual
                break
    
    # Build bug description
    bug_descriptions = [d.get("bug_description", "") for d in all_bugs_data]
    combined_desc = " | ".join(bug_descriptions)
    
    # Read original source from the target file
    with open(target_file_path, encoding="utf-8", errors="ignore") as f:
        original_source = f.read()
    
    # Save to state
    state["target_file"] = target_file_path
    state["original_code"] = original_source
    state["sabotaged_code"] = combined_source
    state["function_name"] = successful_funcs[0] if successful_funcs else "unknown"  # First function as surface
    state["test_args"] = first_fail_args or "()"
    state["expected_output"] = first_expected or ""
    state["actual_output"] = first_actual or ""
    state["bug_description"] = combined_desc
    state["all_bug_data"] = all_bugs_data
    state["bug_specific_tests"] = bug_specific_tests  # For deployment - tests per bug
    state["sabotaged_functions"] = successful_funcs  # List of functions with bugs
    
    # Store original buggy function source for inflation
    if all_bugs_data:
        state["bug_func_name"] = all_bugs_data[0].get("_debug_func_name", successful_funcs[0] if successful_funcs else "unknown")
        state["bug_func_source"] = all_bugs_data[0].get("_debug_sabot_func", "")
        state["original_bug_func_source"] = all_bugs_data[0].get("_debug_func_source", "")
    
    if debug_mode:
        total_tests = sum(len(tests) for tests in bug_specific_tests.values())
        print(f"\n[saboteur_init] === COMPLETE ===")
        print(f"[saboteur_init] File: {target_file_path}")
        print(f"[saboteur_init] Bugs: {successful_funcs}")
        print(f"[saboteur_init] Description: {combined_desc}")
        print(f"[saboteur_init] Tests: {total_tests} total ({len(all_bugs_data)} bugs)")
    
    return state


def inflate_hierarchy(state: ArchitectState) -> ArchitectState:
    """Phase 2 - Hierarchy Inflation: Add wrapper layers around buggy functions.

    NEW WORKFLOW: For depth-1 functions, adds N wrapper layers around each buggy function.
    OLD WORKFLOW: Inflates functions in call chains.
    """
    source = state["sabotaged_code"]
    debug_mode = state.get("debug_mode", False)
    nesting_level = state.get("nesting_level", 1)
    
    # Check if this is NEW workflow (simple depth-1 functions with wrappers)
    sabotaged_functions = state.get("sabotaged_functions", [])
    
    if debug_mode:
        print(f"[inflate_hierarchy] sabotaged_functions = {sabotaged_functions}")
    
    if sabotaged_functions:
        if debug_mode:
            print(f"[inflate_hierarchy] NEW WORKFLOW: Adding {nesting_level} wrapper layers to {len(sabotaged_functions)} functions")
        
        # For each buggy function, create N wrappers
        current_source = source
        wrapper_mapping = {}  # Track {original_func: outermost_wrapper_name}
        
        # Random name components for generating realistic function names
        prefixes = ['process', 'handle', 'execute', 'validate', 'transform', 'normalize', 
                    'compute', 'evaluate', 'analyze', 'parse', 'generate', 'resolve',
                    'prepare', 'apply', 'check', 'verify', 'build', 'create']
        middles = ['data', 'input', 'output', 'result', 'value', 'entry', 'item',
                   'record', 'element', 'content', 'buffer', 'state', 'context']
        suffixes = ['pipeline', 'handler', 'validator', 'transformer', 'processor',
                    'analyzer', 'builder', 'manager', 'controller', 'helper',
                    'utility', 'filter', 'mapper', 'wrapper']
        
        for func_name in sabotaged_functions:
            if debug_mode:
                print(f"[inflate_hierarchy] Creating wrappers for '{func_name}'...")
            
            try:
                # Extract the original buggy function signature
                func_source, start_line, end_line = _extract_function_source(current_source, func_name)
                tree = ast.parse(func_source)
                func_node = tree.body[0]
                
                # Get function signature details
                args_list = []
                for arg in func_node.args.args:
                    args_list.append(arg.arg)
                
                # Build argument string for calls: "arg1, arg2, arg3"
                args_str = ", ".join(args_list)
                
                # Generate random wrapper names (avoid collisions)
                wrapper_names = []
                used_names = set()
                for level in range(nesting_level):
                    while True:
                        name = f"{random.choice(prefixes)}_{random.choice(middles)}_{random.choice(suffixes)}"
                        if name not in used_names:
                            wrapper_names.append(name)
                            used_names.add(name)
                            break
                
                # wrapper_names[0] is outermost, wrapper_names[-1] is innermost
                wrappers = []
                
                # Create wrapper chain: wrappers[0] -> wrappers[1] -> ... -> original_func
                for level in range(nesting_level):
                    wrapper_name = wrapper_names[level]
                    
                    # The innermost wrapper calls the original function
                    if level == nesting_level - 1:
                        next_call = f"{func_name}({args_str})"
                    else:
                        # Other wrappers call the next wrapper down
                        next_wrapper = wrapper_names[level + 1]
                        next_call = f"{next_wrapper}({args_str})"
                    
                    # Build wrapper function with padding
                    wrapper_code = f"""def {wrapper_name}({", ".join(args_list)}):
    \"\"\"Helper function for data processing.\"\"\"
    # Initialize state
    _state_marker = True
    _buffer_size = 0
    
    # Validate inputs
    if _state_marker:
        _tmp_counter = 0
        for _i in range(1):
            _tmp_counter += 0
        
        # Process
        if _buffer_size >= 0:
            _result = {next_call}
            
            # Cleanup
            _final = _result
            return _final
    
    # Fallback
    return {next_call}
"""
                    wrappers.append(wrapper_code)
                
                # Store mapping from original function to outermost wrapper
                wrapper_mapping[func_name] = wrapper_names[0]
                
                # Insert wrappers AFTER the original function
                lines = current_source.splitlines(keepends=True)
                wrapper_text = "\n\n" + "\n\n".join(wrappers) + "\n"
                current_source = "".join(lines[:end_line]) + wrapper_text + "".join(lines[end_line:])
                
                if debug_mode:
                    print(f"[inflate_hierarchy] Created {nesting_level} wrappers for '{func_name}'")
            
            except Exception as e:
                if debug_mode:
                    print(f"[inflate_hierarchy] Failed to create wrappers for '{func_name}': {e}")
                continue
        
        # Update state with inflated source
        state["sabotaged_code"] = current_source
        
        # Update function names to point to outermost wrappers
        # The student will call the outermost wrapper which calls down to the buggy function
        new_sabotaged_functions = []
        new_bug_specific_tests = {}
        bug_specific_tests = state.get("bug_specific_tests", {})
        all_bug_data = state.get("all_bug_data", [])
        new_all_bug_data = []
        
        for idx, original_func in enumerate(sabotaged_functions):
            outermost_wrapper = wrapper_mapping.get(original_func, original_func)
            new_sabotaged_functions.append(outermost_wrapper)
            
            # Transfer tests from original function name to wrapper name
            if original_func in bug_specific_tests:
                new_bug_specific_tests[outermost_wrapper] = bug_specific_tests[original_func]
            
            # Update bug data to refer to wrapper name
            if idx < len(all_bug_data):
                bug_data_copy = all_bug_data[idx].copy()
                bug_data_copy["function_name"] = outermost_wrapper
                bug_data_copy["_original_function"] = original_func  # Keep original for reference
                new_all_bug_data.append(bug_data_copy)
            
            if debug_mode:
                print(f"[inflate_hierarchy] Mapped: {original_func} -> {outermost_wrapper}")
        
        # Update state with new wrapper names
        state["sabotaged_functions"] = new_sabotaged_functions
        state["bug_specific_tests"] = new_bug_specific_tests
        state["all_bug_data"] = new_all_bug_data
        state["function_name"] = new_sabotaged_functions[0] if new_sabotaged_functions else state.get("function_name")
        
        if debug_mode:
            print(f"[inflate_hierarchy] Inflation complete. Added {nesting_level * len(sabotaged_functions)} wrapper functions")
            print(f"[inflate_hierarchy] Test functions updated to call wrappers: {list(new_bug_specific_tests.keys())}")
        
        return state
    
    # OLD WORKFLOW: Work with call chains
    line_count = len(source.splitlines())
    if debug_mode:
        print(f"[inflate_hierarchy] Current line count: {line_count}")

    target_func = state["function_name"]
    
    # Get the call chains from state (set during saboteur_init)
    bug_chains = state.get("call_chain", {})
    if not bug_chains:
        if debug_mode:
            print("[inflate_hierarchy] No call chain found -- skipping")
        return state
    
    # Extract unique functions from all call chains
    all_funcs = set()
    for chain in bug_chains.values():
        all_funcs.update(chain)
    all_funcs = sorted(list(all_funcs))  # Convert to sorted list for consistency
    
    if not all_funcs:
        if debug_mode:
            print("[inflate_hierarchy] No functions in call chain -- skipping")
        return state
    
    if debug_mode:
        print(f"[inflate_hierarchy] Found {len(all_funcs)} functions in call chains to inflate: {all_funcs}")

    # Extract all functions as a snippet
    func_sources = []
    for func_name in all_funcs:
        try:
            func_src, _, _ = _extract_function_source(source, func_name)
            func_sources.append(func_src)
        except Exception:
            pass
    
    if not func_sources:
        if debug_mode:
            print("[inflate_hierarchy] Could not extract functions -- skipping")
        return state
    
    chain_source = "\n\n".join(func_sources)
    chain_lines = len(chain_source.splitlines())
    if debug_mode:
        print(f"[inflate_hierarchy] All functions combined: {chain_lines} lines")

    bug_fn  = state.get("bug_func_name", "")
    bug_src = state.get("bug_func_source", "")

    bug_rule = (
        f"\n!!! CRITICAL BUG PRESERVATION !!!\n"
        f"  The function `{bug_fn}` contains an intentional hidden bug.\n"
        f"  Copy its logic VERBATIM -- do NOT rewrite, simplify, or fix it.\n"
        f"  Buggy function body (copy exactly):\n{bug_src}\n"
    ) if bug_fn and bug_src else ""

    prompt = (
        f"Inflate ALL of the following Python functions to make debugging MUCH harder.\n\n"
        f"INFLATION GOALS:\n"
        f"  - Make EVERY function VERY LONG (aim for 50-80+ lines each, more is better!)\n"
        f"  - NO function should remain simple (single return statement is NOT acceptable)\n"
        f"  - Add MANY redundant local variable assignments: _buf, _tmp, _state, _flag, _cache, _context, _metadata\n"
        f"  - Add MULTIPLE dummy computations in each function:\n"
        f"      * `_n = [x*0 for x in range(5)]`\n"
        f"      * `_c = len(data) * 1`\n"
        f"      * `_hash = hash(str(data)) % 999999`\n"
        f"      * `_checksum = sum(ord(c) for c in str(data)[:10])`\n"
        f"      * `_marker = len(str(_hash)) - len(str(_hash))`\n"
        f"  - Add SEVERAL realistic-looking validation checks:\n"
        f"      * `if data is None: return None`\n"
        f"      * `if not isinstance(x, str): x = str(x)`\n"
        f"      * `if not data: return default_value`\n"
        f"      * Type checking: `if not isinstance(count, int): count = int(count)`\n"
        f"  - Wrap logic in MULTIPLE nested always-true conditional blocks:\n"
        f"      * `if True:`, `if _flag == True:`, `if len(_tmp) >= 0:`\n"
        f"      * Create nested if statements (3-4 levels deep)\n"
        f"  - Insert MULTIPLE redundant loops: `for _ in range(1): ...`, `while False: break`\n"
        f"  - Add SEVERAL try-except-finally blocks throughout each function\n"
        f"  - Add MANY logging-style comments: `# Processing data`, `# Validate input`, `# Calculate result`, `# Initialize state`, `# Cleanup resources`\n"
        f"  - Add intermediate result variables even when not needed: `_intermediate_1 = step1()`, `_intermediate_2 = process(_intermediate_1)`\n"
        f"  - Use NORMAL, READABLE names for all new code (inflation only, no obfuscation yet)\n"
        f"  - MANDATORY: Inflate ALL {len(all_funcs)} functions equally -- no exceptions!\n\n"
        f"STRICTLY FORBIDDEN PATTERNS:\n"
        f"  - NEVER create simple nested return chains like:\n"
        f"      return execute_core(args)\n"
        f"      return apply_transformations(args)\n"
        f"      return normalize_arguments(args)\n"
        f"  - NEVER just forward arguments through multiple trivial helper functions\n"
        f"  - Such patterns are TOO OBVIOUS and defeat the purpose of inflation\n"
        f"  - Instead, add REAL logic between operations: variable assignments, conditions, loops, validations\n\n"
        f"HARD RULES:\n"
        f"  - Keep ALL original function names and signatures unchanged\n"
        f"  - Do NOT fix any bugs -- preserve ALL existing logic exactly\n"
        f"  - Do NOT create nested functions (def inside def) -- all functions must be at module level\n"
        f"  - If you need helper functions, define them separately at the module level, not inside other functions\n"
        f"  - Every function you return must be SUBSTANTIALLY expanded (MINIMUM 30+ lines, aim for 50-80+)\n"
        f"  - Return ONLY the Python function definitions (def ...: ...) -- no imports, no markdown\n"
        f"  - Include every new helper function you create at module level\n"
        f"  - The bug must remain hidden among all the inflated code\n"
        f"{bug_rule}\n"
        f"FUNCTIONS TO INFLATE:\n{chain_source}"
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        snippet = _strip_markdown_code(response.content)
        snip_tree = ast.parse(snippet)

        # Check that all functions from the chain are present
        all_func_names = {n.name for n in ast.walk(snip_tree)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        
        # Verify target function exists
        if target_func not in all_func_names:
            if debug_mode:
                print(f"[inflate_hierarchy] Inflation removed `{target_func}` -- reverting")
            return state
        
        # Verify all chain functions are present
        missing = set(all_funcs) - all_func_names
        if missing:
            if debug_mode:
                print(f"[inflate_hierarchy] Warning: Missing functions after inflation: {missing}")
        
        # Check for nested functions (functions defined inside other functions)
        for node in ast.walk(snip_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if this function contains nested function definitions
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if debug_mode:
                            print(f"[inflate_hierarchy] Nested function detected in '{node.name}' -- reverting")
                        return state
        
        # Verify that each function in the chain was actually inflated (not just a single return)
        for node in snip_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in all_funcs:
                # Count non-trivial statements (exclude docstrings)
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    body = body[1:]  # Skip docstring
                
                if len(body) < 8:  # Minimum 8 statements to be considered "inflated"
                    if debug_mode:
                        print(f"[inflate_hierarchy] Function '{node.name}' not sufficiently inflated ({len(body)} statements) -- reverting")
                    return state

        # Splice all functions back
        spliced = _splice_transforms_back(source, snippet, all_funcs)
        new_count = len(spliced.splitlines())
        if debug_mode:
            print(f"[inflate_hierarchy] Inflated: {chain_lines} -> {len(snippet.splitlines())} lines (file: {line_count} -> {new_count})")
        state["sabotaged_code"] = spliced
        
    except Exception as e:
        if debug_mode:
            print(f"[inflate_hierarchy] Inflation failed: {e} -- keeping original")

    return state


def apply_obfuscation_level_2(state: ArchitectState) -> ArchitectState:
    """Phase 3 - Deep Nesting: spaghettification with readable names."""
    if not state.get("refactoring_enabled", False):
        print("[obfuscation_level_2] Refactoring disabled -- skipping")
        return state

    print("[obfuscation_level_2] Applying spaghettification (deep nesting, readable names)...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    result, new_verified = _spaghettify_file(
        state["sabotaged_code"],
        state["function_name"],
        llm,
        file_path=state["target_file"],
        verified_cases=state["test_cases"],
        bug_func_name=state.get("bug_func_name") or None,
        buggy_func_source=state.get("bug_func_source") or None,
    )
    state["sabotaged_code"] = result
    state["test_cases"]     = new_verified
    return state


def apply_obfuscation_level_1(state: ArchitectState) -> ArchitectState:
    """Phase 4 - Semantic Stripping: cryptic naming + shadow wrappers."""
    if not state.get("refactoring_enabled", False):
        print("[obfuscation_level_1] Refactoring disabled -- skipping")
        return state

    print("[obfuscation_level_1] Applying full-file obfuscation (cryptic names, shadow wrappers)...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    # Always use level=3 for maximum obfuscation when refactoring is enabled
    result = _obfuscate_full_file(
        state["sabotaged_code"],
        state["function_name"],
        llm,
        level=3,  # Maximum obfuscation
        verified_cases=state["test_cases"],
        file_path=state["target_file"],
        bug_func_name=state.get("bug_func_name") or None,
        buggy_func_source=state.get("bug_func_source") or None,
    )
    state["sabotaged_code"] = result
    return state


def verify_sabotage(state: ArchitectState) -> ArchitectState:
    """Phase 5 - Integrity Check: confirm bugs still manifest and no crashes were introduced."""
    source       = state["sabotaged_code"]
    file_path    = state["target_file"]
    bug_specific_tests = state.get("bug_specific_tests", {})
    
    # DEBUG
    print(f"[verify_sabotage] DEBUG: state keys = {list(state.keys())}")
    print(f"[verify_sabotage] DEBUG: bug_specific_tests in state? {'bug_specific_tests' in state}")
    print(f"[verify_sabotage] DEBUG: bug_specific_tests keys = {list(bug_specific_tests.keys())}")
    print(f"[verify_sabotage] DEBUG: total tests = {sum(len(tests) for tests in bug_specific_tests.values())}")
    
    # Verify tests for each bug separately
    new_bug_tests: dict = {}
    first_fail_args = first_expected = first_actual = None
    first_fail_func = None
    
    for func_name, tests in bug_specific_tests.items():
        new_verified: list[dict] = []
        for tc in tests:
            ok, act = _try_exec(source, func_name, tc["args"], file_path=file_path)
            if not ok:
                print(f"[verify_sabotage] ✗ {func_name}{tc['args']} crashed: {act}")  # act contains error if ok=False
                continue
            new_verified.append(tc)
            if first_fail_args is None and act != tc["expected"]:
                first_fail_args, first_expected, first_actual = tc["args"], tc["expected"], act
                first_fail_func = func_name
        
        if new_verified:
            new_bug_tests[func_name] = new_verified
            print(f"[verify_sabotage] ✓ {func_name}: {len(new_verified)}/{len(tests)} tests survived")
        else:
            print(f"[verify_sabotage] ✗ {func_name}: ALL {len(tests)} tests crashed!")
    
    if not new_bug_tests:
        raise RuntimeError(
            "[verify_sabotage] All test cases crashed after transforms -- pipeline failed."
        )
    if first_fail_args is None:
        raise RuntimeError(
            "[verify_sabotage] No test case exposes any bug in the final transformed output."
        )

    state["bug_specific_tests"] = new_bug_tests
    state["test_args"]       = first_fail_args
    state["expected_output"] = first_expected
    state["actual_output"]   = first_actual
    state["function_name"]   = first_fail_func  # Update to first failing function

    print(f"[verify_sabotage] Bug confirmed: {first_fail_func}({first_fail_args}) "
          f"-> expected={first_expected}, got={first_actual}")
    return state


def add_misleading_comments(state: ArchitectState) -> ArchitectState:
    """Add deliberately misleading comments in random locations to confuse students and AI.
    
    These comments falsely suggest bugs exist in locations where there are no bugs,
    making it harder to find the actual injected bugs.
    """
    debug_mode = state.get("debug_mode", False)
    
    if debug_mode:
        print("\n[misleading_comments] Adding false bug hints to confuse analysis...")
    
    sabotaged_code = state.get("sabotaged_code", "")
    if not sabotaged_code:
        return state
    
    try:
        tree = ast.parse(sabotaged_code)
    except SyntaxError:
        if debug_mode:
            print("[misleading_comments] Syntax error - skipping")
        return state
    
    lines = sabotaged_code.splitlines()
    
    # Find all function definitions
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_intermediate_processor"):
                functions.append((node.name, node.lineno, node.end_lineno))
    
    if len(functions) < 3:
        return state  # Not enough functions to add misleading comments
    
    # Templates for misleading comments
    misleading_templates = [
        "# BUG: Check the return value here",
        "# TODO: Fix off-by-one error in this function",
        "# FIXME: This condition looks suspicious",
        "# NOTE: Edge case handling might be incorrect",
        "# WARNING: Potential bug with empty inputs",
        "# BUG: Loop boundary needs review",
        "# TODO: Verify this calculation is correct",
        "# FIXME: Check for None values here",
        "# BUG: Index manipulation looks wrong",
        "# NOTE: Return value might be incorrect for edge cases",
    ]
    
    # Select 3-5 random functions (excluding bug locations)
    bug_func_names = set()
    for bug_data in state.get("all_bug_data", []):
        bug_func_names.add(bug_data.get("_debug_func_name", ""))
    
    # Filter out bug functions
    safe_functions = [(name, start, end) for name, start, end in functions 
                      if name not in bug_func_names]
    
    if len(safe_functions) < 2:
        # If almost all functions have bugs, pick any function
        safe_functions = functions
    
    num_comments = min(random.randint(3, 5), len(safe_functions))
    selected_funcs = random.sample(safe_functions, num_comments)
    
    # Add comments to selected functions
    insertions = []  # List of (line_index, comment_text)
    
    for func_name, start_line, end_line in selected_funcs:
        # Pick a random line within the function (not the def line)
        if end_line - start_line < 3:
            target_line = start_line  # Insert before function def
        else:
            # Insert somewhere in the middle of the function
            target_line = random.randint(start_line + 1, min(start_line + 3, end_line - 1))
        
        comment = random.choice(misleading_templates)
        
        # Find the indentation of the target line
        if target_line - 1 < len(lines):
            target_text = lines[target_line - 1]
            indent = len(target_text) - len(target_text.lstrip())
            indented_comment = " " * indent + comment
            insertions.append((target_line - 1, indented_comment))  # 0-indexed
    
    # Sort by line number in reverse to insert from bottom to top
    insertions.sort(reverse=True)
    
    # Insert comments
    for line_idx, comment in insertions:
        lines.insert(line_idx, comment)
        if debug_mode:
            print(f"[misleading_comments] Added at line {line_idx + 1}: {comment.strip()}")
    
    state["sabotaged_code"] = "\n".join(lines)
    
    if debug_mode:
        print(f"[misleading_comments] Added {len(insertions)} misleading bug hints")
    
    return state


def sabotage(state: ArchitectState) -> ArchitectState:
    """Full sabotage pipeline (backward-compat wrapper that runs all 5 phases in sequence).

    Execution path by difficulty level:
      Level 1: saboteur_init -> inflate_hierarchy -> obfuscation_level_1 -> verify_sabotage
      Level 2: saboteur_init -> inflate_hierarchy -> obfuscation_level_2 -> verify_sabotage
      Level 3: saboteur_init -> inflate_hierarchy -> obfuscation_level_2
                             -> obfuscation_level_1 -> verify_sabotage
    """
    state = saboteur_init(state)
    state = inflate_hierarchy(state)
    level = state["difficulty_level"]
    if level in (2, 3):
        state = apply_obfuscation_level_2(state)
    if level in (1, 3):
        state = apply_obfuscation_level_1(state)
    state = verify_sabotage(state)
    state = add_misleading_comments(state)
    return state

