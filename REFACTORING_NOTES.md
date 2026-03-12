# Two-Phase Bug Injection Refactoring

## Summary
Complete architectural refactor of the bug injection system to guarantee test quality.

**Date**: Current session  
**Status**: ✅ Complete - Ready for testing

---

## The Problem (OLD APPROACH)

### What Was Wrong
Single GPT call asking it to:
1. Inject a bug into a function
2. Generate tests that catch the bug

### Why It Failed
- GPT created bugs independently of tests
- Tests were too generic and passed even with bugs
- Result: "detected by 0 tests" repeatedly
- No validation that tests actually expose the bug

### Example Failure
```
Bug: range(n) → range(n-1)
Test: func([1,2,3,4,5])
- Original: [1,2,3,4,5]
- Buggy:    [1,2,3,4]
BUT: Test expected generic "list" result, so both passed ❌
```

---

## The Solution (NEW APPROACH)

### Revolutionary Idea
**Generate tests FIRST on working code, THEN inject bug**

This guarantees:
1. Tests are valid (run on working code)
2. Tests have concrete expected outputs
3. ANY bug changes outputs → guaranteed detection

### Three-Phase Workflow

#### PHASE 1: Generate Tests on Clean Code
```python
tests = _generate_tests_for_function(working_code)
# Example: 3 public + 3 secret tests
```

**Result**: 6 tests with known-good expected outputs

#### PHASE 2: Execute Tests on Original
```python
original_results = _execute_tests_on_source(original_code, tests)
# Example: {
#   "(5,)": (True, "125"),
#   "(3,)": (True, "27"),
#   ...
# }
```

**Result**: Baseline outputs for all 6 tests

#### PHASE 3: Inject Bug → Validate Detection
```python
buggy_code = _inject_bug_into_function(original_code)
buggy_results = _execute_tests_on_source(buggy_code, tests)

differences = count_differences(original_results, buggy_results)
if differences >= 4:  # At least 4/6 tests must differ
    SUCCESS! ✓
```

**Result**: Only accept bugs that 4+ tests reliably catch

---

## Technical Changes

### New Functions

#### `_generate_tests_for_function()`
- **Purpose**: Generate 3+3 tests for WORKING code
- **Input**: Clean function source  
- **Output**: `{"test_cases_public": [...], "test_cases_secret": [...]}`
- **Key**: No bug injection in this phase

#### `_inject_bug_into_function()`
- **Purpose**: Inject ONE simple bug
- **Input**: Clean function source, list of attempted bugs
- **Output**: `{"sabotaged_function_code": "...", "bug_description": "..."}`
- **Key**: Avoids repeating failed bugs via `attempted_bugs` list

#### `_execute_tests_on_source()`
- **Purpose**: Run all tests on given source
- **Input**: Source code, tests dict
- **Output**: `{test_args_str: (success, result), ...}`
- **Key**: Captures both success status AND result value

### Refactored Functions

#### `_sabotage_one_helper()` - COMPLETE REWRITE
**Before**: 
- Single GPT call: bug + tests together
- ~150 lines of validation
- No output comparison

**After**:
- Phase 1: Generate tests on clean code
- Phase 2: Execute → save outputs  
- Phase 3: Inject bug → execute → compare
- Validation: 4+ tests must show different outputs
- ~180 lines but much more robust

---

## Validation Strategy

### Acceptance Criteria
For a bug to be accepted:
1. ✅ Syntax valid (parses without errors)
2. ✅ Function name preserved
3. ✅ No revealing comments
4. ✅ **NEW**: At least 4/6 tests produce DIFFERENT outputs

### What "Different" Means
```python
orig_success, orig_result = original_results[test]
buggy_success, buggy_result = buggy_results[test]

different = (orig_success != buggy_success) OR (orig_result != buggy_result)
```

Examples of differences:
- Original succeeds, buggy crashes → DIFFERENT ✓
- Both succeed, different values → DIFFERENT ✓
- Both succeed, same value → NOT DIFFERENT ✗

---

## Parameter Simplifications

### Test Count
- **Before**: 5 public + 5 secret = 10 tests per bug
- **After**: 3 public + 3 secret = 6 tests per bug
- **Rationale**: Easier for GPT, still sufficient coverage

### Bug Distribution  
- **Before**: N bugs → 2 functions → [3,2] split
- **After**: N bugs → N functions → 1 bug each
- **Rationale**: Simpler validation, clearer test ownership

---

## Expected Benefits

### Cost Savings
- Fewer retries needed (tests guaranteed to work)
- Fewer GPT calls (separate prompts are clearer)

### Quality Improvement
- **Before**: ~30% bugs detected by 0 tests
- **After (expected)**: 100% bugs detected by 4+ tests

### Debug Visibility
New metadata in results:
```python
{
    "_debug_tests_catching_bug": "5/6",
    ...
}
```

---

## Migration Notes

### Breaking Changes
**None** - `_sabotage_one_helper()` signature unchanged

All existing calls remain compatible:
```python
_sabotage_one_helper(
    bug_func_name, current_source,
    surface_func_name, surface_source,
    instructions, llm, indirect_mode,
    call_chain=...,
    previous_bugs=...,
    target_nesting=...,
    debug_mode=...
)
```

### Unused Parameters
- `call_chain`: Not used in new implementation (reserved for future)
- `previous_bugs`: Not used (replaced by internal `attempted_bugs` list)
- `target_nesting`: Not used in new implementation

These remain in signature for API compatibility.

---

## Testing Plan

### Quick Test
```bash
python main.py --repo Ugly_Legacy_code --num-bugs 2 --debug
```

**Expected Output**:
```
[saboteur] PHASE 1: Generating tests for 'func1' (working code)...
[saboteur] PHASE 1 SUCCESS: Generated 3 public + 3 secret tests
[saboteur] Executing tests on ORIGINAL code...
[saboteur]   Test (5,) -> OK: 125
[saboteur]   Test (3,) -> OK: 27
...
[saboteur] PHASE 2: Bug injection attempt 1/3...
[bug_inject] ✓ Bug injected: Changed range(n) to range(n-1)
[saboteur] Executing tests on BUGGY code...
[saboteur]   Test (5,) -> OK: 24
[saboteur]   DIFFERENCE: (5,)
[saboteur]     Original: 125
[saboteur]     Buggy:    24
...
[saboteur] Bug caught by 5/6 tests
[saboteur] SUCCESS! Bug reliably detected by 5/6 tests
```

### Metrics to Watch
- **Bug acceptance rate**: Should be higher (~80%+ vs ~40% before)
- **Tests catching bugs**: Should always be 4+/6 (100% vs ~30% before)
- **GPT retries**: Should be lower (clearer prompts)

---

## Files Modified

### architect/saboteur.py
- **Lines 595-673**: Added `_generate_tests_for_function()`
- **Lines 673-748**: Added `_inject_bug_into_function()`
- **Lines 779-803**: Added `_execute_tests_on_source()`
- **Lines 821-1000**: Complete rewrite of `_sabotage_one_helper()`

**Total Changes**: ~400 lines added/modified

---

## Rollback Plan

If issues arise, revert commit:
```bash
git log --oneline  # Find commit before refactor
git revert <commit-hash>
```

Old implementation preserved in git history.

---

## Next Steps

1. ✅ Code refactor complete
2. ⏭️ Test with `--num-bugs 2 --debug`
3. ⏭️ Verify "X/6 tests caught bug" in logs
4. ⏭️ Run full test with 5 bugs
5. ⏭️ Compare metrics vs old approach

---

## Credits

**Concept**: User's revolutionary insight - "generate tests first, then inject bug"  
**Implementation**: Complete architectural refactor of saboteur.py  
**Philosophy**: Guarantee test quality through execution validation, not hope
