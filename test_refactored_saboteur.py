"""
Quick unit test for the refactored two-phase bug injection system.
Tests _generate_tests_for_function, _inject_bug_into_function, and _sabotage_one_helper.
"""

import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace before running

from langchain_openai import ChatOpenAI
from architect.saboteur import _generate_tests_for_function, _inject_bug_into_function, _execute_tests_on_source, _sabotage_one_helper

# Simple test function
TEST_FUNCTION = """def power_of_three(n):
    \"\"\"Return n to the power of 3.\"\"\"
    result = 1
    for i in range(n):
        result = result * 3
    return result
"""

TEST_MODULE = """def power_of_three(n):
    \"\"\"Return n to the power of 3.\"\"\"
    result = 1
    for i in range(n):
        result = result * 3
    return result

def multiply(a, b):
    return a * b
"""

def test_generate_tests():
    """Test Phase 1: Generate tests for working code."""
    print("\n" + "="*60)
    print("TEST 1: _generate_tests_for_function()")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    test_data = _generate_tests_for_function(
        func_source=TEST_FUNCTION,
        func_name="power_of_three",
        surface_func_name="power_of_three",
        surface_source=TEST_FUNCTION,
        llm=llm,
        indirect_mode=False,
        debug_mode=True
    )
    
    if test_data:
        print("\n✅ SUCCESS: Generated tests")
        print(f"Public tests: {test_data.get('test_cases_public', [])}")
        print(f"Secret tests: {test_data.get('test_cases_secret', [])}")
        return test_data
    else:
        print("\n❌ FAILED: Could not generate tests")
        return None


def test_inject_bug():
    """Test Phase 2: Inject bug into function."""
    print("\n" + "="*60)
    print("TEST 2: _inject_bug_into_function()")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    buggy_data = _inject_bug_into_function(
        func_source=TEST_FUNCTION,
        func_name="power_of_three",
        llm=llm,
        attempted_bugs=[],
        debug_mode=True
    )
    
    if buggy_data:
        print("\n✅ SUCCESS: Injected bug")
        print(f"Bug description: {buggy_data.get('bug_description', 'N/A')}")
        print(f"Buggy code:\n{buggy_data.get('sabotaged_function_code', '')}")
        return buggy_data
    else:
        print("\n❌ FAILED: Could not inject bug")
        return None


def test_execute_tests():
    """Test Phase 3: Execute tests on source."""
    print("\n" + "="*60)
    print("TEST 3: _execute_tests_on_source()")
    print("="*60)
    
    # First generate tests
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    test_data = _generate_tests_for_function(
        func_source=TEST_FUNCTION,
        func_name="power_of_three",
        surface_func_name="power_of_three",
        surface_source=TEST_FUNCTION,
        llm=llm,
        indirect_mode=False,
        debug_mode=False
    )
    
    if not test_data:
        print("❌ Could not generate tests")
        return False
    
    # Execute on original code
    print("\nExecuting tests on ORIGINAL code...")
    original_results = _execute_tests_on_source(
        TEST_MODULE,
        "power_of_three",
        test_data,
        debug_mode=True
    )
    
    # Inject bug
    buggy_data = _inject_bug_into_function(
        func_source=TEST_FUNCTION,
        func_name="power_of_three",
        llm=llm,
        attempted_bugs=[],
        debug_mode=False
    )
    
    if not buggy_data:
        print("❌ Could not inject bug")
        return False
    
    # Create buggy module
    buggy_module = TEST_MODULE.replace(TEST_FUNCTION, buggy_data["sabotaged_function_code"])
    
    # Execute on buggy code
    print("\nExecuting tests on BUGGY code...")
    buggy_results = _execute_tests_on_source(
        buggy_module,
        "power_of_three",
        test_data,
        debug_mode=True
    )
    
    # Count differences
    differences = 0
    for test_args_str in original_results:
        if test_args_str not in buggy_results:
            continue
        
        orig_success, orig_result = original_results[test_args_str]
        buggy_success, buggy_result = buggy_results[test_args_str]
        
        if orig_success != buggy_success or orig_result != buggy_result:
            differences += 1
            print(f"\n✓ DIFFERENCE: {test_args_str}")
            print(f"  Original: {orig_result}")
            print(f"  Buggy:    {buggy_result}")
    
    print(f"\n{'✅' if differences >= 4 else '❌'} Result: {differences}/6 tests caught the bug")
    return differences >= 4


def test_full_pipeline():
    """Test the complete _sabotage_one_helper function."""
    print("\n" + "="*60)
    print("TEST 4: _sabotage_one_helper() - FULL PIPELINE")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    buggy_source, data = _sabotage_one_helper(
        bug_func_name="power_of_three",
        current_source=TEST_MODULE,
        surface_func_name="power_of_three",
        surface_source=TEST_FUNCTION,
        instructions="Inject a simple bug into the function.",
        llm=llm,
        indirect_mode=False,
        debug_mode=True
    )
    
    if buggy_source and data:
        print("\n" + "="*60)
        print("✅ FULL PIPELINE SUCCESS!")
        print("="*60)
        print(f"Bug description: {data.get('bug_description', 'N/A')}")
        print(f"Tests catching bug: {data.get('_debug_tests_catching_bug', 'N/A')}")
        print(f"\nPublic tests: {len(data.get('test_cases_public', []))}")
        print(f"Secret tests: {len(data.get('test_cases_secret', []))}")
        return True
    else:
        print("\n❌ FULL PIPELINE FAILED")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TWO-PHASE BUG INJECTION SYSTEM - UNIT TESTS")
    print("="*60)
    
    # Check API key
    if "your-api-key-here" in os.environ.get("OPENAI_API_KEY", ""):
        print("\n⚠️  WARNING: Please set your OpenAI API key in this file before running!")
        print("Edit line 7 to add your API key.\n")
        exit(1)
    
    # Run tests
    test_generate_tests()
    test_inject_bug()
    test_execute_tests()
    test_full_pipeline()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60 + "\n")
