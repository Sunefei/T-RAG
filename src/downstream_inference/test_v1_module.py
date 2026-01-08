#!/usr/bin/env python3
"""
Quick test script for V1 adaptive modules
Tests decomposer without needing full dataset or API calls
"""

import json
import sys
from adaptive_modules.query_decomposer import (
    analyze_and_decompose_query,
    is_multi_hop_query,
    format_decomposition_for_display,
    _parse_json_from_response,
    _validate_decomposition_result
)


def mock_llm_single_hop(system_prompt, user_prompt):
    """Mock LLM for single-hop query"""
    return """{
        "user_goal": "Find 2023 revenue from financial report",
        "requirements": [
            {
                "requirement_id": "req1",
                "question": "What is the total revenue in 2023 from financial report tables?",
                "depends_on": null
            }
        ]
    }"""


def mock_llm_multi_hop(system_prompt, user_prompt):
    """Mock LLM for multi-hop query"""
    return """{
        "user_goal": "Find government position of actress",
        "requirements": [
            {
                "requirement_id": "req1",
                "question": "Who portrayed Corliss Archer in the film Kiss and Tell?",
                "depends_on": null
            },
            {
                "requirement_id": "req2",
                "question": "What government position was held by [answer from req1]?",
                "depends_on": "req1"
            }
        ]
    }"""


def mock_llm_with_markdown(system_prompt, user_prompt):
    """Mock LLM that returns JSON in markdown"""
    return """Here's the decomposition:

```json
{
    "user_goal": "Test markdown parsing",
    "requirements": [
        {
            "requirement_id": "req1",
            "question": "Test question?",
            "depends_on": null
        }
    ]
}
```

That's the result!"""


def test_single_hop():
    """Test single-hop query decomposition"""
    print("\n" + "="*60)
    print("TEST 1: Single-hop Query")
    print("="*60)

    query = "What is the total revenue in 2023 from the financial report?"
    result = analyze_and_decompose_query(
        query,
        llm_call_func=mock_llm_single_hop,
        verbose=True
    )

    assert len(result["requirements"]) == 1, "Should have 1 requirement"
    assert not is_multi_hop_query(result), "Should be single-hop"
    assert result["requirements"][0]["depends_on"] is None, "Should have no dependency"

    print("✓ Test 1 passed!")
    return True


def test_multi_hop():
    """Test multi-hop query decomposition"""
    print("\n" + "="*60)
    print("TEST 2: Multi-hop Query")
    print("="*60)

    query = "What government position was held by the woman who portrayed Corliss Archer?"
    result = analyze_and_decompose_query(
        query,
        llm_call_func=mock_llm_multi_hop,
        verbose=True
    )

    assert len(result["requirements"]) == 2, "Should have 2 requirements"
    assert is_multi_hop_query(result), "Should be multi-hop"
    assert result["requirements"][0]["depends_on"] is None, "First should have no dependency"
    assert result["requirements"][1]["depends_on"] == "req1", "Second should depend on req1"

    print("✓ Test 2 passed!")
    return True


def test_json_parsing():
    """Test JSON parsing from various formats"""
    print("\n" + "="*60)
    print("TEST 3: JSON Parsing (Markdown)")
    print("="*60)

    query = "Test query"
    result = analyze_and_decompose_query(
        query,
        llm_call_func=mock_llm_with_markdown,
        verbose=True
    )

    assert "requirements" in result, "Should parse JSON from markdown"
    print("✓ Test 3 passed!")
    return True


def test_validation():
    """Test validation logic"""
    print("\n" + "="*60)
    print("TEST 4: Validation")
    print("="*60)

    # Valid decomposition
    valid = {
        "user_goal": "Test",
        "requirements": [
            {
                "requirement_id": "req1",
                "question": "Test question?",
                "depends_on": None
            }
        ]
    }

    try:
        _validate_decomposition_result(valid)
        print("✓ Valid decomposition passed")
    except Exception as e:
        print(f"✗ Valid decomposition failed: {e}")
        return False

    # Invalid: missing requirements
    invalid1 = {
        "user_goal": "Test"
    }

    try:
        _validate_decomposition_result(invalid1)
        print("✗ Should have rejected missing requirements")
        return False
    except ValueError:
        print("✓ Correctly rejected missing requirements")

    # Invalid: empty requirements
    invalid2 = {
        "user_goal": "Test",
        "requirements": []
    }

    try:
        _validate_decomposition_result(invalid2)
        print("✗ Should have rejected empty requirements")
        return False
    except ValueError:
        print("✓ Correctly rejected empty requirements")

    print("✓ Test 4 passed!")
    return True


def test_format_display():
    """Test display formatting"""
    print("\n" + "="*60)
    print("TEST 5: Display Formatting")
    print("="*60)

    decomposition = {
        "user_goal": "Find position of actress",
        "requirements": [
            {
                "requirement_id": "req1",
                "question": "Who was the actress?",
                "depends_on": None
            },
            {
                "requirement_id": "req2",
                "question": "What position did she hold?",
                "depends_on": "req1"
            }
        ]
    }

    formatted = format_decomposition_for_display(decomposition)
    print("\nFormatted output:")
    print(formatted)

    assert "req1" in formatted, "Should include req1"
    assert "req2" in formatted, "Should include req2"
    assert "[depends on req1]" in formatted, "Should show dependency"

    print("\n✓ Test 5 passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("T-RAG V1 MODULE TESTS")
    print("="*70)

    tests = [
        ("Single-hop decomposition", test_single_hop),
        ("Multi-hop decomposition", test_multi_hop),
        ("JSON parsing", test_json_parsing),
        ("Validation", test_validation),
        ("Display formatting", test_format_display)
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {name} failed")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} crashed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests passed! V1 module is working correctly.")
        print("\nNext step: Run with real data")
        print("  python call_llm_v1.py --dataset sqa --topk 50 --model gpt-4o-mini --testing_num 10 --use_decomposition")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix before running with real data.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
