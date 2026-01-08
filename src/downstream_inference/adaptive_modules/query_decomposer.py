"""
Query Decomposer - Breaks complex table queries into atomic sub-questions
Adapted from REAP for table-based QA scenarios
"""

import json
import re
from typing import Dict, List, Callable, Optional
from .prompts import SYSTEM_PROMPT_QUERY_ANALYSIS, USER_PROMPT_QUERY_ANALYSIS


def analyze_and_decompose_query(
    query: str,
    llm_call_func: Callable[[str, str], str],
    verbose: bool = False
) -> Dict:
    """
    Decompose a complex query into atomic sub-questions (requirements).

    This is the core function for V1. It analyzes the user's question and
    breaks it down into a sequence of table lookup operations, identifying
    dependencies between steps for multi-hop reasoning.

    Args:
        query: The original user question (e.g., "What position was held by
               the actress who portrayed Corliss Archer in Kiss and Tell?")
        llm_call_func: Function to call LLM, signature: (system_prompt, user_prompt) -> response
        verbose: Whether to print detailed logs

    Returns:
        {
            "user_goal": str,  # Brief summary of the goal
            "requirements": [
                {
                    "requirement_id": "req1",
                    "question": "Who portrayed Corliss Archer in Kiss and Tell?",
                    "depends_on": null
                },
                {
                    "requirement_id": "req2",
                    "question": "What position was held by [answer from req1]?",
                    "depends_on": "req1"
                }
            ]
        }

    Example:
        >>> result = analyze_and_decompose_query(
        ...     "What position was held by the actress in Kiss and Tell?",
        ...     llm_call_func=call_gpt
        ... )
        >>> print(len(result["requirements"]))
        2
        >>> print(result["requirements"][0]["question"])
        "Who was the actress in Kiss and Tell?"
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[V1 Query Decomposition] Analyzing query:")
        print(f"  Query: {query}")
        print(f"{'='*60}")

    # Prepare prompts
    system_prompt = SYSTEM_PROMPT_QUERY_ANALYSIS
    user_prompt = USER_PROMPT_QUERY_ANALYSIS.format(query=query)

    try:
        # Call LLM
        response = llm_call_func(system_prompt, user_prompt)

        if verbose:
            print(f"\n[LLM Response (first 500 chars)]:")
            print(response[:500])
            print("...\n")

        # Parse JSON from response
        result = _parse_json_from_response(response)

        # Validate structure
        _validate_decomposition_result(result)

        if verbose:
            print(f"\n[Decomposition Result]:")
            print(f"  User Goal: {result['user_goal']}")
            print(f"  Number of Requirements: {len(result['requirements'])}")
            for req in result['requirements']:
                depends_on_str = f" (depends on {req['depends_on']})" if req['depends_on'] else ""
                print(f"    - {req['requirement_id']}: {req['question']}{depends_on_str}")
            print(f"{'='*60}\n")

        return result

    except json.JSONDecodeError as e:
        if verbose:
            print(f"\n[ERROR] JSON parsing failed: {e}")
            print(f"Response was: {response[:200]}...")

        # Fallback: Return single requirement with original query
        return _create_fallback_decomposition(query, verbose)

    except Exception as e:
        if verbose:
            print(f"\n[ERROR] Decomposition failed: {e}")

        # Fallback: Return single requirement
        return _create_fallback_decomposition(query, verbose)


def _parse_json_from_response(response: str) -> Dict:
    """
    Extract and parse JSON from LLM response.

    LLMs sometimes wrap JSON in markdown code blocks or add extra text.
    This function handles various formats robustly.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    # Strategy 1: Try direct parsing (if LLM followed instructions)
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block
    # Pattern: ```json\n{...}\n```
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract first JSON object found
    # Pattern: {...}
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: More aggressive - find { to last }
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(response[start_idx:end_idx+1])
        except json.JSONDecodeError:
            pass

    # All strategies failed
    raise json.JSONDecodeError(
        "Could not find valid JSON in response",
        response,
        0
    )


def _validate_decomposition_result(result: Dict) -> None:
    """
    Validate that decomposition result has correct structure.

    Args:
        result: Parsed decomposition result

    Raises:
        ValueError: If structure is invalid
    """
    # Check top-level keys
    if not isinstance(result, dict):
        raise ValueError(f"Result must be a dictionary, got {type(result)}")

    if "requirements" not in result:
        raise ValueError("Result missing 'requirements' key")

    requirements = result["requirements"]

    if not isinstance(requirements, list):
        raise ValueError(f"'requirements' must be a list, got {type(requirements)}")

    if len(requirements) == 0:
        raise ValueError("'requirements' list is empty")

    # Check each requirement
    required_keys = {"requirement_id", "question", "depends_on"}

    for i, req in enumerate(requirements):
        if not isinstance(req, dict):
            raise ValueError(f"Requirement {i} must be a dictionary, got {type(req)}")

        # Check required keys
        missing_keys = required_keys - set(req.keys())
        if missing_keys:
            raise ValueError(f"Requirement {i} missing keys: {missing_keys}")

        # Validate requirement_id format
        req_id = req["requirement_id"]
        if not isinstance(req_id, str) or not req_id.startswith("req"):
            raise ValueError(f"Invalid requirement_id: {req_id}")

        # Validate question
        if not isinstance(req["question"], str) or len(req["question"].strip()) == 0:
            raise ValueError(f"Requirement {req_id} has empty question")

        # Validate depends_on (can be None or a string)
        depends_on = req["depends_on"]
        if depends_on is not None and not isinstance(depends_on, str):
            raise ValueError(f"Requirement {req_id} has invalid depends_on: {depends_on}")


def _create_fallback_decomposition(query: str, verbose: bool = False) -> Dict:
    """
    Create a simple single-requirement decomposition when parsing fails.

    Args:
        query: Original user query
        verbose: Whether to print warning

    Returns:
        Simple decomposition with one requirement
    """
    if verbose:
        print(f"\n[WARNING] Using fallback decomposition (single requirement)")

    return {
        "user_goal": query[:50] + "..." if len(query) > 50 else query,
        "requirements": [
            {
                "requirement_id": "req1",
                "question": query,
                "depends_on": None
            }
        ]
    }


def is_multi_hop_query(decomposition: Dict) -> bool:
    """
    Determine if a decomposition represents a multi-hop query.

    A multi-hop query has:
    - More than one requirement, OR
    - At least one requirement with dependencies

    Args:
        decomposition: Result from analyze_and_decompose_query

    Returns:
        True if multi-hop, False if single-hop

    Example:
        >>> decomp = {
        ...     "requirements": [
        ...         {"requirement_id": "req1", "question": "...", "depends_on": None},
        ...         {"requirement_id": "req2", "question": "...", "depends_on": "req1"}
        ...     ]
        ... }
        >>> is_multi_hop_query(decomp)
        True
    """
    requirements = decomposition.get("requirements", [])

    # Multiple requirements usually means multi-hop
    if len(requirements) > 1:
        return True

    # Check if any requirement has dependencies
    # (This handles edge cases where there's only one req but it was split from another)
    for req in requirements:
        if req.get("depends_on") is not None:
            return True

    return False


def get_executable_requirements(
    requirements: List[Dict],
    completed_requirement_ids: set
) -> List[Dict]:
    """
    Get requirements that can be executed now (dependencies satisfied).

    This is a utility function for V2/V3/V4 when we implement sequential execution.
    Included in V1 for completeness.

    Args:
        requirements: List of all requirements
        completed_requirement_ids: Set of requirement IDs already completed

    Returns:
        List of requirements ready to execute

    Example:
        >>> reqs = [
        ...     {"requirement_id": "req1", "depends_on": None},
        ...     {"requirement_id": "req2", "depends_on": "req1"},
        ...     {"requirement_id": "req3", "depends_on": None}
        ... ]
        >>> get_executable_requirements(reqs, set())
        [req1, req3]  # req2 not ready yet
        >>> get_executable_requirements(reqs, {"req1"})
        [req2, req3]  # Now req2 is ready
    """
    executable = []

    for req in requirements:
        req_id = req["requirement_id"]

        # Skip if already completed
        if req_id in completed_requirement_ids:
            continue

        # Check if dependencies are satisfied
        depends_on = req.get("depends_on")

        if depends_on is None:
            # No dependencies, can execute
            executable.append(req)
        elif depends_on in completed_requirement_ids:
            # Dependency satisfied, can execute
            executable.append(req)
        # else: dependency not satisfied yet, skip

    return executable


def format_decomposition_for_display(decomposition: Dict) -> str:
    """
    Format decomposition result as a human-readable string.

    Useful for logging and debugging.

    Args:
        decomposition: Result from analyze_and_decompose_query

    Returns:
        Formatted string representation
    """
    lines = []
    lines.append(f"User Goal: {decomposition.get('user_goal', 'N/A')}")
    lines.append(f"Requirements ({len(decomposition.get('requirements', []))}):")

    for req in decomposition.get('requirements', []):
        req_id = req.get('requirement_id', 'unknown')
        question = req.get('question', 'N/A')
        depends_on = req.get('depends_on')

        dep_str = f" [depends on {depends_on}]" if depends_on else ""
        lines.append(f"  {req_id}: {question}{dep_str}")

    return "\n".join(lines)


# ============================================================================
# Test/Debug Utilities
# ============================================================================

def test_decomposer():
    """
    Simple test function to verify decomposer works.
    Can be run with: python -m adaptive_modules.query_decomposer
    """
    print("Testing Query Decomposer...")

    # Mock LLM function for testing
    def mock_llm(system, user):
        return """{
            "user_goal": "Test query decomposition",
            "requirements": [
                {
                    "requirement_id": "req1",
                    "question": "What is the revenue in 2023?",
                    "depends_on": null
                }
            ]
        }"""

    # Test single-hop
    result = analyze_and_decompose_query(
        "What is the revenue in 2023?",
        llm_call_func=mock_llm,
        verbose=True
    )

    assert len(result["requirements"]) == 1
    assert not is_multi_hop_query(result)
    print("✓ Single-hop test passed")

    # Test multi-hop
    def mock_llm_multihop(system, user):
        return """{
            "user_goal": "Find position of actress",
            "requirements": [
                {
                    "requirement_id": "req1",
                    "question": "Who was the actress?",
                    "depends_on": null
                },
                {
                    "requirement_id": "req2",
                    "question": "What position did [answer from req1] hold?",
                    "depends_on": "req1"
                }
            ]
        }"""

    result = analyze_and_decompose_query(
        "What position was held by the actress?",
        llm_call_func=mock_llm_multihop,
        verbose=True
    )

    assert len(result["requirements"]) == 2
    assert is_multi_hop_query(result)
    print("✓ Multi-hop test passed")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_decomposer()
