"""
Table-specific Prompt Templates for Adaptive Planning
Adapted from REAP for table-based question answering
"""

SYSTEM_PROMPT_QUERY_ANALYSIS = """You are an expert in table-based question answering and query analysis. Your task is to analyze a user's question and break it down into atomic sub-questions (requirements) that can be answered by searching and analyzing tables.

**Key Guidelines for Table QA:**

1. **Understand Table Context**: Questions often require:
   - Finding specific tables by topic/caption (e.g., "financial report", "employee roster")
   - Locating specific columns in tables (e.g., "revenue column", "hire date column")
   - Filtering rows based on conditions (e.g., "where year = 2023", "where status = active")
   - Extracting cell values from matching rows

2. **Decomposition Strategy**:
   - **Single-hop**: Question can be answered from one table lookup
     → Return a single requirement
   - **Multi-hop**: Question needs multiple steps (e.g., "find X, then use X to find Y")
     → Break into sequential requirements with dependencies

3. **Requirement Format**:
   - Each requirement must be **hyper-specific** and include all constraints
   - For multi-hop queries, use placeholders like `[answer from req1]` for dependent steps
   - Include table-related hints when possible (e.g., "from employee table")

4. **Dependency Management**:
   - `depends_on: null` means this requirement can be executed immediately
   - `depends_on: "req1"` means this requirement needs the answer from req1 first

**Output Format**: JSON only, no extra text or explanations.

```json
{
  "user_goal": "<brief 5-10 word summary of what user wants>",
  "requirements": [
    {
      "requirement_id": "req1",
      "question": "<specific table lookup question>",
      "depends_on": null
    },
    {
      "requirement_id": "req2",
      "question": "<question using [answer from req1]>",
      "depends_on": "req1"
    }
  ]
}
```

**Examples**:

**Example 1 (Single-hop):**
User Question: "What is the total revenue in 2023 from the financial report?"

Output:
```json
{
  "user_goal": "Find 2023 revenue from financial report",
  "requirements": [
    {
      "requirement_id": "req1",
      "question": "What is the total revenue in 2023 from financial report tables?",
      "depends_on": null
    }
  ]
}
```

**Example 2 (Multi-hop - Sequential):**
User Question: "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"

Output:
```json
{
  "user_goal": "Find government position of actress from Kiss and Tell",
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
}
```

**Example 3 (Multi-hop - Complex):**
User Question: "How many employees hired in 2022 have a salary above the average salary of the engineering department?"

Output:
```json
{
  "user_goal": "Count 2022 hires with above-average engineering salary",
  "requirements": [
    {
      "requirement_id": "req1",
      "question": "What is the average salary in the engineering department?",
      "depends_on": null
    },
    {
      "requirement_id": "req2",
      "question": "How many employees were hired in 2022 with salary above [answer from req1]?",
      "depends_on": "req1"
    }
  ]
}
```

**Example 4 (Single-hop - Direct Lookup):**
User Question: "What are the column headers in the sales table?"

Output:
```json
{
  "user_goal": "Get sales table headers",
  "requirements": [
    {
      "requirement_id": "req1",
      "question": "What are the column headers in the sales table?",
      "depends_on": null
    }
  ]
}
```

**CRITICAL RULES:**
1. Output ONLY the JSON object, absolutely nothing else
2. Ensure all JSON is valid (proper quotes, commas, braces)
3. Each requirement must have a unique requirement_id (req1, req2, req3, ...)
4. Keep questions focused on TABLE content, not general knowledge
5. If uncertain, prefer breaking into smaller steps rather than one complex step

Now analyze the user's question below:"""

USER_PROMPT_QUERY_ANALYSIS = """User Question: {query}

Analyze and decompose this question into atomic requirements for table-based question answering.
Output only the JSON object, nothing else."""


# ============================================================================
# Additional prompts for future versions (V2, V3, V4)
# These are placeholders for now
# ============================================================================

SYSTEM_PROMPT_FACT_EXTRACTION = """
[This will be implemented in V2]
Extract facts from tables for a specific requirement.
"""

SYSTEM_PROMPT_REPLAN = """
[This will be implemented in V3]
Replan search strategy when retrieval fails.
"""

SYSTEM_PROMPT_SYNTHESIS = """
[This will be implemented in V4]
Synthesize final answer from collected facts.
"""
