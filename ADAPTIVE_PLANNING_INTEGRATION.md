# T-RAG + REAP Adaptive Planning 增量式集成指南

## 📋 目录
1. [T-RAG 运行指南](#1-t-rag-运行指南)
2. [增量式开发计划](#2-增量式开发计划)
3. [版本详细设计](#3-版本详细设计)
4. [评估对比方法](#4-评估对比方法)

---

## 1. T-RAG 运行指南

### 1.1 环境准备

```bash
# 激活环境
conda activate trag

# 验证依赖
python -c "import torch; import transformers; import sentence_transformers; print('Environment OK')"
```

### 1.2 数据准备（首次运行）

选择数据集：**SQA (Sequential Question Answering)**
- 原因：多跳问答，最适合测试 adaptive planning
- 规模：适中，适合快速迭代
- 问题类型：需要多步推理

```bash
cd /Users/sunyifei/Documents/GitHub/T-RAG/src/table2graph

# 准备数据（首次运行）
bash scripts/prepare_data.sh
# 这会下载并处理 MultiTableQA 数据集，包括 SQA

# 验证数据
ls -lh data/sqa/
# 应该看到：
# - sqa_source_tables.jsonl (源表格)
# - sqa_sub_tables.jsonl (分解后的子表)
# - sqa_table_schema.jsonl (表格schema)
# - sqa_example_query.jsonl (示例查询)
# - sqa_test_100.jsonl (测试集，100个样本)
```

### 1.3 完整运行流程（基线 T-RAG）

#### 步骤1：表格聚类（Stage 1&2）

```bash
cd /Users/sunyifei/Documents/GitHub/T-RAG/src/table2graph

# 运行聚类
bash scripts/table_cluster_run.sh

# 参数说明（在 table_cluster_run.sh 中配置）：
# - dataset: sqa
# - n_clusters: 3（聚类数）
# - k: 50（每聚类的典型节点数）
# - embedding_method: contriever

# 输出：
# data/sqa/sqa_clustered_tables_k50_n3_contriever.jsonl

# 查看日志
tail -f logs/sqa/sqa_cluster_k50_n3_contriever.log
```

#### 步骤2：子图检索（Stage 3）

```bash
cd /Users/sunyifei/Documents/GitHub/T-RAG/src/table2graph

# 运行PageRank检索
python scripts/subgraph_retrieve_run.py

# 参数说明（在 subgraph_retrieve_run.py 中配置）：
# - DATASET: sqa
# - testing_num: 100（测试样本数）
# - top_k: 50（最终返回的表格数）
# - cluster_embedding_method: contriever
# - table_to_graph_embedding_method: sentencetransformer

# 输出：
# data/sqa/sqa_retrieved_tables_schema_100_50_contriever.jsonl

# 查看日志
tail -f logs/sqa/sqa_subgraph_testingnum100_topK50_sentencetransformer.log
```

#### 步骤3：LLM 推理（Stage 4）

**注意：需要配置 API Key**

```bash
cd /Users/sunyifei/Documents/GitHub/T-RAG/src/downstream_inference

# 配置 key.json
cat > key.json << EOF
{
    "openai": "YOUR_OPENAI_API_KEY",
    "claude": "YOUR_CLAUDE_API_KEY"
}
EOF

# 运行推理
python call_llm.py \
    --dataset sqa \
    --topk 50 \
    --mode API \
    --model gpt-4o-mini \
    --testing_num 100 \
    --embedding_method contriever

# 输出：
# output/sqa/gpt-4o-mini/output_100_50.jsonl
```

#### 步骤4：评估

```bash
cd /Users/sunyifei/Documents/GitHub/T-RAG/src/downstream_inference

# 运行评估
python evaluation.py \
    --dataset sqa \
    --model gpt-4o-mini \
    --topk 50 \
    --testing_num 100

# 输出：
# output/sqa/gpt-4o-mini/results_100_50.json
# 包含指标：exact_match, f1_score

# 查看结果
cat output/sqa/gpt-4o-mini/results_100_50.json
```

### 1.4 快速测试脚本（推荐）

为了快速验证，我们创建一个小规模测试：

```bash
# 修改 testing_num 为 10（在各个脚本中）
# 这样可以在几分钟内完成一次完整测试
```

---

## 2. 增量式开发计划

### 核心原则
✅ 每个版本都是**独立可运行**的
✅ 每个版本都能**生成评估指标**（EM/F1）
✅ 新功能都有**开关控制**，默认关闭
✅ 保留**原始 T-RAG 代码**，通过新文件扩展

### 版本路线图

```
版本0（基线）: 原始 T-RAG
    ↓
版本1: + 查询分解模块（可选开关：--use_decomposition）
    ↓
版本2: + 事实提取评估（可选开关：--use_fact_extraction）
    ↓
版本3: + 重规划能力（可选开关：--use_replan）
    ↓
版本4: + 完整 orchestrator（可选开关：--use_adaptive_rag）
    ↓
版本5: 性能优化（缓存、批处理、Prompt调优）
```

### 版本对比矩阵

| 版本 | 查询分解 | 事实提取 | 重规划 | 迭代循环 | 预期EM提升 | 开发时间 |
|------|---------|---------|--------|----------|-----------|----------|
| V0   | ❌      | ❌      | ❌     | ❌       | 基线      | 0天      |
| V1   | ✅      | ❌      | ❌     | ❌       | 0-1%      | 1天      |
| V2   | ✅      | ✅      | ❌     | ❌       | +1-2%     | 1天      |
| V3   | ✅      | ✅      | ✅     | ❌       | +2-3%     | 1.5天    |
| V4   | ✅      | ✅      | ✅     | ✅       | +3-5%     | 2天      |
| V5   | ✅      | ✅      | ✅     | ✅       | +4-6%     | 1天      |

---

## 3. 版本详细设计

### 版本0：基线 T-RAG（验证环境）

**目标**：建立可靠的基线指标

**任务清单**：
- [x] 运行完整的 T-RAG pipeline
- [x] 记录 baseline 指标（EM/F1）
- [x] 验证所有脚本可正常运行
- [x] 创建测试数据集（100样本 SQA）

**运行方法**：
```bash
# 按照 1.3 节的步骤运行
# 记录输出到 baseline_results.json
```

**预期输出**：
```json
{
  "dataset": "sqa",
  "model": "gpt-4o-mini",
  "testing_num": 100,
  "topk": 50,
  "exact_match": 0.XX,
  "f1_score": 0.XX,
  "avg_retrieval_time": XX.XX,
  "avg_inference_time": XX.XX
}
```

---

### 版本1：添加查询分解（可选开关）

**目标**：实现查询分解功能，但不改变推理流程

**新增文件**：
```
src/downstream_inference/
├── adaptive_modules/
│   ├── __init__.py
│   ├── query_decomposer.py       # 从 REAP 移植
│   └── prompts.py                 # 表格专用 prompts
└── call_llm_v1.py                 # 扩展版本（继承 call_llm.py）
```

**核心改动**：

**1. adaptive_modules/query_decomposer.py**
```python
"""
查询分解模块 - 从 REAP 移植并适配表格场景
"""
import json
import re
from .prompts import SYSTEM_PROMPT_QUERY_ANALYSIS, USER_PROMPT_QUERY_ANALYSIS

def analyze_and_decompose_query(query: str, llm_call_func) -> dict:
    """
    将复杂查询分解为原子级子问题

    Args:
        query: 原始用户查询
        llm_call_func: LLM调用函数（传入以保持兼容性）

    Returns:
        {
            "user_goal": str,
            "requirements": [
                {
                    "requirement_id": "req1",
                    "question": "子问题1",
                    "depends_on": null
                },
                {
                    "requirement_id": "req2",
                    "question": "子问题2（可能包含占位符）",
                    "depends_on": "req1"
                }
            ]
        }
    """
    system_prompt = SYSTEM_PROMPT_QUERY_ANALYSIS
    user_prompt = USER_PROMPT_QUERY_ANALYSIS.format(query=query)

    # 调用 LLM
    response = llm_call_func(system_prompt, user_prompt)

    # 解析 JSON
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        # 降级：如果分解失败，返回单一需求
        return {
            "user_goal": query,
            "requirements": [
                {
                    "requirement_id": "req1",
                    "question": query,
                    "depends_on": None
                }
            ]
        }

    result = json.loads(match.group(0))

    # 验证结构
    if "requirements" not in result:
        raise ValueError("Invalid decomposition result")

    return result


def is_multi_hop_query(decomposition: dict) -> bool:
    """
    判断是否为多跳查询
    """
    return len(decomposition["requirements"]) > 1
```

**2. adaptive_modules/prompts.py**
```python
"""
表格专用 Prompt 模板
"""

SYSTEM_PROMPT_QUERY_ANALYSIS = """
You are an expert in table-based question answering and query analysis. Your task is to analyze a user's question and break it down into atomic sub-questions (requirements) that can be answered by searching and analyzing tables.

**Key Guidelines for Table QA:**

1. **Understand Table Context**: Questions often require:
   - Finding specific tables by topic/caption
   - Locating specific columns in tables
   - Filtering rows based on conditions
   - Extracting cell values

2. **Decomposition Strategy**:
   - **Single-hop**: Question can be answered from one table lookup
     → Return a single requirement
   - **Multi-hop**: Question needs multiple steps
     → Break into sequential requirements with dependencies

3. **Requirement Format**:
   - Each requirement must be hyper-specific
   - Include all constraints from the original question
   - Use placeholders like [answer from req1] for dependent requirements

**Output Format**: JSON only, no extra text.

```json
{
  "user_goal": "<brief summary of what user wants>",
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

Example 1 (Single-hop):
Query: "What is the total revenue in 2023 from the financial report?"
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

Example 2 (Multi-hop):
Query: "What position was held by the actress in Kiss and Tell who was born in 1928?"
Output:
```json
{
  "user_goal": "Find government position of specific actress",
  "requirements": [
    {
      "requirement_id": "req1",
      "question": "Who was the actress in Kiss and Tell film who was born in 1928?",
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

CRITICAL: Output ONLY the JSON object, nothing else.
"""

USER_PROMPT_QUERY_ANALYSIS = """
User Question: {query}

Analyze and decompose this question into atomic requirements for table-based QA.
"""
```

**3. call_llm_v1.py** (只显示关键修改部分)
```python
"""
T-RAG with Query Decomposition (Version 1)
新增参数：--use_decomposition
"""
import argparse
from adaptive_modules.query_decomposer import analyze_and_decompose_query, is_multi_hop_query

# ... (保留原有的所有函数)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-RAG with Adaptive Planning V1")
    # ... 原有参数 ...
    parser.add_argument("--use_decomposition", action="store_true",
                        help="Enable query decomposition (V1 feature)")

    args = parser.parse_args()

    # ... 原有代码 ...

    # V1 新增逻辑
    decomposition_log = []  # 记录分解结果用于分析

    for i, line in enumerate(tqdm(retrieve_instances)):
        retrieve_instance = json.loads(line)
        query = retrieve_instance["query"]

        # V1: 可选的查询分解
        if args.use_decomposition:
            try:
                decomposition = analyze_and_decompose_query(
                    query,
                    llm_call_func=lambda sys, usr: call_openai_api(sys, usr, model)
                )

                # 记录分解结果
                decomposition_log.append({
                    "query": query,
                    "decomposition": decomposition,
                    "is_multi_hop": is_multi_hop_query(decomposition)
                })

                # V1 阶段：仅分解，不改变推理流程
                # 仍然使用原始 query 进行推理
                print(f"[V1] Decomposed into {len(decomposition['requirements'])} requirements")

            except Exception as e:
                print(f"[V1] Decomposition failed: {e}, falling back to original query")

        # 后续推理逻辑保持不变
        # ... (原有的 prompt 构建和 LLM 调用) ...

    # 保存分解日志
    if args.use_decomposition:
        decomposition_log_file = f"{output_dir}/decomposition_log_{testing_num}_{topk}.jsonl"
        with open(decomposition_log_file, "w") as f:
            for entry in decomposition_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[V1] Decomposition log saved to {decomposition_log_file}")
```

**运行方法**：
```bash
cd /Users/sunyifei/Documents/GitHub/T-RAG/src/downstream_inference

# 对比实验1：不使用分解（基线）
python call_llm_v1.py \
    --dataset sqa \
    --topk 50 \
    --mode API \
    --model gpt-4o-mini \
    --testing_num 100 \
    --embedding_method contriever

# 对比实验2：使用分解
python call_llm_v1.py \
    --dataset sqa \
    --topk 50 \
    --mode API \
    --model gpt-4o-mini \
    --testing_num 100 \
    --embedding_method contriever \
    --use_decomposition

# 评估（两次实验使用相同的评估脚本）
python evaluation.py --dataset sqa --model gpt-4o-mini --topk 50 --testing_num 100
```

**评估方法**：
```bash
# 对比两次运行的结果
# V1 阶段预期：EM/F1 基本相同（因为没有改变推理流程）
# 但是可以从 decomposition_log 中分析：
# - 有多少查询被识别为 multi-hop
# - 分解质量如何
```

---

### 版本2：添加事实提取评估

**目标**：在分解后的每个子问题上评估事实提取质量

**新增文件**：
```
src/downstream_inference/adaptive_modules/
├── fact_extractor.py              # 新增
└── call_llm_v2.py                 # 新版本
```

**核心改动**：

**1. adaptive_modules/fact_extractor.py**
```python
"""
事实提取模块 - 评估从表格中提取事实的质量
"""
import json
import re

# 从 REAP prompts 移植并修改
SYSTEM_PROMPT_FACT_EXTRACTION = """
You are a table data extraction expert. Your task is to extract facts from retrieved tables that answer a specific requirement.

**Input**:
1. A specific requirement (sub-question)
2. Retrieved tables (with caption, headers, and rows)
3. Previously collected facts (context)

**Your Task**:
1. Read the tables carefully
2. Identify the relevant columns and rows
3. Extract the precise fact that answers the requirement
4. Classify the extraction quality

**Extraction Quality Levels**:
- **DIRECT_ANSWER**: Found a clear, direct answer in the tables
- **PARTIAL_CLUE**: Found partial information, but not complete
- **FAILED_EXTRACT**: No relevant information in the tables

**Output Format** (JSON only):
```json
{
  "reasoned_facts": [
    {
      "fulfills_requirement_id": "req1",
      "reasoning": "<explain how you found the answer in the table>",
      "statement": "<the extracted fact>",
      "fulfillment_level": "DIRECT_ANSWER|PARTIAL_CLUE|FAILED_EXTRACT"
    }
  ]
}
```

**Example**:
Requirement: "What is the total revenue in 2023?"
Table: Financial Report with columns [Year, Revenue, Profit]
Row: [2023, $1.2B, $200M]

Output:
```json
{
  "reasoned_facts": [
    {
      "fulfills_requirement_id": "req1",
      "reasoning": "Found in Financial Report table, column 'Revenue', row where Year=2023",
      "statement": "The total revenue in 2023 was $1.2B",
      "fulfillment_level": "DIRECT_ANSWER"
    }
  ]
}
```

CRITICAL: Output ONLY valid JSON.
"""

def extract_fact_from_tables(
    requirement: dict,
    retrieved_tables: list,
    collected_facts: list,
    llm_call_func
) -> dict:
    """
    从检索到的表格中提取事实

    Args:
        requirement: {"requirement_id": "req1", "question": "..."}
        retrieved_tables: 检索到的表格列表
        collected_facts: 已收集的事实（提供上下文）
        llm_call_func: LLM调用函数

    Returns:
        {
            "reasoned_facts": [
                {
                    "fulfills_requirement_id": "req1",
                    "reasoning": "...",
                    "statement": "...",
                    "fulfillment_level": "DIRECT_ANSWER"
                }
            ]
        }
    """
    # 构建 prompt
    system_prompt = SYSTEM_PROMPT_FACT_EXTRACTION

    # 格式化表格
    tables_text = ""
    for i, table in enumerate(retrieved_tables[:10]):  # 限制表格数量
        tables_text += f"\nTable {i+1}:\n"
        tables_text += f"Caption: {table.get('caption', 'N/A')}\n"
        headers = table.get('table', {}).get('header', [])
        rows = table.get('table', {}).get('rows', [])
        tables_text += f"Headers: {' | '.join(headers)}\n"
        tables_text += f"Rows (showing first 5):\n"
        for row in rows[:5]:
            tables_text += f"  {' | '.join(row)}\n"

    # 格式化已知事实
    facts_text = ""
    if collected_facts:
        facts_text = "\nPreviously collected facts:\n"
        for fact in collected_facts:
            facts_text += f"- {fact.get('statement', '')}\n"

    user_prompt = f"""
Requirement ID: {requirement['requirement_id']}
Requirement Question: {requirement['question']}

Retrieved Tables:
{tables_text}
{facts_text}

Extract the fact that answers this requirement.
"""

    # 调用 LLM
    response = llm_call_func(system_prompt, user_prompt)

    # 解析 JSON
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        # 降级：返回失败结果
        return {
            "reasoned_facts": [
                {
                    "fulfills_requirement_id": requirement['requirement_id'],
                    "reasoning": "Failed to parse LLM response",
                    "statement": "N/A",
                    "fulfillment_level": "FAILED_EXTRACT"
                }
            ]
        }

    result = json.loads(match.group(0))
    return result
```

**2. call_llm_v2.py** (关键修改)
```python
"""
T-RAG with Fact Extraction (Version 2)
新增参数：--use_fact_extraction
"""
from adaptive_modules.fact_extractor import extract_fact_from_tables

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-RAG with Adaptive Planning V2")
    # ... 原有参数 ...
    parser.add_argument("--use_decomposition", action="store_true")
    parser.add_argument("--use_fact_extraction", action="store_true",
                        help="Enable fact extraction quality assessment (V2)")

    args = parser.parse_args()

    # ... 原有代码 ...

    fact_extraction_log = []

    for i, line in enumerate(tqdm(retrieve_instances)):
        retrieve_instance = json.loads(line)
        query = retrieve_instance["query"]
        retrieved_tables = retrieve_instance["retrieved_tables"]

        # V1: 查询分解
        decomposition = None
        if args.use_decomposition:
            decomposition = analyze_and_decompose_query(query, llm_call_func)

        # V2: 事实提取评估
        if args.use_fact_extraction and decomposition:
            collected_facts = []

            # 为每个需求提取事实
            for req in decomposition["requirements"]:
                extraction_result = extract_fact_from_tables(
                    requirement=req,
                    retrieved_tables=retrieved_tables,
                    collected_facts=collected_facts,
                    llm_call_func=llm_call_func
                )

                # 记录结果
                fact_extraction_log.append({
                    "query": query,
                    "requirement": req,
                    "extraction": extraction_result
                })

                # 收集成功提取的事实
                for fact in extraction_result["reasoned_facts"]:
                    if fact["fulfillment_level"] != "FAILED_EXTRACT":
                        collected_facts.append(fact)

                print(f"[V2] Extracted fact for {req['requirement_id']}: "
                      f"{fact['fulfillment_level']}")

        # 后续推理逻辑保持不变
        # ... (原有的完整表格推理) ...

    # 保存事实提取日志
    if args.use_fact_extraction:
        fact_log_file = f"{output_dir}/fact_extraction_log_{testing_num}_{topk}.jsonl"
        with open(fact_log_file, "w") as f:
            for entry in fact_extraction_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**运行方法**：
```bash
# 对比实验：V2 vs V1
python call_llm_v2.py \
    --dataset sqa \
    --topk 50 \
    --model gpt-4o-mini \
    --testing_num 100 \
    --use_decomposition \
    --use_fact_extraction
```

**评估分析**：
```python
# 分析 fact_extraction_log.jsonl
import json

with open("fact_extraction_log_100_50.jsonl") as f:
    logs = [json.loads(line) for line in f]

# 统计提取质量
stats = {
    "DIRECT_ANSWER": 0,
    "PARTIAL_CLUE": 0,
    "FAILED_EXTRACT": 0
}

for log in logs:
    level = log["extraction"]["reasoned_facts"][0]["fulfillment_level"]
    stats[level] += 1

print("Fact Extraction Quality:")
print(f"  DIRECT_ANSWER: {stats['DIRECT_ANSWER']} ({stats['DIRECT_ANSWER']/len(logs)*100:.1f}%)")
print(f"  PARTIAL_CLUE: {stats['PARTIAL_CLUE']} ({stats['PARTIAL_CLUE']/len(logs)*100:.1f}%)")
print(f"  FAILED_EXTRACT: {stats['FAILED_EXTRACT']} ({stats['FAILED_EXTRACT']/len(logs)*100:.1f}%)")
```

---

### 版本3：添加重规划能力

**目标**：当事实提取失败时，触发查询改写并重新检索

**新增文件**：
```
src/downstream_inference/adaptive_modules/
├── replanner.py                   # 新增
└── call_llm_v3.py                 # 新版本
```

**核心改动**：

**1. adaptive_modules/replanner.py**
```python
"""
重规划模块 - 当检索失败时改写查询
"""
import json
import re

SYSTEM_PROMPT_REPLAN_LITE = """
You are a table search query optimization expert. When a search query fails to retrieve relevant tables or extract facts, you need to reformulate it.

**Your Task**:
Given:
1. Original query that failed
2. Why it failed (e.g., PARTIAL_CLUE or FAILED_EXTRACT)
3. Retrieved tables (which were not helpful)

Generate:
- A reformulated query that is more likely to succeed

**Reformulation Strategies**:
1. **Be more specific**: Add constraints, column names, or table types
2. **Change keywords**: Use synonyms or alternative phrasings
3. **Simplify**: If query was too complex, break it down further
4. **Add context**: Include domain-specific terms

**Output Format** (JSON only):
```json
{
  "diagnosis": "<why the original query failed>",
  "reformulated_query": "<the new query to try>",
  "strategy": "<which strategy you used>"
}
```

Example:
Original: "What is the revenue?"
Failed because: Too vague, many revenue tables
Reformulated: "What is the total revenue in 2023 from the annual financial report?"

CRITICAL: Output ONLY JSON.
"""

def replan_on_failure(
    original_query: str,
    failed_fact: dict,
    retrieved_tables: list,
    llm_call_func
) -> str:
    """
    当事实提取失败时，重新规划查询

    Args:
        original_query: 原始查询
        failed_fact: 失败的事实提取结果
        retrieved_tables: 检索到的表格（未成功）
        llm_call_func: LLM调用函数

    Returns:
        reformulated_query: 改写后的查询
    """
    system_prompt = SYSTEM_PROMPT_REPLAN_LITE

    # 构建失败信息
    failure_info = f"""
Original Query: {original_query}
Failure Level: {failed_fact.get('fulfillment_level', 'UNKNOWN')}
Failure Reasoning: {failed_fact.get('reasoning', 'N/A')}

Retrieved Tables (captions):
"""
    for i, table in enumerate(retrieved_tables[:5]):
        failure_info += f"{i+1}. {table.get('caption', 'No caption')}\n"

    user_prompt = failure_info + "\n\nReformulate the query to improve retrieval."

    # 调用 LLM
    response = llm_call_func(system_prompt, user_prompt)

    # 解析 JSON
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        # 降级：返回原查询
        return original_query

    result = json.loads(match.group(0))
    return result.get("reformulated_query", original_query)
```

**2. call_llm_v3.py** (关键修改)
```python
"""
T-RAG with Replanning (Version 3)
新增参数：--use_replan
"""
from adaptive_modules.replanner import replan_on_failure

# 需要集成表格检索模块
import sys
sys.path.append("../table2graph/subgraph_retrieve")
from subgraph_retrieve_sentencetransformer import retrieve_tables_for_query  # 封装后的接口

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-RAG with Adaptive Planning V3")
    # ... 原有参数 ...
    parser.add_argument("--use_replan", action="store_true",
                        help="Enable query replanning on extraction failure (V3)")
    parser.add_argument("--max_replan_attempts", type=int, default=2,
                        help="Maximum replanning attempts")

    args = parser.parse_args()

    # ... 原有代码 ...

    replan_log = []

    for i, line in enumerate(tqdm(retrieve_instances)):
        retrieve_instance = json.loads(line)
        query = retrieve_instance["query"]
        retrieved_tables = retrieve_instance["retrieved_tables"]

        # V1: 查询分解
        decomposition = None
        if args.use_decomposition:
            decomposition = analyze_and_decompose_query(query, llm_call_func)

        # V2 & V3: 事实提取 + 重规划
        if args.use_fact_extraction and decomposition:
            collected_facts = []

            for req in decomposition["requirements"]:
                current_query = req["question"]
                current_tables = retrieved_tables
                attempts = 0

                while attempts < args.max_replan_attempts:
                    # 提取事实
                    extraction_result = extract_fact_from_tables(
                        requirement=req,
                        retrieved_tables=current_tables,
                        collected_facts=collected_facts,
                        llm_call_func=llm_call_func
                    )

                    fact = extraction_result["reasoned_facts"][0]

                    # V3: 如果提取失败且启用重规划
                    if (fact["fulfillment_level"] in ["PARTIAL_CLUE", "FAILED_EXTRACT"]
                        and args.use_replan and attempts < args.max_replan_attempts - 1):

                        print(f"[V3] Extraction failed ({fact['fulfillment_level']}), "
                              f"replanning... (attempt {attempts + 1})")

                        # 重新规划查询
                        reformulated_query = replan_on_failure(
                            original_query=current_query,
                            failed_fact=fact,
                            retrieved_tables=current_tables,
                            llm_call_func=llm_call_func
                        )

                        # 记录重规划
                        replan_log.append({
                            "original_query": current_query,
                            "reformulated_query": reformulated_query,
                            "attempt": attempts + 1,
                            "reason": fact["fulfillment_level"]
                        })

                        # 重新检索（注意：这里需要调用 T-RAG 的检索模块）
                        # 简化版：使用相同的表格集合但改变查询
                        # 完整版：需要重新运行 PageRank
                        current_query = reformulated_query
                        # current_tables = retrieve_tables_for_query(reformulated_query, topk)

                        attempts += 1
                    else:
                        # 提取成功或达到最大尝试次数
                        if fact["fulfillment_level"] != "FAILED_EXTRACT":
                            collected_facts.append(fact)
                        break

        # 后续推理逻辑保持不变
        # ...

    # 保存重规划日志
    if args.use_replan:
        replan_log_file = f"{output_dir}/replan_log_{testing_num}_{topk}.jsonl"
        with open(replan_log_file, "w") as f:
            for entry in replan_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**运行方法**：
```bash
# V3 完整测试
python call_llm_v3.py \
    --dataset sqa \
    --topk 50 \
    --model gpt-4o-mini \
    --testing_num 100 \
    --use_decomposition \
    --use_fact_extraction \
    --use_replan \
    --max_replan_attempts 2
```

**评估分析**：
```python
# 分析重规划效果
import json

with open("replan_log_100_50.jsonl") as f:
    replan_logs = [json.loads(line) for line in f]

print(f"Total replanning events: {len(replan_logs)}")
print(f"Queries that triggered replan: {len(set(log['original_query'] for log in replan_logs))}")

# 对比 V2 和 V3 的 EM/F1
# 预期：V3 在多跳查询上有提升
```

---

### 版本4：完整 Orchestrator

**目标**：实现完整的迭代式检索-推理循环

**新增文件**：
```
src/downstream_inference/adaptive_modules/
├── orchestrator.py                # 新增
└── call_adaptive_rag.py           # 全新文件（独立于 call_llm.py）
```

**核心改动**：

**1. adaptive_modules/orchestrator.py**
```python
"""
Adaptive RAG 编排器 - 完整的迭代循环
"""
import json
from .query_decomposer import analyze_and_decompose_query
from .fact_extractor import extract_fact_from_tables
from .replanner import replan_on_failure

class AdaptiveRAGOrchestrator:
    def __init__(
        self,
        llm_call_func,
        retrieve_func,
        max_iterations=3,
        enable_replan=True
    ):
        """
        初始化编排器

        Args:
            llm_call_func: LLM调用函数
            retrieve_func: 表格检索函数 (query, topk) -> tables
            max_iterations: 最大迭代次数
            enable_replan: 是否启用重规划
        """
        self.llm_call_func = llm_call_func
        self.retrieve_func = retrieve_func
        self.max_iterations = max_iterations
        self.enable_replan = enable_replan
        self.trace_log = []

    def run(self, query: str, topk: int = 50) -> dict:
        """
        运行完整的 Adaptive RAG 流程

        Returns:
            {
                "final_answer": str,
                "collected_facts": list,
                "iterations": int,
                "trace": list
            }
        """
        # Stage 1: 查询分解
        decomposition = analyze_and_decompose_query(query, self.llm_call_func)
        requirements = decomposition["requirements"]

        self.trace_log.append({
            "stage": "decomposition",
            "requirements": requirements
        })

        # Stage 2: 构建依赖图并拓扑排序
        sorted_reqs = self._topological_sort(requirements)

        # Stage 3: 迭代执行
        collected_facts = []
        iteration = 0

        for req in sorted_reqs:
            # 用已知事实替换占位符
            concrete_query = self._substitute_facts(req["question"], collected_facts)

            # 检索表格
            retrieved_tables = self.retrieve_func(concrete_query, topk)

            # 提取事实（最多尝试2次）
            for attempt in range(2):
                extraction_result = extract_fact_from_tables(
                    requirement=req,
                    retrieved_tables=retrieved_tables,
                    collected_facts=collected_facts,
                    llm_call_func=self.llm_call_func
                )

                fact = extraction_result["reasoned_facts"][0]

                # 如果失败且启用重规划
                if (fact["fulfillment_level"] != "DIRECT_ANSWER"
                    and self.enable_replan and attempt == 0):

                    # 重新规划
                    reformulated_query = replan_on_failure(
                        original_query=concrete_query,
                        failed_fact=fact,
                        retrieved_tables=retrieved_tables,
                        llm_call_func=self.llm_call_func
                    )

                    # 重新检索
                    concrete_query = reformulated_query
                    retrieved_tables = self.retrieve_func(reformulated_query, topk)

                    self.trace_log.append({
                        "stage": "replan",
                        "requirement_id": req["requirement_id"],
                        "original_query": req["question"],
                        "reformulated_query": reformulated_query
                    })
                else:
                    # 成功或放弃
                    break

            # 收集事实
            if fact["fulfillment_level"] != "FAILED_EXTRACT":
                collected_facts.append(fact)

            self.trace_log.append({
                "stage": "extraction",
                "requirement_id": req["requirement_id"],
                "fact": fact
            })

            iteration += 1
            if iteration >= self.max_iterations:
                break

        # Stage 4: 合成最终答案
        final_answer = self._synthesize_answer(query, collected_facts)

        return {
            "final_answer": final_answer,
            "collected_facts": collected_facts,
            "iterations": iteration,
            "trace": self.trace_log
        }

    def _topological_sort(self, requirements):
        """拓扑排序（处理依赖关系）"""
        # 简化实现：假设依赖是线性的
        sorted_reqs = []
        visited = set()

        def visit(req_id):
            if req_id in visited:
                return
            req = next(r for r in requirements if r["requirement_id"] == req_id)
            if req["depends_on"]:
                visit(req["depends_on"])
            sorted_reqs.append(req)
            visited.add(req_id)

        for req in requirements:
            visit(req["requirement_id"])

        return sorted_reqs

    def _substitute_facts(self, question, collected_facts):
        """用已知事实替换占位符"""
        import re

        # 查找占位符 [answer from req1]
        pattern = r'\[answer from (req\d+)\]'
        matches = re.findall(pattern, question)

        for req_id in matches:
            # 找到对应的事实
            fact = next(
                (f for f in collected_facts
                 if f["fulfills_requirement_id"] == req_id),
                None
            )
            if fact:
                # 替换占位符
                placeholder = f"[answer from {req_id}]"
                question = question.replace(placeholder, fact["statement"])

        return question

    def _synthesize_answer(self, query, collected_facts):
        """合成最终答案"""
        # 简化版：直接使用最后一个事实的 statement
        if collected_facts:
            return collected_facts[-1]["statement"]
        return "Unable to answer based on retrieved tables."
```

**2. call_adaptive_rag.py** (全新独立文件)
```python
"""
T-RAG with Full Adaptive Planning (Version 4)
完全独立的实现，不修改原有 call_llm.py
"""
import json
import argparse
from tqdm import tqdm
from adaptive_modules.orchestrator import AdaptiveRAGOrchestrator
import sys
sys.path.append("../table2graph/subgraph_retrieve")

def create_retrieve_function(dataset, cluster_method, topk):
    """
    创建表格检索函数的工厂
    """
    # 这里需要封装 T-RAG 的检索逻辑
    # 简化版：从预检索的结果中读取
    def retrieve_func(query, k):
        # TODO: 调用 T-RAG 的实际检索模块
        # 现在先返回预检索的结果
        return []

    return retrieve_func

def main():
    parser = argparse.ArgumentParser(description="T-RAG with Full Adaptive Planning")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--testing_num", type=int, required=True)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--enable_replan", action="store_true")

    args = parser.parse_args()

    # 读取预检索的结果
    retrieve_file = f"../table2graph/data/{args.dataset}/{args.dataset}_retrieved_tables_schema_{args.testing_num}_{args.topk}_contriever.jsonl"

    with open(retrieve_file) as f:
        instances = [json.loads(line) for line in f]

    # 创建 LLM 调用函数
    def llm_call_func(system_prompt, user_prompt):
        # TODO: 调用实际的 LLM API
        return ""

    # 创建检索函数
    retrieve_func = create_retrieve_function(args.dataset, "contriever", args.topk)

    # 创建编排器
    orchestrator = AdaptiveRAGOrchestrator(
        llm_call_func=llm_call_func,
        retrieve_func=retrieve_func,
        max_iterations=args.max_iterations,
        enable_replan=args.enable_replan
    )

    # 运行
    results = []
    for instance in tqdm(instances):
        query = instance["query"]
        result = orchestrator.run(query, args.topk)

        results.append({
            "query": query,
            "final_answer": result["final_answer"],
            "ground_truth": instance["query_label"],
            "iterations": result["iterations"],
            "trace": result["trace"]
        })

    # 保存结果
    output_file = f"output/{args.dataset}/{args.model}/adaptive_rag_output_{args.testing_num}_{args.topk}.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
```

**运行方法**：
```bash
# V4 完整 Adaptive RAG
python call_adaptive_rag.py \
    --dataset sqa \
    --topk 50 \
    --model gpt-4o-mini \
    --testing_num 100 \
    --max_iterations 3 \
    --enable_replan
```

---

## 4. 评估对比方法

### 4.1 建立 Baseline

```bash
# 运行原始 T-RAG
cd /Users/sunyifei/Documents/GitHub/T-RAG/src/downstream_inference

# 测试集：100个样本
python call_llm.py \
    --dataset sqa \
    --topk 50 \
    --model gpt-4o-mini \
    --testing_num 100 \
    --embedding_method contriever

python evaluation.py \
    --dataset sqa \
    --model gpt-4o-mini \
    --topk 50 \
    --testing_num 100

# 记录基线指标
cp output/sqa/gpt-4o-mini/results_100_50.json baseline_results.json
```

### 4.2 版本对比表

创建对比脚本：

```python
# compare_versions.py
import json
import pandas as pd

versions = [
    "baseline",
    "v1_decomposition",
    "v2_fact_extraction",
    "v3_replan",
    "v4_full_adaptive"
]

results = []

for version in versions:
    result_file = f"output/sqa/gpt-4o-mini/{version}_results_100_50.json"
    with open(result_file) as f:
        data = json.load(f)
        results.append({
            "Version": version,
            "EM": data["exact_match"],
            "F1": data["f1_score"],
            "Avg_Time": data.get("avg_inference_time", 0)
        })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))

# 计算提升
df["EM_Gain"] = df["EM"] - df.loc[0, "EM"]
df["F1_Gain"] = df["F1"] - df.loc[0, "F1"]
print("\nGains over baseline:")
print(df[["Version", "EM_Gain", "F1_Gain"]].to_markdown(index=False))
```

### 4.3 错误分析

```python
# error_analysis.py
import json

def analyze_errors(output_file, ground_truth_file):
    """
    分析哪些问题回答错误，以及原因
    """
    with open(output_file) as f:
        outputs = [json.loads(line) for line in f]

    errors = []
    for output in outputs:
        if output["predicted"] != output["ground_truth"]:
            errors.append(output)

    print(f"Total errors: {len(errors)}")

    # 分类错误类型
    error_types = {
        "retrieval_failure": 0,  # 检索失败
        "extraction_failure": 0,  # 提取失败
        "reasoning_failure": 0    # 推理失败
    }

    # 需要人工标注一部分错误
    return errors

# 使用
baseline_errors = analyze_errors("baseline_output.jsonl", "ground_truth.jsonl")
v4_errors = analyze_errors("v4_output.jsonl", "ground_truth.jsonl")

# 对比：V4 修复了哪些错误？引入了哪些新错误?
fixed = set(baseline_errors) - set(v4_errors)
new_errors = set(v4_errors) - set(baseline_errors)

print(f"Fixed errors: {len(fixed)}")
print(f"New errors: {len(new_errors)}")
```

---

## 5. 实施时间表

| 天数 | 任务 | 交付物 |
|------|------|--------|
| Day 0 | 环境验证 + 基线测试 | baseline_results.json |
| Day 1 | 实现 V1（查询分解） | call_llm_v1.py + decomposition_log |
| Day 2 | 实现 V2（事实提取） | call_llm_v2.py + fact_extraction_log |
| Day 3 | 实现 V3（重规划） | call_llm_v3.py + replan_log |
| Day 4-5 | 实现 V4（完整编排器） | call_adaptive_rag.py + orchestrator.py |
| Day 6 | 性能优化 + Prompt调优 | 最终版本 |
| Day 7 | 完整评估 + 报告 | 评估报告 + 对比表 |

---

## 6. 下一步行动

**立即开始**：
1. 验证 T-RAG 环境并运行基线（按照 1.3 节）
2. 记录基线指标
3. 我将开始实现 V1（查询分解模块）

**需要你确认的问题**：
1. 是否有 OpenAI API Key？（用于测试）
2. 测试集规模：100个样本够吗？还是需要更多？
3. 优先级：是否按照 V1 → V2 → V3 → V4 的顺序？
4. 是否需要我现在就开始写代码？

请告诉我你想从哪一步开始！
