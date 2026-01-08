# T-RAG V1: Query Decomposition

Version 1 adds **query decomposition** capability to T-RAG, enabling analysis of query complexity and preparation for multi-step reasoning in future versions.

## What's New in V1

✅ **Query Decomposition**: Automatically breaks complex queries into atomic sub-questions
✅ **Multi-hop Detection**: Identifies queries that require multiple reasoning steps
✅ **Decomposition Logging**: Records decomposition results for analysis
✅ **Backward Compatible**: Can run with/without decomposition (controlled by flag)
✅ **Statistics**: Tracks multi-hop vs single-hop query distribution

## Quick Start

### 1. Configure API Key

Edit `key.json`:
```json
{
    "openai": "sk-your-actual-api-key-here",
    "claude": "<YOUR_CLAUDE_API_KEY>"
}
```

### 2. Run Baseline (without decomposition)

```bash
python call_llm_v1.py \
    --dataset sqa \
    --topk 50 \
    --model gpt-4o-mini \
    --testing_num 100 \
    --embedding_method contriever \
    --mode API
```

This produces: `output/sqa/gpt-4o-mini/output_100_50_v1_baseline.jsonl`

### 3. Run with Decomposition (V1 feature)

```bash
python call_llm_v1.py \
    --dataset sqa \
    --topk 50 \
    --model gpt-4o-mini \
    --testing_num 100 \
    --embedding_method contriever \
    --mode API \
    --use_decomposition \
    --decomposition_verbose
```

This produces:
- `output/sqa/gpt-4o-mini/output_100_50_v1_decomp.jsonl` (inference results)
- `output/sqa/gpt-4o-mini/decomposition_log_100_50.jsonl` (decomposition details)
- `output/sqa/gpt-4o-mini/decomposition_stats_100_50.json` (statistics)

### 4. Run Comparison Script (Recommended)

```bash
bash run_v1_comparison.sh
```

This automatically runs both experiments and saves all results.

## Command Line Arguments

### Original Arguments (from base T-RAG)
- `--dataset`: Dataset name (e.g., `sqa`, `hybridqa`, `wtq`, `tabfact`)
- `--topk`: Number of retrieved tables (e.g., `50`)
- `--model`: Model name (e.g., `gpt-4o-mini`, `gpt-4o`)
- `--testing_num`: Number of test samples (e.g., `100`)
- `--embedding_method`: Embedding method used in retrieval (default: `contriever`)
- `--mode`: API or offline (default: `API`)
- `--starting_idx`: Starting index for partial runs (default: `0`)

### V1 New Arguments
- `--use_decomposition`: Enable query decomposition (default: `False`)
- `--decomposition_verbose`: Print detailed decomposition logs (default: `False`)

## Output Files

### 1. Inference Results
**File**: `output/{dataset}/{model}/output_{testing_num}_{topk}_v1_{baseline|decomp}.jsonl`

Format:
```json
{
  "query": "What position was held by the actress in Kiss and Tell?",
  "ground_truth": "Chief of Protocol",
  "generated_text": "<reasoning>...</reasoning><answer>Chief of Protocol</answer>",
  "decomposition": {
    "user_goal": "Find government position of actress",
    "requirements": [
      {
        "requirement_id": "req1",
        "question": "Who was the actress in Kiss and Tell?",
        "depends_on": null
      },
      {
        "requirement_id": "req2",
        "question": "What government position was held by [answer from req1]?",
        "depends_on": "req1"
      }
    ]
  }
}
```

### 2. Decomposition Log
**File**: `output/{dataset}/{model}/decomposition_log_{testing_num}_{topk}.jsonl`

Format:
```json
{
  "query_idx": 0,
  "query": "What position was held by the actress in Kiss and Tell?",
  "decomposition": { ... },
  "is_multi_hop": true,
  "num_requirements": 2
}
```

### 3. Decomposition Statistics
**File**: `output/{dataset}/{model}/decomposition_stats_{testing_num}_{topk}.json`

Format:
```json
{
  "total_queries": 100,
  "multi_hop_queries": 45,
  "single_hop_queries": 55,
  "avg_requirements": 1.65,
  "decomposition_failures": 0
}
```

## Analysis Examples

### View Decomposition Stats

```bash
cat output/sqa/gpt-4o-mini/decomposition_stats_100_50.json | python -m json.tool
```

### Find Multi-hop Queries

```python
import json

# Load decomposition log
with open('output/sqa/gpt-4o-mini/decomposition_log_100_50.jsonl') as f:
    logs = [json.loads(line) for line in f]

# Filter multi-hop queries
multi_hop = [log for log in logs if log['is_multi_hop']]

print(f"Total multi-hop queries: {len(multi_hop)}")
for log in multi_hop[:5]:  # Show first 5
    print(f"\nQuery: {log['query']}")
    print(f"Requirements: {log['num_requirements']}")
    for req in log['decomposition']['requirements']:
        print(f"  - {req['requirement_id']}: {req['question']}")
```

### Compare Decomposition Quality

```python
import json

with open('output/sqa/gpt-4o-mini/decomposition_log_100_50.jsonl') as f:
    logs = [json.loads(line) for line in f]

# Analyze requirement distribution
req_counts = {}
for log in logs:
    count = log['num_requirements']
    req_counts[count] = req_counts.get(count, 0) + 1

print("Requirement Distribution:")
for count in sorted(req_counts.keys()):
    print(f"  {count} requirements: {req_counts[count]} queries ({req_counts[count]/len(logs)*100:.1f}%)")
```

## Expected Behavior in V1

### What V1 Does
- ✅ Decomposes queries into atomic sub-questions
- ✅ Identifies dependencies between requirements
- ✅ Logs decomposition results for analysis
- ✅ Provides statistics on query complexity

### What V1 Does NOT Do (Yet)
- ❌ Does NOT use decomposition for inference (still single-pass)
- ❌ Does NOT execute requirements sequentially
- ❌ Does NOT extract facts per requirement
- ❌ Does NOT replan on failures

> **Note**: In V1, decomposition is purely for **analysis and logging**. The actual inference still uses the original T-RAG single-pass approach. This allows us to:
> 1. Validate decomposition quality without changing the baseline
> 2. Understand the distribution of query complexity in the dataset
> 3. Build a foundation for V2/V3/V4 where we'll actually use the decomposition

## Evaluation

After running inference, evaluate with:

```bash
python evaluation.py \
    --dataset sqa \
    --model gpt-4o-mini \
    --topk 50 \
    --testing_num 100
```

This will read the output file (either baseline or decomp) and compute:
- Exact Match (EM)
- F1 Score

## Troubleshooting

### Error: Retrieved tables file not found

```
❌ ERROR: Retrieved tables file not found!
Please run the retrieval pipeline first:
  cd ../table2graph
  bash scripts/table_cluster_run.sh
  python scripts/subgraph_retrieve_run.py
```

**Solution**: Run the T-RAG retrieval pipeline first to generate the retrieved tables file.

### Error: API key not found

```
ValueError: API key not found for model: gpt-4o-mini
```

**Solution**: Edit `key.json` and add your OpenAI API key.

### Decomposition returns single requirement for all queries

This might indicate:
1. The prompt needs adjustment for your specific dataset
2. The queries are actually simple single-hop queries
3. The LLM is not following the decomposition instruction

**Solution**: Check decomposition log and verify a few examples manually.

## Next Steps

After running V1:

1. **Analyze decomposition quality**: Check if multi-hop queries are correctly identified
2. **Review examples**: Manually review some decompositions in the log
3. **Prepare for V2**: If decomposition looks good, proceed to implement V2 (fact extraction)

## File Structure

```
downstream_inference/
├── adaptive_modules/
│   ├── __init__.py
│   ├── prompts.py              # Table-specific prompts
│   └── query_decomposer.py     # Decomposition logic
├── call_llm_v1.py               # V1 main script
├── run_v1_comparison.sh         # Automated comparison
├── V1_README.md                 # This file
└── output/
    └── {dataset}/
        └── {model}/
            ├── output_*_v1_baseline.jsonl
            ├── output_*_v1_decomp.jsonl
            ├── decomposition_log_*.jsonl
            └── decomposition_stats_*.json
```

## Development Notes

### Testing the Decomposer Module Alone

```bash
cd /Users/sunyifei/Documents/GitHub/T-RAG/src/downstream_inference
python -m adaptive_modules.query_decomposer
```

This runs the built-in test suite.

### Adjusting Decomposition Prompts

Edit `adaptive_modules/prompts.py` to customize:
- `SYSTEM_PROMPT_QUERY_ANALYSIS`: Main decomposition instruction
- Examples in the prompt to guide the LLM

## Version History

- **V1.0.0** (2025-01-08): Initial release with query decomposition
