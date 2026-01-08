"""
T-RAG with Adaptive Planning - Version 1: Query Decomposition

This version extends the original T-RAG inference with query decomposition capability.
In V1, we decompose queries but still use the original single-pass inference for compatibility.
This allows us to:
1. Analyze query complexity (single-hop vs multi-hop)
2. Log decomposition quality
3. Compare results with/without decomposition awareness

New features in V1:
- --use_decomposition flag: Enable query decomposition
- Decomposition logging: Save decomposition results for analysis
- Backward compatible: Can run with original behavior when flag is off

Usage:
    # Baseline (same as original call_llm.py)
    python call_llm_v1.py --dataset sqa --topk 50 --model gpt-4o-mini --testing_num 100

    # With decomposition (V1 feature)
    python call_llm_v1.py --dataset sqa --topk 50 --model gpt-4o-mini --testing_num 100 --use_decomposition
"""

import json
import openai
import pandas as pd
from tqdm import tqdm
import argparse
import os
import sys

# Import adaptive modules
from adaptive_modules.query_decomposer import (
    analyze_and_decompose_query,
    is_multi_hop_query,
    format_decomposition_for_display
)

# Import original utility functions (from call_llm.py)
# We'll copy the essential functions here to maintain independence


def get_few_shot_prompt(task_name: str):
    """Get few-shot examples for each task (same as original)"""
    few_shot_prompt_list = {
        "tabfact":
'''
- Example 1:
# Final Answer: <answer>1</answer>
- Example 2:
# Final Answer: <answer>0</answer>
- Example 3:
# Final Answer: <answer>1</answer>
''',
        "hybridqa":
'''
- Example 1:
Final Answer: <answer>Jerry</answer>
- Example 2:
Final Answer: <answer>Starke Rudolf</answer>
- Example 3:
Final Answer: <answer>British</answer>
''',
        "wtq":
'''
- Example 1:
# Final Answer: <answer>["Aberdeen vs Hamilton Academical"]</answer>
- Example 2:
# Final Answer: <answer>["19 min", "20 min", "33 min"]</answer>
- Example 3:
# Final Answer: <answer>["RC Narbonne", "Montpellier RC", "Aviron Bayonnais", "Section Paloise", "RC Toulonnais"]</answer>
''',
        "sqa":
'''
- Example 1:
# Final Answer: <answer>["Roberto Feliberti Cintron"]</answer>
- Example 2:
# Final Answer: <answer>["53"]</answer>
- Example 3:
# Final Answer: <answer>["13", "55", "S01", "S30", "S800c", "S1200pj"]</answer>
'''
    }
    return few_shot_prompt_list.get(task_name, "")


def get_instruction(task_name: str):
    """Get task-specific instructions (same as original)"""
    instruction_list = {
        "tabfact": "Use the retrieved most relevant tables to verify whether the provided claim/query are true or false. Work through the problem step by step, and then return 0 if it's false, or 1 if it's true. Only return 0 or 1 without any other information. \n",
        "hybridqa": "Use the retrieved most relevant tables to answer the question. Only return the string instead of other format information. Do not repeat the question. \n",
        "sqa": "Utilize the most relevant retrieved tables to answer the question. Work through the problem step by step, and then return a list of strings to include ALL POSSIBLE final answers to the query. Note: Do not add extra content in the final answer lists. \n",
        "wtq": "Utilize the most relevant retrieved tables to answer the question. Work through the problem step by step, and then return a list of strings to include ALL POSSIBLE final answers to the query. Note: Do not add extra content in the final answer lists\n",
    }
    return instruction_list.get(task_name, "")


def table_to_html(table_data: dict) -> str:
    """Convert table to HTML format (same as original)"""
    caption = table_data.get("caption", "")
    header = table_data["table"]["header"]
    rows = table_data["table"]["rows"]
    df = pd.DataFrame(rows, columns=header)
    html_table = df.to_html(index=False, escape=False)
    return f"<h3>{caption}</h3>\n{html_table}"


def construct_prompt_gpt(retrieve_instance: dict, dataset: str) -> str:
    """Construct GPT prompt (same as original)"""
    prompt = ""
    testing_query = retrieve_instance["query"]
    retrieve_tables = retrieve_instance["retrieved_tables"]

    system_prompt = '''
As a expert in tabular data analysis and RAG, you are given a query and a set of tables.
The query is the question you need to answer and the set of tables are the source of information you can retrieve to help you answer the given query.
You are asked to provide a response to the query based on the information in the tables. Follow the instructions below:

# Step one: Find most relevant tables to answer the query
1. Read the query and the tables carefully.
2. Given the query information, figure out and find the most relevant tables (normally 1-3 tables) from the set of tables to answer the query.
3. Once you have identified the relevant tables, follow the step two to answer the query.
4. Note that sometimes the answer of the query may not be directly obtained from the given tables or the tables might totally irrelevant to the query. In this case, You need to think step by step and try your best to answer the question based on your pre-trained knowledge.

# Step two: Given Task Instructions:
'''
    prompt += system_prompt
    prompt += get_instruction(dataset)
    prompt += '''
# The Query:
'''
    prompt += testing_query + "\n"
    prompt += '''
# The Table Set:
'''

    for i, table in enumerate(retrieve_tables):
        table_html = table_to_html(table)
        prompt += f"Table {i+1}:\n {table_html}\n"

    prompt += f'''
# Step three: Output Instructions: Here we provide output instructions that you MUST strictly follow.
1. You MUST think step by step via the chain-of-thought for the given task and then give a final answer.
2. Your output MUST conclude two compenents: the chain-of-thought (CoT) steps to reach the final answer and the final answer.
3. For the CoT component, you MUST enclose your reasoning between <reasoning> and </reasoning> tags.
4. For the final answer component, you MUST enclose your reasoning between <answer> and </answer> tags.
Here are few-shot examples to demonstrate the final answer component format:
{get_few_shot_prompt(dataset)}
5. If you try your best but still cannot find the answer from both the given table sources and your pretrained knowledge, then output your thinking steps and the final answer using <answer>NA</answer> to indicate that the answer can not be answered.
'''
    prompt += "\n# Now Output Your response below:"
    return prompt


def call_openai_api(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str:
    """
    Call OpenAI API (simplified for decomposition)

    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        model: Model name (e.g., gpt-4o-mini)
        api_key: OpenAI API key

    Returns:
        LLM response text
    """
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=2048
    )

    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="T-RAG with Adaptive Planning V1 - Query Decomposition")

    # Original arguments
    parser.add_argument("--topk", type=int, required=True, help="Number of retrieved tables")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--mode", type=str, default="API", help="API or offline")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--starting_idx", type=int, default=0, help="Starting index")
    parser.add_argument("--testing_num", type=int, required=True, help="Number of test queries")
    parser.add_argument("--embedding_method", type=str, default="contriever",
                        help="Embedding method used during retrieval")

    # V1 new argument
    parser.add_argument("--use_decomposition", action="store_true",
                        help="[V1] Enable query decomposition and logging")
    parser.add_argument("--decomposition_verbose", action="store_true",
                        help="[V1] Print detailed decomposition logs")

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*70)
    print("T-RAG with Adaptive Planning - Version 1")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Top-K: {args.topk}")
    print(f"Testing samples: {args.testing_num}")
    print(f"Embedding method: {args.embedding_method}")
    print(f"V1 Decomposition: {'ENABLED' if args.use_decomposition else 'DISABLED'}")
    print("="*70 + "\n")

    # Setup paths
    retrieve_table_file = f"../table2graph/data/{args.dataset}/{args.dataset}_retrieved_tables_schema_{args.testing_num}_{args.topk}_{args.embedding_method}.jsonl"

    output_dir = f"./output/{args.dataset}/{args.model}/"
    os.makedirs(output_dir, exist_ok=True)

    # V1: Add suffix for decomposition runs
    suffix = "_v1_decomp" if args.use_decomposition else "_v1_baseline"
    output_file = f"{output_dir}/output_{args.testing_num}_{args.topk}{suffix}.jsonl"

    # Load API key
    key_file = "./key.json"
    with open(key_file, "r") as f:
        keys = json.load(f)

    if "gpt" in args.model:
        api_key = keys["openai"]
    elif "claude" in args.model:
        api_key = keys["claude"]
    else:
        raise ValueError(f"API key not found for model: {args.model}")

    # Load retrieved instances
    print(f"Loading retrieved tables from: {retrieve_table_file}")
    if not os.path.exists(retrieve_table_file):
        print(f"\n❌ ERROR: Retrieved tables file not found!")
        print(f"Please run the retrieval pipeline first:")
        print(f"  cd ../table2graph")
        print(f"  bash scripts/table_cluster_run.sh")
        print(f"  python scripts/subgraph_retrieve_run.py")
        sys.exit(1)

    with open(retrieve_table_file, "r") as f:
        retrieve_instances = [json.loads(line) for line in f]

    print(f"Loaded {len(retrieve_instances)} instances\n")

    # V1: Initialize decomposition logging
    decomposition_log = []
    decomposition_stats = {
        "total_queries": 0,
        "multi_hop_queries": 0,
        "single_hop_queries": 0,
        "avg_requirements": 0,
        "decomposition_failures": 0
    }

    # Process queries
    print("Processing queries...")
    results = []

    for idx, line in enumerate(tqdm(retrieve_instances, desc="Inference")):
        retrieve_instance = line
        query = retrieve_instance["query"]
        groundtruth = retrieve_instance["query_label"]

        # =================================================================
        # V1: QUERY DECOMPOSITION (NEW)
        # =================================================================
        decomposition_result = None

        if args.use_decomposition:
            try:
                # Create LLM call function for decomposer
                def llm_call_func(system_prompt, user_prompt):
                    return call_openai_api(system_prompt, user_prompt, args.model, api_key)

                # Decompose query
                decomposition_result = analyze_and_decompose_query(
                    query=query,
                    llm_call_func=llm_call_func,
                    verbose=args.decomposition_verbose
                )

                # Update statistics
                decomposition_stats["total_queries"] += 1
                if is_multi_hop_query(decomposition_result):
                    decomposition_stats["multi_hop_queries"] += 1
                else:
                    decomposition_stats["single_hop_queries"] += 1

                num_reqs = len(decomposition_result["requirements"])
                decomposition_stats["avg_requirements"] += num_reqs

                # Log this decomposition
                decomposition_log.append({
                    "query_idx": idx,
                    "query": query,
                    "decomposition": decomposition_result,
                    "is_multi_hop": is_multi_hop_query(decomposition_result),
                    "num_requirements": num_reqs
                })

                if args.decomposition_verbose:
                    print(f"\n[Query {idx}] Decomposition:")
                    print(format_decomposition_for_display(decomposition_result))
                    print()

            except Exception as e:
                print(f"\n[WARNING] Decomposition failed for query {idx}: {e}")
                decomposition_stats["decomposition_failures"] += 1
                decomposition_result = None

        # =================================================================
        # ORIGINAL INFERENCE LOGIC (UNCHANGED)
        # =================================================================
        # In V1, we still use the original single-pass inference
        # The decomposition is only for logging and analysis

        # Construct prompt (same as original)
        if "gpt" in args.model:
            prompt = construct_prompt_gpt(retrieve_instance, args.dataset)
        else:
            raise ValueError(f"Prompt construction not implemented for model: {args.model}")

        # Call LLM (same as original)
        try:
            response = call_openai_api(
                system_prompt="",  # Already in prompt
                user_prompt=prompt,
                model=args.model,
                api_key=api_key
            )

            # Save result
            results.append({
                "query": query,
                "ground_truth": groundtruth,
                "generated_text": response,
                "decomposition": decomposition_result if args.use_decomposition else None
            })

        except Exception as e:
            print(f"\n[ERROR] Inference failed for query {idx}: {e}")
            results.append({
                "query": query,
                "ground_truth": groundtruth,
                "generated_text": f"ERROR: {str(e)}",
                "decomposition": decomposition_result if args.use_decomposition else None
            })

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    print(f"\nSaving results to: {output_file}")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # V1: Save decomposition log
    if args.use_decomposition:
        decomposition_log_file = f"{output_dir}/decomposition_log_{args.testing_num}_{args.topk}.jsonl"
        print(f"Saving decomposition log to: {decomposition_log_file}")

        with open(decomposition_log_file, "w") as f:
            for entry in decomposition_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Calculate final statistics
        if decomposition_stats["total_queries"] > 0:
            decomposition_stats["avg_requirements"] /= decomposition_stats["total_queries"]

        # Save statistics
        stats_file = f"{output_dir}/decomposition_stats_{args.testing_num}_{args.topk}.json"
        with open(stats_file, "w") as f:
            json.dump(decomposition_stats, f, indent=2)

        # Print statistics
        print("\n" + "="*70)
        print("V1 DECOMPOSITION STATISTICS")
        print("="*70)
        print(f"Total queries: {decomposition_stats['total_queries']}")
        print(f"Multi-hop queries: {decomposition_stats['multi_hop_queries']} "
              f"({decomposition_stats['multi_hop_queries']/max(decomposition_stats['total_queries'],1)*100:.1f}%)")
        print(f"Single-hop queries: {decomposition_stats['single_hop_queries']} "
              f"({decomposition_stats['single_hop_queries']/max(decomposition_stats['total_queries'],1)*100:.1f}%)")
        print(f"Average requirements per query: {decomposition_stats['avg_requirements']:.2f}")
        print(f"Decomposition failures: {decomposition_stats['decomposition_failures']}")
        print("="*70 + "\n")

    print("✅ Inference complete!")
    print(f"\nNext step: Run evaluation")
    print(f"  python evaluation.py --dataset {args.dataset} --model {args.model} "
          f"--topk {args.topk} --testing_num {args.testing_num}")


if __name__ == "__main__":
    main()
