#!/bin/bash
#
# T-RAG V1 Comparison Script
# This script runs both baseline and decomposition modes for comparison
#

# Configuration
DATASET="sqa"
MODEL="gpt-4o-mini"
TOPK=50
TESTING_NUM=100
EMBEDDING_METHOD="contriever"

echo "=========================================="
echo "T-RAG V1 Comparison Experiment"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Top-K: $TOPK"
echo "Testing samples: $TESTING_NUM"
echo "=========================================="
echo ""

# Check if retrieved tables exist
RETRIEVE_FILE="../table2graph/data/${DATASET}/${DATASET}_retrieved_tables_schema_${TESTING_NUM}_${TOPK}_${EMBEDDING_METHOD}.jsonl"

if [ ! -f "$RETRIEVE_FILE" ]; then
    echo "❌ ERROR: Retrieved tables file not found!"
    echo "File: $RETRIEVE_FILE"
    echo ""
    echo "Please run the retrieval pipeline first:"
    echo "  cd ../table2graph"
    echo "  bash scripts/table_cluster_run.sh"
    echo "  python scripts/subgraph_retrieve_run.py"
    exit 1
fi

echo "✓ Retrieved tables file found"
echo ""

# ==========================================
# Experiment 1: Baseline (no decomposition)
# ==========================================
echo "=========================================="
echo "Experiment 1: Baseline (V1 without decomposition)"
echo "=========================================="
echo "Running inference..."

python call_llm_v1.py \
    --dataset $DATASET \
    --topk $TOPK \
    --model $MODEL \
    --testing_num $TESTING_NUM \
    --embedding_method $EMBEDDING_METHOD \
    --mode API

if [ $? -ne 0 ]; then
    echo "❌ Baseline inference failed!"
    exit 1
fi

echo ""
echo "Running evaluation..."

python evaluation.py \
    --dataset $DATASET \
    --model $MODEL \
    --topk $TOPK \
    --testing_num $TESTING_NUM

echo ""
echo "✓ Baseline complete!"
echo "Results saved to: output/${DATASET}/${MODEL}/output_${TESTING_NUM}_${TOPK}_v1_baseline.jsonl"
echo ""

# ==========================================
# Experiment 2: With Decomposition
# ==========================================
echo "=========================================="
echo "Experiment 2: With Query Decomposition (V1 feature)"
echo "=========================================="
echo "Running inference with decomposition..."

python call_llm_v1.py \
    --dataset $DATASET \
    --topk $TOPK \
    --model $MODEL \
    --testing_num $TESTING_NUM \
    --embedding_method $EMBEDDING_METHOD \
    --mode API \
    --use_decomposition \
    --decomposition_verbose

if [ $? -ne 0 ]; then
    echo "❌ Decomposition inference failed!"
    exit 1
fi

echo ""
echo "Running evaluation..."

python evaluation.py \
    --dataset $DATASET \
    --model $MODEL \
    --topk $TOPK \
    --testing_num $TESTING_NUM

echo ""
echo "✓ Decomposition complete!"
echo "Results saved to: output/${DATASET}/${MODEL}/output_${TESTING_NUM}_${TOPK}_v1_decomp.jsonl"
echo "Decomposition log: output/${DATASET}/${MODEL}/decomposition_log_${TESTING_NUM}_${TOPK}.jsonl"
echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "V1 COMPARISON COMPLETE"
echo "=========================================="
echo ""
echo "Result files:"
echo "  Baseline:      output/${DATASET}/${MODEL}/output_${TESTING_NUM}_${TOPK}_v1_baseline.jsonl"
echo "  Decomposition: output/${DATASET}/${MODEL}/output_${TESTING_NUM}_${TOPK}_v1_decomp.jsonl"
echo "  Decomp Log:    output/${DATASET}/${MODEL}/decomposition_log_${TESTING_NUM}_${TOPK}.jsonl"
echo "  Decomp Stats:  output/${DATASET}/${MODEL}/decomposition_stats_${TESTING_NUM}_${TOPK}.json"
echo ""
echo "Next steps:"
echo "  1. Compare results using evaluation.py"
echo "  2. Analyze decomposition log to understand query complexity"
echo "  3. Check decomposition stats for multi-hop vs single-hop distribution"
echo ""
