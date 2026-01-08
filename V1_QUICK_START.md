# T-RAG V1 快速开始指南（服务器运行）

## ✅ V1 已完成的工作

### 新增文件
```
T-RAG/src/downstream_inference/
├── adaptive_modules/
│   ├── __init__.py              ✅ 模块初始化
│   ├── prompts.py               ✅ 表格专用 Prompt 模板
│   └── query_decomposer.py      ✅ 查询分解逻辑（300+ 行，完整实现）
├── call_llm_v1.py               ✅ V1 主程序（400+ 行，完全兼容原版）
├── run_v1_comparison.sh         ✅ 对比实验脚本
├── test_v1_module.py            ✅ 模块测试脚本
└── V1_README.md                 ✅ 详细使用文档
```

### 核心功能
1. ✅ 查询分解：将复杂查询拆分为原子级子问题
2. ✅ 多跳检测：自动识别单跳/多跳查询
3. ✅ 日志记录：完整的分解过程日志
4. ✅ 统计分析：查询复杂度统计
5. ✅ 向后兼容：可选开关，不影响原有功能

---

## 🚀 服务器运行步骤

### 步骤 0：Push 代码到服务器

```bash
# 在本地 T-RAG 目录
cd /Users/sunyifei/Documents/GitHub/T-RAG

# 添加所有新文件
git add src/downstream_inference/adaptive_modules/
git add src/downstream_inference/call_llm_v1.py
git add src/downstream_inference/run_v1_comparison.sh
git add src/downstream_inference/test_v1_module.py
git add src/downstream_inference/V1_README.md
git add V1_QUICK_START.md
git add ADAPTIVE_PLANNING_INTEGRATION.md

# 提交
git commit -m "feat: Add V1 query decomposition module

- Implement adaptive_modules with query decomposer
- Add table-specific prompts for decomposition
- Create call_llm_v1.py with decomposition support
- Add comparison script and documentation
- Backward compatible with --use_decomposition flag"

# 推送到远程
git push
```

### 步骤 1：在服务器上配置

```bash
# SSH 到服务器
ssh your-server

# 拉取最新代码
cd /path/to/T-RAG
git pull

# 激活环境
conda activate trag

# 配置 API Key
cd src/downstream_inference
vim key.json
```

在 `key.json` 中填入：
```json
{
    "openai": "sk-your-actual-openai-api-key",
    "claude": "<YOUR_CLAUDE_API_KEY>"
}
```

### 步骤 2：准备数据（如果还没有）

```bash
cd /path/to/T-RAG/src/table2graph

# 下载并处理数据集
bash scripts/prepare_data.sh

# 等待数据下载完成...
# 这会下载 MultiTableQA 的所有数据集（包括 SQA）
```

### 步骤 3：运行检索流程（如果还没有）

```bash
cd /path/to/T-RAG/src/table2graph

# 步骤 3.1: 表格聚类（Stage 1&2）
bash scripts/table_cluster_run.sh

# 步骤 3.2: 子图检索（Stage 3）
python scripts/subgraph_retrieve_run.py

# 检查输出文件是否生成
ls -lh data/sqa/sqa_retrieved_tables_schema_100_50_contriever.jsonl
# 应该看到这个文件存在
```

### 步骤 4：运行 V1 对比实验

```bash
cd /path/to/T-RAG/src/downstream_inference

# 方法 1：使用自动化脚本（推荐）
bash run_v1_comparison.sh

# 方法 2：手动运行对比实验

# 实验 1：基线（不使用分解）
python call_llm_v1.py \
    --dataset sqa \
    --topk 50 \
    --model gpt-4o-mini \
    --testing_num 100 \
    --embedding_method contriever

# 实验 2：使用分解（V1 功能）
python call_llm_v1.py \
    --dataset sqa \
    --topk 50 \
    --model gpt-4o-mini \
    --testing_num 100 \
    --embedding_method contriever \
    --use_decomposition \
    --decomposition_verbose
```

### 步骤 5：查看结果

```bash
cd output/sqa/gpt-4o-mini/

# 查看分解统计
cat decomposition_stats_100_50.json

# 查看分解日志（前10个）
head -n 10 decomposition_log_100_50.jsonl | python -m json.tool

# 查看推理结果
head -n 5 output_100_50_v1_decomp.jsonl | python -m json.tool
```

---

## ⚠️ 重要注意事项

### 1. 数据集规模选择

由于你要和论文对比，需要使用**完整测试集**：

**修改测试集大小的位置**：

```bash
# 在 subgraph_retrieve_run.py 中修改
testing_num = 100  # 改为你需要的数量

# 或者在运行时修改：
cd table2graph
vim scripts/subgraph_retrieve_run.py
# 找到 testing_num = 100，改为你想要的数量
```

**各数据集的完整规模**：
- SQA: ~8,000+ 测试样本
- HybridQA: ~2,000+ 测试样本
- WTQ: ~4,000+ 测试样本
- TabFact: ~12,000+ 测试样本

**建议**：
- 先用 `testing_num=100` 快速验证 V1 功能正常
- 确认无误后，再运行完整数据集

### 2. API 成本估算

使用 GPT-4o-mini 的成本：
- 每个查询 ≈ 2 次 LLM 调用（原始推理 + 分解）
- 100 个样本 ≈ $0.50 - $1.00
- 8,000 个样本（完整 SQA）≈ $40 - $80

**建议**：
1. 先用小规模（100个）验证
2. 确认效果后再跑完整数据集
3. 考虑使用 `gpt-4o-mini`（更便宜）而非 `gpt-4o`

### 3. 文件路径检查

**确保这个文件存在**：
```bash
# 检索结果文件（由 table2graph 生成）
../table2graph/data/sqa/sqa_retrieved_tables_schema_100_50_contriever.jsonl
```

如果不存在，运行会报错：
```
❌ ERROR: Retrieved tables file not found!
```

**解决方法**：先运行步骤 3 的检索流程。

### 4. 输出文件命名

V1 的输出文件有特殊后缀：
- 基线（无分解）：`output_100_50_v1_baseline.jsonl`
- 分解版本：`output_100_50_v1_decomp.jsonl`

**这样可以避免覆盖原始 T-RAG 的结果**，方便对比。

### 5. 模块导入问题

如果遇到导入错误：
```python
ModuleNotFoundError: No module named 'adaptive_modules'
```

**解决方法**：确保在 `src/downstream_inference` 目录下运行：
```bash
cd /path/to/T-RAG/src/downstream_inference
python call_llm_v1.py ...
```

---

## 📊 预期输出示例

### 成功运行后会看到：

#### 1. 控制台输出
```
======================================================================
T-RAG with Adaptive Planning - Version 1
======================================================================
Dataset: sqa
Model: gpt-4o-mini
Top-K: 50
Testing samples: 100
Embedding method: contriever
V1 Decomposition: ENABLED
======================================================================

Loading retrieved tables from: ../table2graph/data/sqa/sqa_retrieved_tables_schema_100_50_contriever.jsonl
Loaded 100 instances

Processing queries...
Inference: 100%|████████████████████| 100/100 [05:23<00:00,  3.23s/it]

Saving results to: output/sqa/gpt-4o-mini/output_100_50_v1_decomp.jsonl
Saving decomposition log to: output/sqa/gpt-4o-mini/decomposition_log_100_50.jsonl

======================================================================
V1 DECOMPOSITION STATISTICS
======================================================================
Total queries: 100
Multi-hop queries: 45 (45.0%)
Single-hop queries: 55 (55.0%)
Average requirements per query: 1.65
Decomposition failures: 0
======================================================================

✅ Inference complete!

Next step: Run evaluation
  python evaluation.py --dataset sqa --model gpt-4o-mini --topk 50 --testing_num 100
```

#### 2. 分解日志示例
```bash
cat decomposition_log_100_50.jsonl | head -n 1 | python -m json.tool
```

输出：
```json
{
  "query_idx": 0,
  "query": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
  "decomposition": {
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
  },
  "is_multi_hop": true,
  "num_requirements": 2
}
```

#### 3. 统计文件
```bash
cat decomposition_stats_100_50.json
```

输出：
```json
{
  "total_queries": 100,
  "multi_hop_queries": 45,
  "single_hop_queries": 55,
  "avg_requirements": 1.65,
  "decomposition_failures": 0
}
```

---

## 🔍 如何验证 V1 正确运行

### 检查清单：

1. ✅ **分解日志文件存在**
   ```bash
   ls -lh output/sqa/gpt-4o-mini/decomposition_log_100_50.jsonl
   # 应该有内容（不是空文件）
   ```

2. ✅ **分解统计正常**
   ```bash
   cat output/sqa/gpt-4o-mini/decomposition_stats_100_50.json
   # 应该看到 multi_hop_queries > 0
   ```

3. ✅ **输出文件包含分解信息**
   ```bash
   head -n 1 output/sqa/gpt-4o-mini/output_100_50_v1_decomp.jsonl | python -m json.tool
   # 应该看到 "decomposition" 字段
   ```

4. ✅ **无错误信息**
   ```bash
   # 控制台不应该有大量 ERROR 或 WARNING
   # 允许个别查询的 warning，但不应该全部失败
   ```

---

## 🐛 常见问题排查

### 问题 1：JSON 解析失败
```
[WARNING] Decomposition failed for query X: Expecting value: line 1 column 1 (char 0)
```

**可能原因**：
- LLM 没有返回 JSON
- API 调用失败

**解决方法**：
- 检查 API Key 是否正确
- 查看 LLM 返回的原始文本（打开 `--decomposition_verbose`）
- 可能需要调整 prompt（在 `adaptive_modules/prompts.py`）

### 问题 2：所有查询都是单跳
```
Multi-hop queries: 0 (0.0%)
```

**可能原因**：
- Prompt 不够明确
- 数据集本身确实是单跳为主
- LLM 没有理解分解任务

**解决方法**：
- 手动检查几个分解结果，看是否合理
- 如果是 SQA 数据集，应该有一定比例的多跳查询

### 问题 3：分解日志为空
```
decomposition_log_100_50.jsonl 文件不存在
```

**原因**：没有启用 `--use_decomposition`

**解决方法**：
```bash
# 确保添加这个参数
python call_llm_v1.py ... --use_decomposition
```

---

## 📈 V1 的评估和对比

### 运行评估

```bash
cd /path/to/T-RAG/src/downstream_inference

# 评估基线
python evaluation.py \
    --dataset sqa \
    --model gpt-4o-mini \
    --topk 50 \
    --testing_num 100

# 评估分解版本（相同命令，会自动找对应的输出文件）
python evaluation.py \
    --dataset sqa \
    --model gpt-4o-mini \
    --topk 50 \
    --testing_num 100
```

### V1 预期结果

**重要**：V1 阶段的 EM/F1 应该与基线**基本相同**。

为什么？
- V1 只做分解和记录，**不改变推理流程**
- 实际推理仍然是原始 T-RAG 的单次推理
- 分解信息仅用于日志和分析

**V1 的目标**：
1. ✅ 验证分解逻辑正确
2. ✅ 分析数据集中多跳查询的比例
3. ✅ 为 V2/V3/V4 打好基础

**真正的提升会在**：
- V2：事实提取评估（+1-2% EM）
- V3：重规划能力（+2-3% EM）
- V4：完整迭代循环（+3-5% EM）

---

## 🎯 下一步行动

### 完成 V1 后：

1. **检查分解质量**
   ```bash
   # 随机查看 10 个分解结果
   shuf -n 10 decomposition_log_100_50.jsonl | python -m json.tool
   ```

2. **分析多跳查询**
   ```python
   import json

   with open('decomposition_log_100_50.jsonl') as f:
       logs = [json.loads(line) for line in f]

   multi_hop = [log for log in logs if log['is_multi_hop']]
   print(f"Multi-hop: {len(multi_hop)}/{len(logs)}")

   # 查看几个例子
   for log in multi_hop[:3]:
       print(f"\nQuery: {log['query']}")
       for req in log['decomposition']['requirements']:
           print(f"  {req['requirement_id']}: {req['question']}")
   ```

3. **准备 V2 开发**
   - 如果分解质量好 → 可以开始 V2（事实提取）
   - 如果分解质量差 → 需要调优 Prompt

---

## 📞 需要帮助？

如果遇到问题，检查：

1. **日志文件**：`logs/sqa/*.log`
2. **错误输出**：控制台的 ERROR/WARNING 信息
3. **API 调用**：是否有 rate limit 或 quota 错误

**准备好继续 V2 了吗？** 告诉我 V1 的运行结果，我会帮你开始 V2 的开发！
