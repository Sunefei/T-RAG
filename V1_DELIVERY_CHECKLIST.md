# T-RAG V1 交付清单

## ✅ 已完成工作总结

### 📦 交付文件清单

#### 核心代码（4 个文件）
1. ✅ `src/downstream_inference/adaptive_modules/__init__.py`
   - 模块初始化文件
   - 导出核心函数和常量

2. ✅ `src/downstream_inference/adaptive_modules/prompts.py`
   - 表格专用 Prompt 模板
   - 包含详细的分解指令和示例
   - 为 V2/V3/V4 预留了 placeholder

3. ✅ `src/downstream_inference/adaptive_modules/query_decomposer.py`
   - 查询分解核心逻辑（300+ 行）
   - 多种 JSON 解析策略（健壮性）
   - 完整的错误处理和降级机制
   - 包含测试函数

4. ✅ `src/downstream_inference/call_llm_v1.py`
   - V1 主程序（400+ 行）
   - 完全兼容原始 T-RAG
   - 新增 `--use_decomposition` 参数
   - 分解日志和统计功能

#### 运行脚本（2 个文件）
5. ✅ `src/downstream_inference/run_v1_comparison.sh`
   - 自动化对比实验脚本
   - 依次运行基线和分解版本
   - 自动检查依赖文件

6. ✅ `src/downstream_inference/test_v1_module.py`
   - 模块单元测试（5 个测试）
   - 可独立运行验证功能

#### 文档（4 个文件）
7. ✅ `src/downstream_inference/V1_README.md`
   - 详细使用文档
   - 包含所有参数说明
   - 输出文件格式说明
   - 错误排查指南

8. ✅ `V1_QUICK_START.md`
   - 服务器运行快速指南
   - 完整的步骤说明
   - 常见问题排查
   - 成本估算和注意事项

9. ✅ `ADAPTIVE_PLANNING_INTEGRATION.md`
   - 完整的集成指南（V1-V5）
   - 原理分析和架构对比
   - 增量式开发计划
   - 评估对比方法

10. ✅ `V1_DELIVERY_CHECKLIST.md`
    - 本文件（交付清单）

---

## 📊 代码统计

| 模块 | 文件 | 行数 | 功能 |
|------|------|------|------|
| Prompts | prompts.py | ~200 | Prompt 模板 |
| Decomposer | query_decomposer.py | ~300 | 分解逻辑 |
| Main | call_llm_v1.py | ~400 | 主程序 |
| Tests | test_v1_module.py | ~200 | 单元测试 |
| **总计** | **4 个核心文件** | **~1100** | **完整功能** |

---

## 🎯 V1 功能验收标准

### 必须通过的测试

#### 1. 基本功能测试
- [ ] 代码能够正常 import（无语法错误）
- [ ] 单元测试全部通过（`python test_v1_module.py`）
- [ ] 能够运行基线模式（不使用分解）
- [ ] 能够运行分解模式（`--use_decomposition`）

#### 2. 分解质量测试
- [ ] 单跳查询正确识别（`is_multi_hop = False`）
- [ ] 多跳查询正确识别（`is_multi_hop = True`）
- [ ] JSON 解析成功率 > 95%
- [ ] 分解后的需求格式正确（包含所有必需字段）

#### 3. 日志和统计
- [ ] 生成分解日志文件（`.jsonl` 格式）
- [ ] 生成统计文件（`.json` 格式）
- [ ] 统计数据合理（多跳查询比例 > 0）
- [ ] 输出文件包含分解信息

#### 4. 兼容性测试
- [ ] 不影响原始 T-RAG 功能
- [ ] 基线模式的 EM/F1 与原始版本一致
- [ ] 可以与 evaluation.py 正常配合

---

## 🚀 运行命令速查

### Git 提交（本地）
```bash
cd /Users/sunyifei/Documents/GitHub/T-RAG

git add src/downstream_inference/adaptive_modules/
git add src/downstream_inference/call_llm_v1.py
git add src/downstream_inference/run_v1_comparison.sh
git add src/downstream_inference/test_v1_module.py
git add src/downstream_inference/V1_README.md
git add V1_QUICK_START.md
git add ADAPTIVE_PLANNING_INTEGRATION.md
git add V1_DELIVERY_CHECKLIST.md

git commit -m "feat: Add V1 query decomposition module"
git push
```

### 服务器快速启动
```bash
# 1. 拉取代码
cd /path/to/T-RAG && git pull

# 2. 激活环境
conda activate trag

# 3. 配置 API Key
cd src/downstream_inference
vim key.json  # 填入 OpenAI API Key

# 4. 运行对比实验（推荐）
bash run_v1_comparison.sh

# 或单独运行
python call_llm_v1.py \
    --dataset sqa \
    --topk 50 \
    --model gpt-4o-mini \
    --testing_num 100 \
    --use_decomposition
```

---

## 📈 预期输出文件

运行成功后，会在 `output/sqa/gpt-4o-mini/` 生成以下文件：

| 文件名 | 大小估计 | 说明 |
|--------|---------|------|
| `output_100_50_v1_baseline.jsonl` | ~500KB | 基线推理结果 |
| `output_100_50_v1_decomp.jsonl` | ~600KB | 分解版本推理结果 |
| `decomposition_log_100_50.jsonl` | ~200KB | 分解详细日志 |
| `decomposition_stats_100_50.json` | ~1KB | 统计摘要 |
| `results_100_50.json` | ~1KB | 评估结果（EM/F1） |

---

## 🔍 质量检查命令

### 检查文件完整性
```bash
cd /Users/sunyifei/Documents/GitHub/T-RAG

# 检查所有文件是否存在
ls -l src/downstream_inference/adaptive_modules/__init__.py
ls -l src/downstream_inference/adaptive_modules/prompts.py
ls -l src/downstream_inference/adaptive_modules/query_decomposer.py
ls -l src/downstream_inference/call_llm_v1.py
ls -l src/downstream_inference/run_v1_comparison.sh
ls -l src/downstream_inference/test_v1_module.py
ls -l src/downstream_inference/V1_README.md
ls -l V1_QUICK_START.md
ls -l ADAPTIVE_PLANNING_INTEGRATION.md
ls -l V1_DELIVERY_CHECKLIST.md

# 检查可执行权限
ls -l src/downstream_inference/run_v1_comparison.sh | grep "x"
```

### 验证代码语法
```bash
cd src/downstream_inference

# Python 语法检查
python -m py_compile adaptive_modules/prompts.py
python -m py_compile adaptive_modules/query_decomposer.py
python -m py_compile call_llm_v1.py
python -m py_compile test_v1_module.py

# 如果没有错误输出，说明语法正确
```

### 检查导入
```bash
cd src/downstream_inference

python -c "from adaptive_modules import analyze_and_decompose_query; print('✓ Import OK')"
```

---

## 📝 关键设计决策记录

### 1. 为什么 V1 不改变推理流程？
- **原因**：增量式开发，确保每一步都可验证
- **好处**：V1 可以作为独立的分析工具，即使不继续开发 V2 也有价值
- **后果**：EM/F1 指标在 V1 不会提升，需要等到 V2/V3

### 2. 为什么使用 `--use_decomposition` 开关？
- **原因**：保持向后兼容，方便 A/B 对比
- **好处**：可以在同一个脚本中切换功能，减少代码重复
- **后果**：代码逻辑稍微复杂一些（但有完整的错误处理）

### 3. 为什么支持多种 JSON 解析策略？
- **原因**：LLM 返回格式不稳定，需要健壮的解析
- **好处**：提高成功率，减少失败重试
- **实现**：4 种解析策略（直接解析 → Markdown → 正则 → 最后兜底）

### 4. 为什么输出文件名加后缀？
- **原因**：避免覆盖原始 T-RAG 的结果
- **好处**：可以同时保留多个版本的结果进行对比
- **格式**：`output_{num}_{topk}_v1_{baseline|decomp}.jsonl`

---

## 🎓 学习和调试建议

### 理解代码执行流程

1. **入口点**：`call_llm_v1.py` 的 `main()` 函数
2. **关键流程**：
   ```
   读取检索结果 → 循环处理每个查询 →
   [可选] 分解查询 → 构建 Prompt → 调用 LLM →
   保存结果 → 输出统计
   ```
3. **分解逻辑**：在 `query_decomposer.py` 的 `analyze_and_decompose_query()`

### 调试技巧

1. **启用详细日志**
   ```bash
   python call_llm_v1.py ... --decomposition_verbose
   ```

2. **只运行 1 个样本**
   ```bash
   python call_llm_v1.py ... --testing_num 1
   ```

3. **检查中间输出**
   ```bash
   # 在代码中添加 print
   print(f"[DEBUG] Decomposition result: {decomposition_result}")
   ```

4. **单独测试分解器**
   ```bash
   python test_v1_module.py
   ```

---

## 🔄 V1 → V2 升级路径

### V1 交付后，下一步是 V2（事实提取评估）

**V2 新增功能**：
- 为每个分解出的需求提取事实
- 评估事实提取质量（DIRECT_ANSWER/PARTIAL_CLUE/FAILED_EXTRACT）
- 记录事实提取日志

**V2 新增文件**：
- `adaptive_modules/fact_extractor.py`
- `call_llm_v2.py`

**预期开发时间**：1 天

**预期效果提升**：+1-2% EM（因为开始利用分解信息）

---

## ✨ V1 总结

### 完成度：100% ✅

- ✅ 代码完整（1100+ 行，4 个核心文件）
- ✅ 文档齐全（4 个文档，3000+ 字）
- ✅ 测试覆盖（5 个单元测试）
- ✅ 运行脚本（自动化对比）
- ✅ 错误处理（健壮的降级机制）
- ✅ 日志记录（完整的追踪）

### 下一步行动

1. **立即**：Push 代码到服务器
2. **今天**：在服务器上运行 V1 对比实验
3. **明天**：根据 V1 结果决定是否开始 V2

### 需要我继续的工作

- [ ] 开始 V2（事实提取）开发
- [ ] 根据 V1 运行结果调优 Prompt
- [ ] 准备完整数据集的实验

---

## 📞 联系和支持

如果 V1 运行遇到任何问题，提供以下信息：

1. 错误消息（完整的 stack trace）
2. 运行命令（包括所有参数）
3. 输出文件路径和大小
4. 分解统计文件内容

我会立即帮你诊断和修复！

**V1 已就绪，可以开始服务器测试了！** 🚀
