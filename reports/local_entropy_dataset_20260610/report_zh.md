# 局部语义熵删边数据集汇报稿（当前版本）

## 1. 一句话概括
我们已经构建了一个基于 GDesigner 训练后拓扑的“删边反事实数据集”：对训练阶段使用的题目，逐条删除图中的通信边，只重跑这条边两端相关 agent 的局部输出，并用多次采样后的语义熵作为该边的软贡献标签。这个数据集的作用不是直接作为最终准确率，而是训练一个 explainer/post-processor，让它学习哪些边被删除后会导致通信不稳定，从而用于后续图剪枝。

## 2. 数据怎么来的
- 图生成器：当前先选用 `GDesigner`，避免同时混入多个生成器导致变量太多。
- 题目范围：使用各方法训练/调图阶段对应的训练子集；MMLU 使用 AgentPrune 对齐的 153 子集中的训练部分/分片结果；HumanEval 当前仍在继续跑，因此目前是 partial。
- 反事实操作：对每个样本图中的一条 active edge 执行 `drop_edge`，记录被删边的 source/target、agent role、round、局部 trace。
- 局部采样：每条删边样本采样 5 次，并行采样，记录 source agent 输出、target agent 输出和 pair 输出。
- 软标签：对 5 次输出做简单语义聚类，计算熵并归一化：

```text
H_norm(x) = H(x) / log(5)
Q(e) = 0.5 * H_norm(pair) + 0.25 * H_norm(source) + 0.25 * H_norm(target)
```

这里 `Q(e)` 越大，表示删掉边 `e` 后，该边两端局部通信输出越不稳定，说明这条边更可能是重要边；`Q(e)=0` 通常表示删边后 5 次输出都落在同一语义簇里，局部影响很弱。

## 3. 当前数据规模

| Dataset | Rows(edge counterfactuals) | Unique tasks | Spatial | Temporal | Nonzero labels | Mean Q | P90 Q | Max Q | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 240 | 40 | 240 | 0 | 75 (31.2%) | 0.058 | 0.233 | 0.510 | 5,074,010 |
| multiarith | 240 | 40 | 240 | 0 | 52 (21.7%) | 0.027 | 0.105 | 0.373 | 4,461,551 |
| svamp | 240 | 40 | 240 | 0 | 51 (21.2%) | 0.035 | 0.105 | 0.443 | 4,423,870 |
| aqua | 240 | 40 | 240 | 0 | 136 (56.7%) | 0.094 | 0.242 | 0.639 | 3,550,671 |
| mmlu | 184 | 40 | 184 | 0 | 44 (23.9%) | 0.027 | 0.105 | 0.314 | 2,934,609 |
| humaneval | 1013 | 28 | 336 | 677 | 457 (45.1%) | 0.089 | 0.311 | 1.000 | 15,295,592 |
| **Total** | **2157** | **228** |  |  | **815 (37.8%)** | **0.068** |  | **1.000** | **35,740,303** |

备注：HumanEval 的数据仍在 `gdesigner_local_entropy_humaneval_gpu2` 中继续生成；当前统计是读取已有 JSONL 的快照。

## 4. 初步可训练性
- 已用当前缓存训练过一个 trial explainer：样本数 2155，训练/验证划分 1724/431。
- 验证集 Pearson=0.606，Spearman=0.542，MSE=0.0128。
- 这说明当前 `Q(e)` 不是完全随机噪声，边的结构特征、端点角色表示和局部图特征能中等程度预测删边后的不稳定性。

## 5. 可以展示给导师的图
- 数据规模和平均标签：[dataset_overview.svg](dataset_overview.svg)
- 标签分布：[label_distribution.svg](label_distribution.svg)
- Explainer 可训练性：[explainer_learnability.svg](explainer_learnability.svg)

## 6. 两个直观例子

### 例子 1: humaneval / eval:HumanEval_160_do_algebra / label=1.000
- 被删边: `temporal` `mWqx -> 4g8J`，edge_index=15
- 端点角色: `Test Analyst` -> `Project Manager`
- 熵: source=1.609, target=1.609, pair=1.609; samples=5
- token: total=47286, mean/sample=9457.2
- 任务: from typing import List def do_algebra(operator: List[str], operand: List[int]) -> int: """ Given two lists operator, and operand. The first list has basic algebra operations, and the second list is a list of integers. Use the two given lists to build the algebric expression and return the evaluation of this expression. The basic algebra operations: Addition ( + ) Subtraction ( - ) Multiplication ( * ) Floor division...
- 聚类: source={'cluster_0': 1, 'cluster_1': 1, 'cluster_2': 1, 'cluster_3': 1, 'cluster_4': 1}, target={'cluster_0': 1, 'cluster_1': 1, 'cluster_2': 1, 'cluster_3': 1, 'cluster_4': 1}, pair={'cluster_0': 1, 'cluster_1': 1, 'cluster_2': 1, 'cluster_3': 1, 'cluster_4': 1}
- 一次 source 输出摘要: The code uses `eval`, which is unsafe and not recommended. It also does not handle operator precedence correctly as per the note. Special cases like division by zero or large exponents may cause errors.
- 一次 target 输出摘要: Use a loop to apply operators sequentially to operands, respecting precedence without `eval`. Handle each operator explicitly. Ensure division by zero is checked.

### 例子 2: humaneval / eval:HumanEval_23_strlen / label=0.000
- 被删边: `spatial` `3ehr -> vVK7`，edge_index=7
- 端点角色: `Algorithm Designer` -> `Programming Expert`
- 熵: source=-0.000, target=-0.000, pair=-0.000; samples=5
- token: total=3557, mean/sample=711.4
- 任务: def strlen(string: str) -> int: """ Return length of given string >>> strlen('') 0 >>> strlen('abc') 3 """
- 聚类: source={'cluster_0': 5}, target={'cluster_0': 5}, pair={'cluster_0': 5}
- 一次 source 输出摘要: ```python def strlen(string: str) -> int: """ Return length of given string >>> strlen('') 0 >>> strlen('abc') 3 """ return len(string) ```
- 一次 target 输出摘要: ```python def strlen(string: str) -> int: """ Return length of given string >>> strlen('') 0 >>> strlen('abc') 3 """ return len(string) ```

## 7. 汇报时建议怎么说
- 这不是最终模型结果，而是为了训练后处理剪枝器构造的“边级因果/反事实监督数据”。
- 目前先用 GDesigner 作为单一图生成器，控制变量；后面如果这个 explainer 在真实剪枝评测中有效，再加入 AgentPrune/AgentDropout 的图，做跨生成器泛化。
- 标签选择从 0/1 正确率改为局部语义熵，是为了避免单题正确/错误信号过稀疏；局部熵能更细粒度地反映删边后通信是否变得不稳定。
- 当前风险是：局部不稳定不一定等价于最终答案变差，所以下一步必须做 real pruning evaluation，验证用 explainer 剪出来的子图是否能保持/提升最终准确率并降低 token/边数。

## 8. 文件位置
- 汇总 CSV: `/home/ljz/llm-mas/CausalCommunicationRefiner/reports/local_entropy_dataset_20260610/dataset_summary.csv`
- 汇总 JSON: `/home/ljz/llm-mas/CausalCommunicationRefiner/reports/local_entropy_dataset_20260610/dataset_summary.json`
- 样例详情: `/home/ljz/llm-mas/CausalCommunicationRefiner/reports/local_entropy_dataset_20260610/examples.md`
- Trial explainer: `/dev/shm/ccr_local_entropy_explainer_trial/20260610_071843/local_entropy_explainer.pt`
