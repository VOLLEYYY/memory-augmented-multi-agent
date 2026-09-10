# ReAct 范式：推理 + 行动

ReAct = Reasoning（推理）+ Acting（行动），是目前最流行的 LLM Agent 通用架构。它让模型交替生成「思考」和「行动」，边推理边与外部环境交互。

## 核心思想

用一个公式表达：

```
agent ≈ model + tools，在一个 for-loop + environment 里运行
```

语言模型作为推理引擎，决定每一步「采取什么行动、以什么顺序」。相比传统的「给输入→出输出」的被动模式，Agent 强调自主地发现问题、确定目标、选择方案、执行、检查更新。

## 核心循环

Thought（思考）→ Action（行动）→ Observation（观察）→ 再思考 → … 直到完成：

```
用户问题
  → [Thought: 我需要搜索]
  → [Action: Search[华为手机]]
  → [Observation: 搜索结果...]
  → [Thought: 信息够了]
  → [Action: Finish[华为最新是 Mate 80]]
```

每一轮观察都把下一轮思考锚定在真实数据上，用真实数据替换模型自行编造的内容。

## 为什么需要 ReAct

纯 CoT（思维链）是「黑盒推理」，只用模型内部表示，不与外部世界交互，容易事实幻觉、错误传播。ReAct 补上了「与外部交互」这一环，让模型能查询实时信息、校验事实、根据中间结果调整策略。

## 与 CoT 的对比

| 方面 | CoT | ReAct |
|------|-----|-------|
| 核心 | 纯推理链 | 推理 + 行动交互 |
| 外部交互 | 无 | 支持工具调用 |
| 错误处理 | 内部修正 | 通过观察结果调整 |
| 结构 | 线性推理 | 循环决策 |
| 适用场景 | 纯推理问题 | 需要外部信息的任务 |

## 关键参数：max_steps

max_steps 限制循环迭代次数，防止无限循环。

- 设太小：Agent 还没做完就放弃了
- 设太大：Token 和预算一起烧，且迭代越多跑偏概率越大

建议从上限 5 次起步，根据线上数据再调。

## 伪代码

```
for i in range(max_steps):
    thought = think(context)            # 1. 思考下一步
    action = decide_action(thought)     # 2. 决定行动（调工具 or 结束）
    if action.type == "FINISH":
        return action.answer
    observation = execute(action)       # 3. 执行工具调用
    context += "观察：" + observation   # 4. 观察结果，进入下一轮
```

## 相关增强架构

- Plan-and-Execute：先规划子任务清单，再逐个执行，减少每步都问 LLM 的开销
- Reflection：让 Agent 反思和改进过去的决策，从错误中学习
