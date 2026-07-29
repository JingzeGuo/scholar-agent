# ScholarAgent 项目状态与 Agent 岗位面试要点

## 结论

**可以作为 Agent/RAG 开发方向的毕设和初级岗位面试主项目。**

它目前不是只有调用大模型的 Demo，而是一个完成了“语料处理 → 多路检索 →
Agent 决策 → 证据验证 → 引用生成 → 消融评测 → 离线演示 → 工程测试”闭环的
研究型工程原型。`CODEX_IMPLEMENTATION_PLAN.md` 的核心 Phase 0–10 已全部完成，
逐项证据见 [`phase_acceptance.md`](phase_acceptance.md)。

面试定位应当是：

> 面向学术论文问答的、证据驱动的 Multi-Agent GraphRAG 研究原型，重点解决
> 检索策略选择、纠错循环、知识图谱溯源和页级引用可信度问题，并用统一冻结
> 数据集验证每个模块到底有没有帮助。

不要称它为生产级 SaaS、多租户知识库平台或已经全面优于普通 RAG 的系统。

## 当前项目状态

| 维度 | 当前状态 |
|---|---|
| 核心功能 | Phase 0–10 全部实现并验收 |
| 语料 | 本地完成 120 篇论文、2593 页、5858 个 chunks 的处理 |
| 检索 | Dense、BM25、RRF、Cross-Encoder、Graph、纠错检索均可独立测试 |
| Agent | Planner、Researcher、Verifier、Writer 和 Citation Validator 完整串联 |
| 图谱 | 2636 个节点、4263 条边；4263/4263 关系均可回溯到 PDF 证据 |
| 引用 | Claim → Evidence → Chunk → Paper → PDF page 全链路验证 |
| 评测 | 冻结 50 题，7 个系统，共 350 个 system-question 组合 |
| 模型实测 | Hash、BGE、Cross-Encoder、DeepSeek、RAGAS 路径均实际运行过 |
| 演示 | Streamlit + 无 API 的 Saved Replay + 可重复生成的 GIF |
| 工程质量 | Ruff、Mypy 通过；fresh clone 252 通过/6 跳过，本地全量 258 通过 |
| 当前限制 | 单机研究原型；纠错检索未带来召回提升；50 题不是人工签字标注 |

## 一分钟项目介绍

可以直接按下面的版本讲：

> ScholarAgent 是一个面向论文研究的多 Agent GraphRAG 系统。普通 RAG 的问题是
> 固定使用一种检索方式，而且生成结果经常无法准确定位到原论文页面。我的系统
> 先把 120 篇论文处理成带稳定 ID 和页码的 canonical chunk store，同时建立
> Dense、BM25 和带证据跨度的知识图谱。查询进入后，Planner 将复杂问题拆成
> 结构化子问题，Researcher 根据问题类型选择检索工具，Verifier 检查证据覆盖和
> 冲突，并在缺失时发起有预算限制的纠错检索，Writer 只能使用验证过的证据，
> 最后 Citation Validator 检查每条引用是否真的映射到 PDF 页面。我还冻结了
> 50 道题，对 7 种系统配置做消融。确定性 Hash 实验中，Hybrid 将 paper
> Recall@8 从 0.16 提升到 0.61，Graph 提升到 0.67；同时我也诚实记录了 Graph
> 延迟较高、纠错循环召回提升为 0.0 等负面结果。

## 系统主链路

```text
PDF → 页面/章节解析 → 稳定 Chunk ID → Canonical Chunk Store
                                    ├─ Dense/BGE index
                                    ├─ BM25 index
                                    └─ Provenance Graph

Question → Planner → Researcher/Router → Evidence Ledger
                                      ↓
                Writer ← Verifier ← coverage/conflict check
                   ↓          ↘ evidence gap: corrective retrieval
          Citation Validator → page-grounded answer + trace
```

## 最值得讲的五个技术点

### 1. 不是固定 Chain，而是有条件决策的 Agent 工作流

- Planner 输出 Pydantic `QueryPlan`，不是一段不可验证的自由文本。
- Researcher 根据 semantic、keyword、comparison、relational 等类型选择工具。
- Verifier 可以决定继续检索、拒答或进入写作阶段。
- 工具调用、迭代、token、延迟和证据数量都有硬预算。
- 没有新证据、预算耗尽或证据充分都会终止流程。

这比“按顺序调用多个 LLM”更能说明真正的 Agent 状态管理和控制流能力。

### 2. Canonical Chunk Store 保证多索引一致性

Dense、BM25 和 Graph 都使用相同的稳定 `chunk_id`，避免不同索引返回无法对齐的
文档。内容、页码或配置改变时可以通过 fingerprint 检测不一致。这是项目中很适合
讲工程设计的一点。

### 3. GraphRAG 中的每条边都有证据

图谱不把 LLM 抽取出的 triple 当作事实。每条 relation 必须保存 `chunk_id` 和
`evidence_span`，并通过 canonical store 回到真实 PDF 页。全量审计中 4263/4263
条关系通过证据映射检查。

### 4. Writer 与 Verifier 分离

Researcher 负责尽量找证据，Verifier 负责判断证据是否覆盖问题，Writer 只能消费
Verifier 接受的 Evidence Ledger。这样可以测试和限制“检索到错误材料后照样生成”
的问题，Citation Validator 还会删除不存在、页码错误或无法支撑 claim 的引用。

### 5. 有真实消融，而不是只展示成功案例

冻结 50 道题后，所有 7 种系统都运行同一份数据和指标。报告包含检索、答案、引用、
拒答、延迟、成本和 Agent 操作指标，并保留失败案例。这让项目更像毕设研究和真实
工程实验，而不是界面演示。

## 可以使用的实测数据

### 确定性离线 Hash 检索

| 系统 | Paper Recall@8 | Citation Precision | 平均延迟 |
|---|---:|---:|---:|
| Naive Dense | 0.16 | 0.038 | 3.6 ms |
| Hybrid + Rerank | 0.61 | 0.166 | 10.7 ms |
| Hybrid + Graph | **0.67** | 0.212 | 289.9 ms |
| Full Agent | 0.54 | **0.288** | 572.2 ms |

正确解读：Hybrid 是最大召回增益；Graph 进一步提高论文召回但延迟明显增加；
Full Agent 的引用精度和拒答更好，但总召回并不是最高。

### BGE + Cross-Encoder

- BGE dense-only paper Recall@8：**0.70**。
- static-all-tools 最佳 paper Recall@8：**0.73**。
- 说明 Hash 实验适合确定性回归，但不能代替生产 embedding 质量结论。

### Live DeepSeek + RAGAS

- 50 题 × 7 系统，共 350 行，生成调用 350/350，系统错误 0。
- Hash + Graph 的 RAGAS faithfulness：**0.959**。
- BGE/CE + Full Agent 的 RAGAS faithfulness：**0.965**。
- RAGAS 覆盖率 0.90，因为 5 道不可回答题在 7 个系统中产生空拒答，未强行计为 0。

## 三个很好的失败案例

1. **Dense 漏掉精确术语，BM25/Hybrid 找回。** 可以讲为什么向量检索不能完全
   替代关键词检索，以及为什么使用 RRF 而不是直接混合未校准分数。
2. **Graph 提升召回但增加约 27 倍检索延迟。** 可以讲选择性路由、Graph fan-out
   和“不是所有问题都应该走 GraphRAG”。
3. **纠错循环能触发并安全结束，但 recall improvement = 0.0。** 可以说明控制流
   正确不等于算法有效，下一步应改进 gap-to-query 映射、证据预算和检索模型。

面试中主动讲一个负面结果，通常比声称所有 Agent 模块都有效更可信。

## 五分钟演示顺序

1. 用 README 的图解释离线索引和在线 Agent 流程。
2. 在 Streamlit Replay 中选择 `selfrag_vs_crag`，无需 API。
3. 展示 Planner 子问题和 Router 选择的工具。
4. 展示 Evidence Ledger、Verifier 缺口和 corrective iteration。
5. 点击 Source Card，说明 claim 如何映射到 chunk 和 PDF page。
6. 展示 Naive/Hybrid/Graph/Full Agent 消融表。
7. 用纠错提升为 0.0 或 Graph 延迟结束，说明下一步优化计划。

如果现场环境出问题，可直接使用 committed GIF 和 saved replay；这也是工程可靠性
设计的一部分，但要说明 GIF 是离线回放，不是假装成实时运行。

## 高频追问与回答重点

| 问题 | 回答重点 |
|---|---|
| 为什么用 LangGraph？ | 有条件边、循环、预算、状态 reducer 和明确终止原因，不只是线性 chain。 |
| 什么地方是真正的 Agent？ | 工具选择、证据充分性判断、纠错决策和终止决策会随状态变化。 |
| 为什么需要 BM25？ | 精确名词、缩写和稀有标识符；实测 Hybrid 明显优于 Hash dense。 |
| 为什么使用 RRF？ | 不要求 Dense 与 BM25 分数在同一尺度，融合逻辑明确、可测试。 |
| 如何防止无限循环？ | corrective iteration、tool、token、latency、no-new-evidence 多重停止条件。 |
| 如何避免假引用？ | Writer 只能使用 ledger ID；validator 检查 canonical chunk、PDF 路径和物理页码。 |
| Graph 是否一定有效？ | 否。总召回有提升，但延迟和个别 slot displacement 是明确代价。 |
| 为什么 Full Agent recall 更低？ | 子问题预算和 ledger 排序提高部分比较题质量，但可能丢失简单题的 gold chunk。 |
| 怎么做线上化？ | 增加 API/Auth、多租户索引隔离、任务队列、托管向量库、可观测平台和限流。 |
| 如何防 Prompt Injection？ | 把论文内容视为不可信输入，使用边界标记，禁止材料触发工具调用，并有回归测试。 |

## 简历描述参考

可以根据简历空间选择三到四条：

- 独立实现面向 120 篇论文的 Multi-Agent GraphRAG 系统，使用 LangGraph 构建
  Planner–Researcher–Verifier–Writer 条件工作流，并实现工具、迭代、token 和延迟预算。
- 构建 BGE Dense + BM25 + RRF + Cross-Encoder + provenance Graph 多路检索，
  通过稳定 Chunk ID 和 corpus fingerprint 保证 5858 个 chunks 的跨索引一致性。
- 实现 claim-to-evidence 引用管线，4263/4263 图谱关系可回溯至原始 PDF 证据，
  并验证引用对应的物理页码。
- 设计冻结 50 题 × 7 系统的消融评测；Hash 实验中将 paper Recall@8 从 0.16
  提升到 0.67，并完成 BGE、Cross-Encoder、DeepSeek 和 RAGAS 实测。
- 建立 258 个离线测试、live/provider 测试隔离、结构化无密钥日志、缓存、重试和
  Saved Replay；fresh clone 中 252 个通过、6 个可选资产测试跳过，默认测试与
  面试 Demo 均不依赖付费 API。

## 必须如实说明的边界

- 它是单机研究原型，不具备生产系统的鉴权、多租户、水平扩容和 SLA。
- 50 题经过 AI 辅助独立复核，但不是领域专家的人工签字标注。
- 纠错循环的控制和终止已经实现，但当前实验没有带来 gold recall 提升。
- Graph 使用约束 ontology 和启发式抽取，不应宣称为完整、无噪声知识库。
- PDF、模型和索引因体积与许可原因不提交；仓库克隆后可运行 fixture 和 saved replay，
  全量实验需要按文档重新下载和构建本地资产。
- RAGAS 是辅助自动评估，不能代替专家对答案事实性的最终判断。

## 面试前最后检查

```bash
make quality
uv run scholar-agent demo --replay selfrag_vs_crag
uv run scholar-agent demo --replay unanswerable_market
```

同时准备：

- 能在代码中指出 Planner、Router、Researcher、Verifier、Writer 和 reducer 的位置。
- 能手写或解释 RRF 公式、Recall@K、MRR、Citation Precision。
- 能解释为什么 0.67 recall 不代表 67% 的答案完全正确。
- 能讲清一个成功案例和一个失败案例。
- 能回答“如果再给两周，优先改什么”：先改善 corrective query 质量和 graph fusion，
  再扩大人工标注开发/测试集，最后补多租户 API 与线上观测。

## 相关材料

- [`phase_acceptance.md`](phase_acceptance.md)：Phase 0–10 客观验收证据
- [`interview_guide.md`](interview_guide.md)：英文详细问答
- [`failure_analysis.md`](failure_analysis.md)：六个真实失败分析
- [`demo_script.md`](demo_script.md)：5–7 分钟演示脚本
- [`evaluation.md`](evaluation.md)：评测方法与运行方式
- [`results/`](results/)：Hash、BGE/CE、DeepSeek/RAGAS 数值快照
