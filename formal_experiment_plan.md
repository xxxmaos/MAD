# 正式实验方案：异构轻量模型本地部署 + Summarizer + 早退机制

---

## 一、推荐的三种异构轻量模型

选择原则：三个模型来自不同公司/架构系列，参数量都在 7-8B 级别（单卡 24GB 可跑），均有 Instruct 版本（适合辩论场景），且在推理 benchmark 上各有优劣势（提供观点多样性）。

| 模型 | 公司 | 参数量 | 优势领域 | 劣势领域 | License |
|------|------|--------|---------|---------|---------|
| **Qwen2.5-7B-Instruct** | 阿里 | 7.6B | 数学推理、多语言、代码 | 英文常识略弱于 Llama | Apache 2.0 |
| **Llama-3.1-8B-Instruct** | Meta | 8B | 英文理解、指令跟随、长文本(128K) | 数学推理不如 Qwen | Llama Community |
| **Mistral-7B-Instruct-v0.3** | Mistral AI | 7.2B | 推理速度快、推理效率高 | 上下文窗口短(32K)、中文弱 | Apache 2.0 |

**为什么选这三个：**

第一，架构异构。Qwen 用的是改进的 Transformer + GQA，Llama 用 RoPE + GQA，Mistral 用 Sliding Window Attention。不同的注意力机制意味着模型在处理辩论上下文时的"关注点"不同——这正是异构辩论的核心价值。

第二，能力互补。Qwen 数学强但英文常识略弱，Llama 英文理解强但数学不如 Qwen，Mistral 速度快但上下文短。在辩论中，不同 agent 的"知识盲区"不同，更容易通过辩论互相纠正。

第三，社区成熟。三个模型都有 GGUF 量化版本（可用 Ollama/llama.cpp）和 FP16 版本（可用 vLLM），部署教程和社区支持充分。

---

## 二、本地部署步骤

### 硬件需求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 1 × 24GB（RTX 3090/4090/A5000） | 2 × 24GB 或 1 × 48GB（A6000） |
| 内存 | 32GB | 64GB |
| 硬盘 | 100GB SSD（存放三个模型） | 200GB+ |

单卡 24GB 可以用 4-bit 量化跑所有三个模型，或者 FP16 跑其中一个。如果有两张卡，可以同时部署两个模型做并行实验。

### 方案 A：用 vLLM 部署（推荐，适合批量实验）

vLLM 支持高吞吐推理、连续批处理、OpenAI 兼容 API。部署后你的实验代码不需要改——和调 API 的接口完全一样，只是 base_url 换成本地地址。

```bash
# 1. 创建环境
conda create -n mad-exp python=3.10 -y
conda activate mad-exp

# 2. 安装 vLLM
pip install vllm

# 3. 下载模型（三个模型各约 15GB FP16）
# HuggingFace 国内镜像加速
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen2.5-7b
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/llama3.1-8b
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ./models/mistral-7b

# 4. 启动 vLLM 服务（每次启动一个模型，用不同端口）

# 启动 Qwen（端口 8001）
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen2.5-7b \
    --port 8001 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85

# 启动 Llama（端口 8002）— 需要第二张卡或者关掉 Qwen 再启动
python -m vllm.entrypoints.openai.api_server \
    --model ./models/llama3.1-8b \
    --port 8002 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85

# 启动 Mistral（端口 8003）
python -m vllm.entrypoints.openai.api_server \
    --model ./models/mistral-7b \
    --port 8003 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
```

如果只有一张卡，不需要同时启动三个模型。实验代码中按顺序调用：Agent 1 用 Qwen → 请求发到 8001，Agent 2 用 Llama → 关掉 Qwen 启动 Llama 在 8001。或者用 4-bit 量化，单卡同时装两个模型：

```bash
# 4-bit 量化，单卡可同时跑两个 7B 模型
pip install bitsandbytes
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen2.5-7b \
    --port 8001 \
    --quantization awq \
    --max-model-len 4096
```

### 方案 B：用 Ollama 部署（更简单，适合快速验证）

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. 拉取三个模型（自动下载量化版本）
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull mistral:7b

# 3. 启动服务（默认端口 11434，兼容 OpenAI 格式）
ollama serve

# 4. 在代码中通过模型名切换
# Agent 1 → model="qwen2.5:7b"
# Agent 2 → model="llama3.1:8b"
# Agent 3 → model="mistral:7b"
```

Ollama 的好处是自动管理模型加载/卸载，单卡也能跑三个模型（轮流加载）。缺点是批量实验时吞吐量不如 vLLM。

### 代码层面的改动

在 config.py 中将单一模型改为模型列表：

```python
AGENT_CONFIGS = [
    {"model": "qwen2.5:7b",   "base_url": "http://localhost:11434/v1", "name": "Qwen2.5-7B"},
    {"model": "llama3.1:8b",  "base_url": "http://localhost:11434/v1", "name": "Llama3.1-8B"},
    {"model": "mistral:7b",   "base_url": "http://localhost:11434/v1", "name": "Mistral-7B"},
]
```

在 agent.py 中让每个 agent 根据自己的 agent_id 选择对应的模型配置。

---

## 三、数据集方案

| 数据集 | 类型 | 题数 | 答案形式 | 评测方式 | 用途 |
|--------|------|------|---------|---------|------|
| **MMLU-Pro** | 单选（10选1） | 12032 | 单字母 A-J | Exact Match | 主实验：比 MMLU 更难，辩论空间更大 |
| **GSM8K** | 数学 | 1319 | 数值 | Exact Match | 主实验：测试推理链压缩的价值 |
| **ARC-Challenge** | 单选（4选1） | 1172 | 单字母 A-D | Exact Match | 补充：科学推理 |
| **MultiRC** | 多选 | 4848 | 字母集合 | F1 / Exact Match | 扩展：多选题场景 |
| **MT-Bench** | 开放 | 80 | 自由文本 | LLM-as-Judge (1-10分) | 扩展：开放性问答 |

主实验用 MMLU-Pro + GSM8K（有确定答案，和现有 MAD 文献可对标）。
扩展实验用 MultiRC（多选）和 MT-Bench（开放），验证框架的泛化性。
每个数据集抽取 300 题做正式实验（保证统计显著性）。

---

## 四、实验组设计

### 主实验组

| 编号 | 方法 | 说明 |
|------|------|------|
| E0 | Single Agent | 每道题只用 Qwen（最强的单模型）回答一次 |
| E1 | SC (Self-Consistency) | 三个异构模型各独立回答一次，majority vote |
| E2 | MAD-Fixed | 三个异构模型链式辩论固定 3 轮，无 summarizer 无早退 |
| E3 | MAD + Summarizer | 加入方案 B 的 summarizer（按需压缩），固定 3 轮 |
| E4 | MAD + Early Exit | 无 summarizer，但加入早退机制 |
| **E5** | **MAD + Summarizer + Early Exit（完整方案）** | 方案 B 完整版 |
| E6 | MAD + Summarizer + Early Exit + DA | 完整方案 + Devil's Advocate |

### 消融实验

| 消融组 | 去掉什么 | 验证什么 |
|--------|---------|---------|
| E5 vs E3 | 去掉早退 | 早退的独立贡献 |
| E5 vs E4 | 去掉 summarizer | Summarizer 的独立贡献 |
| E5 vs E2 | 去掉 summarizer 和早退 | 两者联合贡献 |
| E6 vs E5 | 去掉 DA | DA 的增量价值 |
| E5 (同构) vs E5 (异构) | 换成三个相同模型 | 异构 vs 同构的影响 |
| E5 (Full only) vs E5 (Full+Partial) | 去掉 Partial 模式 | 部分压缩的价值 |

---

## 五、论文中需要的实验结果图表

### 图 1：主结果表（Table）

所有实验组在所有数据集上的准确率和 token 消耗。这是论文最核心的表格。

| Method | MMLU-Pro Acc | MMLU-Pro Tokens | GSM8K Acc | GSM8K Tokens | ARC-C Acc | ARC-C Tokens |
|--------|-------------|----------------|-----------|-------------|-----------|-------------|
| E0 Single | ... | ... | ... | ... | ... | ... |
| E1 SC | ... | ... | ... | ... | ... | ... |
| E2 MAD-Fixed | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |
| **E5 Ours** | **...** | **...** | **...** | **...** | **...** | **...** |

### 图 2：Accuracy vs Token 的 Pareto 曲线

**横轴：** 总 token 消耗（对数刻度）
**纵轴：** 准确率 (%)
**每个实验组是一个点**，用不同颜色和形状区分。理想结果是我们的方法（E5）位于左上角——token 少、准确率高。

这张图最直观地展示"性价比"——reviewer 一眼就能看出哪个方法在 accuracy-cost 空间中占优。

### 图 3：每轮准确率和 Token 消耗的双轴图

**横轴：** 辩论轮次（R0, R1, R2, R3）
**左纵轴：** Majority vote 准确率 (%)
**右纵轴：** 该轮累计 token 消耗
**两条线：** MAD-Fixed（每轮都跑）vs 我们的方法（某些题提前退出后该轮 token 降低）

展示"随着轮次增加，准确率的边际收益递减但 token 成本线性增长"，以及我们的方法如何在最佳时刻停止。

### 图 4：早退分布直方图

**横轴：** 实际退出轮次（R1, R2, R3, 跑满）
**纵轴：** 题数占比 (%)
**分颜色：** 退出时答案正确（绿色）vs 退出时答案错误（红色）

展示早退机制的行为分布——大部分题在 R1-R2 就退出了且答案正确，只有少数难题跑满了轮数。

### 图 5：压缩模式分布（堆叠柱状图）

**横轴：** 辩论轮次（R0, R1, R2, R3）
**纵轴：** 比例 (%)
**三种颜色堆叠：** Full 压缩 / Partial 压缩 / None（不压缩）

展示随着辩论推进，Full 压缩的比例越来越高（agent 越来越一致），和"局部收敛→压缩→持续收敛→退出"的设计逻辑一致。

### 图 6：四分类分析（iMAD 框架）

**柱状图**，每个实验组一组，四种颜色：
- ✗→✓（辩论纠正了错误）—— 绿色
- ✓→✗（辩论把对的改错了）—— 红色
- ✓→✓（本来就对，辩论没变）—— 灰色
- ✗→✗（本来就错，辩论也没救）—— 深灰

展示我们的方法相比 MAD-Fixed：绿色不减少（纠错能力不变），红色减少（早退避免了 groupthink），灰色对应的 token 更少（不需要辩论的题更快退出）。

### 图 7：Summarizer 的压缩效果

**横轴：** 辩论轮次
**纵轴：** 平均每题的 prompt tokens
**两条线：** MAD-Fixed（prompt 逐轮膨胀）vs MAD + Summarizer（prompt 增长被压缩抑制）

直接量化 summarizer 对 prompt token 膨胀的抑制效果。

### 图 8：Scaling 实验（如果有算力）

**横轴：** Agent 数量（3, 5, 7）
**纵轴（双轴）：** 准确率 + Token 节省比例
**两组柱子：** MAD-Fixed vs 我们的方法

展示"agent 越多，我们的方法节省的 token 比例越大"——因为冗余随 agent 数量增加，压缩的收益也越大。

### 图 9：消融实验表

| Ablation | MMLU-Pro | GSM8K | Avg Token |
|----------|----------|-------|-----------|
| Full method (E5) | ... | ... | ... |
| w/o Summarizer | ... | ... | ... |
| w/o Early Exit | ... | ... | ... |
| w/o Partial mode | ... | ... | ... |
| w/o DA | ... | ... | ... |
| Homogeneous agents | ... | ... | ... |

### 图 10：MT-Bench 开放性任务结果（扩展实验）

**横轴：** 评估维度（Writing, Reasoning, Math, Coding, ...）
**纵轴：** LLM-as-Judge 平均分（1-10）
**分组柱状图：** SC vs MAD-Fixed vs 我们的方法

开放性任务没有 exact match 准确率，用 LLM-as-Judge 的多维度分数替代。

---

## 六、Baseline 选择

| Baseline | 为什么选 | 对标什么 |
|----------|---------|---------|
| Single Agent (E0) | 最低成本的 baseline | 辩论本身是否有价值 |
| SC (E1) | 无辩论的多模型聚合 | 辩论比单纯投票好多少 |
| MAD-Fixed (E2) | 固定轮数辩论 | 我们的压缩+早退能省多少 |
| S²-MAD | 内容过滤方法（如果有算力复现） | 我们的 summarizer 和过滤谁更好 |
| ChatEval w/ Summarizer | 无条件压缩 | 我们的按需压缩是否优于无条件压缩 |

最核心的对比是 **E5 vs E2**——同样是辩论，加了我们的机制后 token 省了多少、准确率有没有变。这是论文 contribution 的直接量化。

---

## 七、Sensitivity Analysis

除了消融实验，还需要测试关键参数的敏感性：

| 参数 | 测试值 | 预期结论 |
|------|--------|---------|
| 最大轮数 t_max | 3, 4, 6, 8 | 6 左右为甜点 |
| 争议摘要兜底 K | 2, 3, 4, 5 | 3-4 为甜点 |
| Agent 数量 N | 3, 5, 7 | Agent 越多，节省比例越大 |
| 连续 Full 阈值 | 1, 2, 3 | 2 是平衡点 |

---

## 八、实验优先级和时间规划

| 阶段 | 内容 | 时间 |
|------|------|------|
| 1 | 部署三个模型到本地，跑通 test_chain.py | 2-3 天 |
| 2 | 实现 Summarizer（方案 B 的三种模式），谁来做 Summarizer 需要确定——建议用 Qwen（最强的）或者用 API 调 GLM-4 | 3-4 天 |
| 3 | E0-E2 在 MMLU-Pro 300 题上跑完（验证异构辩论有效性） | 2-3 天 |
| 4 | E3-E6 在 MMLU-Pro 上跑完（验证 Summarizer + 早退） | 3-4 天 |
| 5 | 在 GSM8K、ARC-C 上重复 E0-E6 | 3-4 天 |
| 6 | 消融实验和 Sensitivity Analysis | 3-4 天 |
| 7 | 扩展实验（MultiRC、MT-Bench） | 3-4 天 |
| 8 | 画图、统计检验、写论文 | 1-2 周 |

总计约 5-6 周。如果算力充足（多卡并行），可以压缩到 3-4 周。
