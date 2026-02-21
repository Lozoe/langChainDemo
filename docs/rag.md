# 📘 RAG 技术与应用 — 学习笔记

---

## 一、RAG 在大模型应用中的定位

### 1. 大模型应用三种模式

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#e8edf2', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b', 'secondaryColor': '#f0f4f8'}}}%%
flowchart LR
    A[🧑 用户问题] --> B{选择方案}
    B --> C[💬 提示工程 Prompt]
    B --> D[🔍 RAG 检索增强]
    B --> E[🔧 微调 Fine-tune]

    C --> C1[适合通用知识]
    D --> D1[适合私有知识 / 实时信息]
    E --> E1[适合领域深度定制]

    style A fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style B fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style C fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style D fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style E fill:#f0e4e8,stroke:#b07a8a,color:#5a2a3a
    style C1 fill:#f2f7ee,stroke:#9ab88a,color:#3a5a2a
    style D1 fill:#eaf3fa,stroke:#7aaece,color:#1f4e6f
    style E1 fill:#f8eef1,stroke:#c499a8,color:#5a2a3a
```

### 使用场景对比

| 方法 | 适用场景 | 优点 | 缺点 |
|:---:|:---:|:---:|:---:|
| **Prompt 工程** | 通用问答 | 快速、简单 | 易产生幻觉 |
| **RAG** | 私有知识库 | 可更新、可溯源 | 架构复杂 |
| **微调** | 垂直领域 | 精度高 | 成本高 |

---

## 二、什么是 RAG？

**RAG = Retrieval-Augmented Generation（检索增强生成）**

> 💡 核心思想：**先检索 → 再生成**

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart LR
    Q[🧑 用户问题] --> R[🔎 向量检索]
    R --> C[📄 相关文档]
    C --> LLM[🤖 大模型生成]
    LLM --> A[✅ 最终答案]

    style Q fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style R fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style C fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style LLM fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style A fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
```

### RAG 的优势

| | 优势 | 说明 |
|:---:|:---|:---|
| ✅ | 解决知识时效性问题 | 知识库随时可更新 |
| ✅ | 减少模型幻觉 | 基于真实文档生成 |
| ✅ | 提升专业领域质量 | 注入领域知识 |
| ✅ | 支持私有数据部署 | 数据不出域 |

---

## 三、RAG 核心流程

### 整体流程

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart TD
    A[📥 数据预处理] --> B[📦 向量化存储]
    B --> C[🧑 用户查询]
    C --> D[🔎 相似度检索]
    D --> E[📝 上下文增强]
    E --> F[🤖 LLM 生成答案]

    style A fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style B fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style C fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style D fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style E fill:#f0e4e8,stroke:#b07a8a,color:#5a2a3a
    style F fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
```

---

### Step 1：数据预处理

| 步骤 | 说明 |
|:---:|:---|
| 1️⃣ | 文档收集 |
| 2️⃣ | 文档分块（Chunking） |
| 3️⃣ | Embedding 向量化 |
| 4️⃣ | 存入向量数据库 |

**关键参数：**

```python
chunk_size    = 1000   # 每块大小
chunk_overlap = 200    # 块间重叠
```

> 👉 **平衡原则**：块太大 → 检索不精确；块太小 → 语义不完整

---

### Step 2：检索阶段

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart LR
    Q[🔤 Query] --> E[📐 Embedding]
    E --> V[🗄️ VectorDB]
    V --> T[📋 Top-K 文档]
    T --> R[🏆 重排序 Rerank]

    style Q fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style E fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style V fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style T fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style R fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
```

---

### Step 3：生成阶段

| 步骤 | 说明 |
|:---:|:---|
| 1️⃣ | 拼接检索到的上下文 |
| 2️⃣ | 连同用户问题一起送入 LLM |
| 3️⃣ | 输出答案 + 来源引用 |

---

## 四、Embedding 模型选择

### 分类对比

| 类型 | 模型 | 特点 | 适用场景 |
|:---:|:---|:---|:---|
| 🌐 通用 | BGE-M3 | 多语言 + 长文本 | 企业级 RAG |
| 🌐 通用 | text-embedding-3-large | 英文强 | 国际应用 |
| ⚡ 轻量 | Jina-v2-small | 实时推理 | 边缘设备 |
| 🇨🇳 中文 | M3E-base | 本地部署友好 | 中文检索 |
| 🇨🇳 中文 | xiaobu-embedding | 中文语义强 | 中文 NLP |
| 🎯 指令型 | gte-Qwen2 | 复杂任务 | 智能问答 |

---

### Embedding 选择策略

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart TD
    A[🧩 选择 Embedding] --> B{🌍 语言}
    B -->|中文| C[M3E / xiaobu]
    B -->|多语言| D[BGE-M3]

    A --> E{🖥️ 部署环境}
    E -->|本地| F[⚡ 轻量模型]
    E -->|云端| G[🔋 大模型]

    style A fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style B fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style C fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style D fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style E fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style F fill:#f0e4e8,stroke:#b07a8a,color:#5a2a3a
    style G fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
```

---

## 五、案例：DeepSeek + Faiss 本地知识库

### 架构图

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart TD
    subgraph 离线索引
        PDF[📄 PDF 文档] --> Extract[📝 文本提取]
        Extract --> Split[✂️ 文本分割]
        Split --> Embed[📐 Embedding]
        Embed --> Faiss[🗄️ Faiss 向量库]
    end

    subgraph 在线查询
        User[🧑 用户问题] --> Search[🔎 相似度检索]
        Search --> Context[📋 上下文拼接]
        Context --> DeepSeek[🤖 DeepSeek LLM]
        DeepSeek --> Answer[✅ 答案]
    end

    Faiss -.->|索引查询| Search

    style PDF fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style Extract fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style Split fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style Embed fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style Faiss fill:#f0e4e8,stroke:#b07a8a,color:#5a2a3a
    style User fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style Search fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style Context fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style DeepSeek fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style Answer fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
```

### 技术栈

| 模块 | 技术 | 说明 |
|:---:|:---:|:---|
| 文档解析 | `PyPDF2` | PDF 文本提取 |
| 分割 | `LangChain Splitter` | 文本分块 |
| 向量库 | `Faiss` | 高效近邻检索 |
| Embedding | `DashScope` | 阿里云向量化 |
| LLM | `DeepSeek` | 大语言模型 |
| 编排 | `LangChain` | 流程编排框架 |

---

## 六、LangChain 问答链类型

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart TD
    A[📄 文档 Chunks] --> B{⚙️ chain_type}

    B --> C[stuff]
    B --> D[map_reduce]
    B --> E[refine]
    B --> F[map_rerank]

    style A fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style B fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style C fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
    style D fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style E fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style F fill:#f0e4e8,stroke:#b07a8a,color:#5a2a3a
```

| 类型 | 特点 | 适合场景 | 推荐 |
|:---:|:---|:---|:---:|
| **stuff** | 所有文档一次性输入 LLM | 小文档、短上下文 | ⭐ |
| **map_reduce** | 每段独立处理后汇总 | 大文档、并行处理 | |
| **refine** | 逐段迭代优化答案 | 长上下文、高精度 | |
| **map_rerank** | 每段评分后取最佳 | 精准筛选 | |

> 👉 推荐优先使用 **stuff**，简单高效，适合大多数场景

---

## 七、Query 改写（提升检索质量）

### 为什么需要改写？

| 用户 Query | 知识库文档 |
|:---:|:---:|
| 口语化 | 书面化 |
| 模糊表达 | 结构化描述 |
| 上下文依赖 | 独立完整 |

> 👉 Query 改写就是 **"翻译器"**，弥合用户表达与知识库之间的语义鸿沟

---

### Query 改写类型

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50'}}}%%
mindmap
    root((🔄 Query 改写))
        上下文依赖型
        对比型
        模糊指代型
        多意图型
        反问型
```

---

### 改写流程

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart LR
    Q[💬 原始 Query] --> C[🎯 意图识别]
    C --> R[✏️ Query 改写]
    R --> S[🔎 检索]

    style Q fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style C fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style R fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style S fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
```

---

## 八、Query + 联网搜索

### 什么时候需要联网？

| 类型 | 示例 | 原因 |
|:---:|:---|:---|
| ⏰ 时效性 | "今天开放吗？" | 信息实时变化 |
| 💰 价格 | "门票多少钱？" | 价格可能调整 |
| 🌤️ 天气 | "明天天气怎样？" | 实时气象数据 |
| 📊 实时状态 | "现在人多吗？" | 动态变化信息 |

---

### 联网判断流程

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart TD
    Q[🧑 用户 Query] --> Judge{🧠 是否需要联网?}

    Judge -->|否| RAG[🗄️ RAG 检索]
    Judge -->|是| Web[🌐 联网搜索]
    Web --> Merge[🔗 结果融合]
    RAG --> Merge
    Merge --> Answer[✅ 生成答案]

    style Q fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style Judge fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style RAG fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style Web fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style Merge fill:#f0e4e8,stroke:#b07a8a,color:#5a2a3a
    style Answer fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
```

---

### 联网搜索系统设计

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart LR
    A[🔍 Query 识别] --> B[✏️ 搜索改写]
    B --> C[📋 生成搜索策略]
    C --> D[⚡ 执行搜索]
    D --> E[🔗 融合 RAG 结果]

    style A fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style B fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style C fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style D fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style E fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
```

---

## 九、RAG vs 无限上下文 LLM

> 即使 LLM 支持超长上下文，RAG **仍然有意义**：

| | 优势 | 说明 |
|:---:|:---|:---|
| 🚀 | **更高效率** | 只检索相关片段，无需处理全文 |
| 💰 | **更低成本** | 减少 Token 消耗 |
| 🔄 | **实时更新** | 知识库独立更新，无需重训模型 |
| 🔍 | **可解释性** | 答案可追溯到原始文档 |
| 🔒 | **数据隐私** | 敏感数据不必上传至模型服务商 |

---

## 十、完整 RAG 系统架构

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eaf3fa', 'primaryTextColor': '#2c3e50', 'lineColor': '#7f8c9b'}}}%%
flowchart TD
    User[🧑 用户问题] --> QR[✏️ Query 改写]

    QR --> Judge{🧠 需要联网?}

    Judge -->|是| Web[🌐 联网搜索]
    Judge -->|否| Vec[🗄️ 向量检索]

    Web --> Fusion[🔗 信息融合]
    Vec --> Fusion

    Fusion --> LLM[🤖 LLM 生成]
    LLM --> Answer[✅ 最终答案]

    style User fill:#dce6f1,stroke:#5b7a9d,color:#2c3e50
    style QR fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style Judge fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style Web fill:#e5eeda,stroke:#7a9a5a,color:#3a5a2a
    style Vec fill:#daeaf5,stroke:#5b8fb9,color:#1f4e6f
    style Fusion fill:#f0e4e8,stroke:#b07a8a,color:#5a2a3a
    style LLM fill:#f5f0e8,stroke:#c4a97d,color:#5a4a32
    style Answer fill:#e0f0e3,stroke:#6aaa7a,color:#1f4e3f
```

---

## 十一、实战 Checklist

### 搭建自己的 RAG 系统

- [ ] 📥 收集知识库文档
- [ ] 📄 PDF 文本提取
- [ ] ✂️ 文本分块（Chunking）
- [ ] 📐 选择 Embedding 模型
- [ ] 🗄️ 构建向量库
- [ ] 🤖 接入 LLM
- [ ] ✏️ Query 改写
- [ ] 🌐 联网搜索判断
- [ ] 🔗 结果溯源

---

## 总结

> 💡 RAG 的本质：**让 LLM 会查资料再回答**

### 核心能力

| 能力 | 说明 |
|:---:|:---|
| 🔍 高质量检索 | 精准匹配相关知识 |
| 🧠 Query 理解 | 意图识别与改写 |
| 📝 上下文增强 | 注入外部知识 |
| 🏗️ 可扩展架构 | 灵活接入多种数据源 |

### 未来趋势

> 👉 **RAG + Agent + Web Search = 智能知识系统**