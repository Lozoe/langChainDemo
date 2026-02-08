# Embedding 与向量数据库学习笔记

## 一、什么是 Embedding？

**Embedding** 是一种将离散对象（如词、句子、图像等）映射为**低维、稠密向量**的技术，目的是捕捉语义信息，使得语义相近的对象在向量空间中距离更近。

```mermaid
graph LR
    subgraph 输入["离散对象"]
        A["词语"]
        B["句子"]
        C["图像"]
    end
    
    subgraph 转换["Embedding 模型"]
        D["向量化"]
    end
    
    subgraph 输出["稠密向量"]
        E["[0.12, -0.45, 0.78, ...]"]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E
```

### 核心特性

| 特性 | 说明 | 示例 |
|:---:|:---|:---|
| **固定维度** | 所有对象映射为相同长度的向量 | 768 维、1024 维 |
| **语义相似度** | 通过余弦相似度等度量计算相似性 | `cos(v₁, v₂) → [0, 1]` |
| **向量运算** | 支持语义层面的数学运算 | `king - man + woman ≈ queen` |

---

## 二、传统文本特征表示方法

### 1. N-Gram

基于"第 n 个词与前 n−1 个词相关"的假设，用于扩展特征，提升模型对局部上下文的感知能力。

| 类型 | 名称 | 示例（原句：我爱学习） |
|:---:|:---:|:---|
| 1-gram | Unigram | `[我, 爱, 学, 习]` |
| 2-gram | Bigram | `[我爱, 爱学, 学习]` |
| 3-gram | Trigram | `[我爱学, 爱学习]` |

### 2. TF-IDF

```mermaid
graph LR
    subgraph TF["TF (词频)"]
        A["词在文档中出现的频率"]
    end
    
    subgraph IDF["IDF (逆文档频率)"]
        B["衡量词的区分度<br/>越少文档包含该词，IDF 越高"]
    end
    
    subgraph 结果["TF-IDF"]
        C["突出重要且具区分性的关键词"]
    end
    
    A --> |"×"| C
    B --> |"×"| C
```

> [!WARNING]
> **缺点**：N-Gram + TF-IDF 生成的特征矩阵**极度稀疏**，计算开销大。

---

## 三、Word Embedding 技术

### 1. Word2Vec

将词映射到连续向量空间，语义相近的词距离更近，输出为一个"查找表"（Lookup Table）。

```mermaid
graph TB
    subgraph CBOW["CBOW 模式"]
        direction TB
        C1["上下文词"] --> C2["预测中心词"]
        C3["The cat ___ on mat"] --> C4["预测: sat"]
    end
    
    subgraph SkipGram["Skip-Gram 模式"]
        direction TB
        S1["中心词"] --> S2["预测上下文"]
        S3["输入: sat"] --> S4["预测: The, cat, on, mat"]
    end
```

| 模式 | 输入 | 输出 | 适用场景 |
|:---:|:---:|:---:|:---|
| **Skip-Gram** | 中心词 | 上下文词 | 低频词效果更好 |
| **CBOW** | 上下文词 | 中心词 | 训练速度更快 |

### 2. 工具支持

- **Gensim**：支持 Word2Vec、FastText、Doc2Vec 等
- 可用于训练自定义语料（如《西游记》《三国演义》），进行词相似度或类比推理

---

## 四、如何选择合适的 Embedding 模型？

### 1. MTEB 榜单

**MTEB（Massive Text Embedding Benchmark）** 包含 **8 大任务类型**、**58 个数据集**，全面评估模型性能：

```mermaid
mindmap
  root((MTEB<br/>评估任务))
    检索类
      Retrieval["检索"]
      Reranking["重排序"]
    相似度类
      STS["语义文本相似度"]
      PairClassification["对分类"]
    分类聚类类
      Classification["分类"]
      Clustering["聚类"]
    其他
      BitextMining["双语挖掘"]
      Summarization["摘要评估"]
```

> [!TIP]
> 根据具体任务（如检索 or 分类）筛选候选模型。

### 2. 向量维度的影响

```mermaid
graph LR
    subgraph 高维["高维 (1024/2048)"]
        H1["✅ 表达能力强"]
        H2["✅ 适合复杂语义任务"]
        H3["❌ 内存占用大"]
        H4["❌ 计算速度慢"]
    end
    
    subgraph 低维["低维 (128/256)"]
        L1["✅ 速度快"]
        L2["✅ 资源消耗少"]
        L3["✅ 适合实时场景"]
        L4["❌ 表达能力受限"]
    end
```

> [!IMPORTANT]
> **选择原则**：若升维带来的性能提升 < 1%，但内存增加 > 30%，则**不值得**。

### 3. "俄罗斯套娃"技术（MRL）

**Matryoshka Representation Learning** 允许模型内部生成完整高维向量，可按需截取前 N 维，仍保持高质量。

```mermaid
graph LR
    A["完整向量<br/>2048 维"] --> B["截取 1024 维"]
    B --> C["截取 512 维"]
    C --> D["截取 256 维"]
    D --> E["截取 128 维"]
    
    style A fill:#e1f5fe
    style B fill:#b3e5fc
    style C fill:#81d4fa
    style D fill:#4fc3f7
    style E fill:#29b6f6
```

### 4. 单语言 vs 多语言模型

| 类型 | 代表模型 | 特点 | 适用场景 |
|:---:|:---|:---|:---|
| **单语言模型** | BGE-large-zh | 针对特定语言优化，理解更深 | 中文客服问答、电商 FAQ 匹配 |
| **多语言模型** | m3e-base, multilingual-E5 | 多语言统一语义空间 | 跨语言评论分析、国际酒店评价 |

> [!NOTE]
> 跨语言检索依赖：不同语言表达相同语义时，向量应接近（如 `"clean room" ≈ "干净的房间"`）

### 5. 模型选型流程

```mermaid
flowchart TD
    A["明确业务目标"] --> B["确定评估指标<br/>Recall@K, Accuracy 等"]
    B --> C["构建黄金测试集<br/>真实业务数据小样本"]
    C --> D["从 MTEB 榜单<br/>筛选候选模型"]
    D --> E["在测试集上实测"]
    E --> F{"效果达标?"}
    F -->|"是"| G["综合考虑速度、部署成本"]
    F -->|"否"| D
    G --> H["最终选定模型"]
    
    style A fill:#fff3e0
    style H fill:#c8e6c9
```

---

## 五、什么是向量数据库？

向量数据库是专门用于存储和高效检索**高维向量**的数据库，核心能力是**相似性搜索**。

```mermaid
graph TB
    subgraph 非结构化数据
        A["文本"] 
        B["图像"]
        C["音频"]
    end
    
    subgraph Embedding
        D["Embedding 模型"]
    end
    
    subgraph 向量数据库
        E["向量存储"]
        F["索引结构"]
        G["相似性搜索"]
    end
    
    subgraph 应用
        H["语义搜索"]
        I["推荐系统"]
        J["智能问答"]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    G --> I
    G --> J
```

### 核心价值

| 价值 | 说明 |
|:---:|:---|
| 🧠 **长期记忆** | 为大模型提供外部记忆，突破上下文窗口限制 |
| 📚 **私有知识库** | 实现企业私有知识库问答 |
| 🎯 **智能应用** | 支撑推荐系统、以图搜图、语义搜索等 AI 应用 |

---

## 六、主流向量数据库对比

```mermaid
quadrantChart
    title 向量数据库定位分析
    x-axis 轻量级 --> 企业级
    y-axis 需运维 --> 免运维
    quadrant-1 全托管 + 企业级
    quadrant-2 轻量 + 免运维
    quadrant-3 轻量 + 需运维
    quadrant-4 企业级 + 需运维
    
    Pinecone: [0.85, 0.9]
    Milvus: [0.8, 0.3]
    FAISS: [0.2, 0.4]
    Weaviate: [0.6, 0.7]
    Qdrant: [0.7, 0.5]
    Elasticsearch: [0.75, 0.4]
```

| 数据库 | 特点 | 性能 | 适用场景 |
|:---:|:---|:---|:---|
| **FAISS** | Meta 开发，算法库（非完整 DB），支持 CPU/GPU | ⚡ 极快（内存内裸向量检索） | 算法研究、需深度集成的系统 |
| **Milvus** | 开源、云原生、高扩展 | 📈 大规模数据下表现优异 | 企业级应用，需私有部署 |
| **Pinecone** | 全托管 Serverless，API 简洁 | ⏱️ 低延迟，性能稳定 | 快速上线、免运维团队 |
| **Weaviate** | 内置自动向量化，支持多种模型 | 🛠️ 易用性强 | 快速构建端到端检索链路 |
| **Qdrant** | Rust 编写，内存安全，过滤能力强 | 🔍 混合查询性能突出 | 金融、电商等复杂过滤场景 |
| **Elasticsearch** | 通用搜索引擎，新增 k-NN 功能 | 🔗 混合搜索强 | 以文本搜索为主，向量为辅 |

---

## 七、向量数据库 vs 传统数据库

```mermaid
graph TB
    subgraph 传统数据库["传统数据库 (SQL)"]
        T1["结构化数据<br/>表格、字段"]
        T2["精确匹配查询<br/>WHERE id = 123"]
        T3["事务处理<br/>报表统计"]
    end
    
    subgraph 向量数据库["向量数据库"]
        V1["高维向量<br/>非结构化数据嵌入"]
        V2["相似性搜索<br/>余弦相似度、L2距离"]
        V3["语义搜索<br/>内容推荐、智能问答"]
    end
    
    style 传统数据库 fill:#fff3e0
    style 向量数据库 fill:#e3f2fd
```

| 维度 | 传统数据库 | 向量数据库 |
|:---:|:---:|:---:|
| **数据类型** | 结构化（表格、字段） | 高维向量（非结构化数据的嵌入） |
| **查询方式** | 精确匹配（`=`、`<`、`>`） | 相似性搜索（余弦、L2 距离） |
| **应用场景** | 事务处理、报表统计 | 语义搜索、内容推荐、智能问答 |

---

## 八、数据导入向量数据库的流程

```mermaid
flowchart LR
    subgraph 准备阶段
        A["原始数据"] --> B["数据清洗"]
    end
    
    subgraph 向量化阶段
        B --> C{"数据类型?"}
        C -->|"文本"| D["BGE / Jina / text-embedding"]
        C -->|"图像"| E["CLIP / ResNet"]
        D --> F["生成向量"]
        E --> F
    end
    
    subgraph 存储阶段
        F --> G["向量数据库<br/>存储向量"]
        F --> H["元数据库<br/>Redis / PostgreSQL"]
        G <-->|"唯一 ID 关联"| H
    end
    
    subgraph 检索阶段
        I["查询输入"] --> J["生成查询向量"]
        J --> K["检索 Top-K"]
        K --> L["返回结果 + 元数据"]
    end
    
    G --> K
    H --> L
```

### 架构设计要点

```mermaid
graph TB
    subgraph 推荐架构
        direction LR
        VDB["向量数据库<br/>专注高速检索"]
        MDB["元数据库<br/>Redis / PostgreSQL"]
        VDB <-->|"ID 关联"| MDB
    end
    
    style VDB fill:#e8f5e9
    style MDB fill:#fff3e0
```

> [!TIP]
> **架构建议**：向量数据库专注**高速检索**，元数据管理交由专业数据库，实现**解耦与高效协同**。

---

## 九、总结

```mermaid
mindmap
  root((关键要点))
    Embedding
      语义理解基石
      选模型需结合任务、语言、资源
    MTEB 榜单
      重要参考
      不能替代业务测试
    向量数据库
      LLM 时代的外部记忆
      权衡性能、功能与运维成本
    选型建议
      FAISS: 轻量/研究场景
      Milvus/Pinecone: 生产环境
    元数据管理
      不可忽视
      实现可解释、可追溯检索
```

| 核心观点 | 说明 |
|:---:|:---|
| 🎯 **Embedding 是基石** | 选择合适模型需结合任务、语言、资源 |
| 📊 **MTEB 是参考** | 重要但不能替代业务测试 |
| 🧠 **向量数据库是外部记忆** | 选型需权衡性能、功能与运维成本 |
| ⚙️ **选型建议** | FAISS 适合轻量/研究，Milvus/Pinecone 适合生产 |
| 📋 **元数据不可忽视** | 是实现可解释、可追溯检索的关键 |

---

> 📅 **最后更新**：2026-02-08
