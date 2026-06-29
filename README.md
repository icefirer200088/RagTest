# RagTest

**轻量级 RAG 检索实验项目**

用 LLM Embedding + 向量检索做自然语言模糊搜索。和传统的关键词精确匹配不同，RAG 的"搜索"更像是用自然语言去"想"——你问一个问题，它找到最相关的片段，而不是命中同一个词。

## 目标

- 把一段文本切碎 → 算向量 → 存起来
- 问一句话 → 找到最相关的碎片
- 纯 Python，最小依赖，跑起来就能用

## Quick Start

```bash
pip install -r requirements.txt
python src/rag_search.py --load data/demo.txt --query "你的问题"
```

## 项目结构

```
RagTest/
├── src/
│   ├── rag_search.py    # 主入口 — 吃数据、建索引、查
│   ├── embedder.py      # 文本 → 向量（LLM Embedding API）
│   ├── chunk.py         # 文本分块策略
│   └── vector_store.py  # 向量存储 + 最近邻检索（纯 numpy）
├── data/
│   └── demo.txt         # 一份示例文本
├── examples/            # Jupyter / 示例脚本
├── scripts/             # 小工具
├── requirements.txt
└── README.md
```
