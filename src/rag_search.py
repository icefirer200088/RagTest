"""
rag_search.py — RAG 检索主入口

用法：
    # 用本地 embedding（零依赖，推荐）
    python src/rag_search.py --load data/demo.txt --query "Agent 框架有哪些"

    # 用 API embedding（需要 DEEPSEEK_API_KEY）
    python src/rag_search.py --load data/demo.txt --query "向量检索" --mode api

    # 交互模式（不传 --query 即可）
    python src/rag_search.py --load data/demo.txt

流程：
    1. 读文件 → 分块 → 算向量 → 建索引
    2. 用户输入查询 → 算查询向量 → 余弦相似度检索 → 打印 top-k
"""

import argparse
import sys
import os

# 确保能 import src 下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import Embedder as ApiEmbedder
from src.embedder_local import LocalEmbedder
from src.embedder_onnx import ONNXEmbedder
from src.chunk import chunk_by_sentence, chunk_by_size, chunk_by_paragraph
from src.vector_store import VectorStore


def build_index(text: str, embedder, strategy: str = "sentence"):
    """
    读文本 → 分块 → 向量化 → 建索引
    """
    # 分块
    if strategy == "sentence":
        chunks = chunk_by_sentence(text)
    elif strategy == "paragraph":
        chunks = chunk_by_paragraph(text)
    elif strategy == "size":
        chunks = chunk_by_size(text)
    else:
        raise ValueError(f"未知分块策略: {strategy}")

    print(f"📦 分块数: {len(chunks)}")
    print("🧮 计算向量...")
    vectors = embedder.embed_batch(chunks)

    # 建索引
    store = VectorStore()
    store.add_batch(chunks, vectors)
    print(f"✅ 索引完成，共 {store.size} 条记录")
    return store


def search_loop(store: VectorStore, embedder):
    """交互式检索循环"""
    print("\n🔍 输入查询（空行退出）:")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            break

        qv = embedder.embed(query)
        results = store.search(qv, top_k=5)

        print(f"\n📊 Top {len(results)} 结果:")
        print("-" * 60)
        for i, (chunk, score) in enumerate(results, 1):
            score_pct = score * 100
            print(f"  [{i}] 相似度: {score_pct:.1f}%")
            print(f"      {chunk[:200]}{'…' if len(chunk) > 200 else ''}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="RagTest — 轻量 RAG 检索实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/rag_search.py --load data/demo.txt --query "Agent 是什么"
  python src/rag_search.py --load data/demo.txt --query "向量数据库" --mode api
  python src/rag_search.py --load data/demo.txt
        """,
    )
    parser.add_argument("--load", type=str, help="加载文本文件建索引")
    parser.add_argument("--query", type=str, help="单次查询（非交互模式）")
    parser.add_argument("--strategy", type=str, default="sentence",
                        choices=["sentence", "size", "paragraph"],
                        help="分块策略")
    parser.add_argument("--text", type=str, help="直接传文本（不走文件）")
    parser.add_argument("--mode", type=str, default="local",
                        choices=["local", "api", "onnx"],
                        help="embedding 模式: local（零依赖）/ api（需 DEEPSEEK_API_KEY）/ onnx（真实模型，需先下载）")
    parser.add_argument("--top-k", type=int, default=5, help="返回几条结果")
    parser.add_argument("--dim", type=int, default=256, help="向量维度（local 模式有效）")
    args = parser.parse_args()

    # 获取文本
    text = None
    if args.load:
        with open(args.load, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"📄 加载文件: {args.load} ({len(text)} 字符)")
    elif args.text:
        text = args.text

    if text is None:
        parser.print_help()
        print("\n💡 至少提供 --load 或 --text")
        sys.exit(1)

    # 选择 embedder
    if args.mode == "onnx":
        try:
            embedder = ONNXEmbedder()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print(f"💡 先运行: python scripts/download_embedding_model.py")
            sys.exit(1)
        embedder_cls = embedder.name
    elif args.mode == "api":
        try:
            embedder = ApiEmbedder()
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
        embedder_cls = "DeepSeek API"
    else:
        embedder = LocalEmbedder(dim=args.dim)
        embedder_cls = f"LocalEmbedder (dim={args.dim})"

    print(f"🧠 Embedder: {embedder_cls}")
    print(f"📐 分块策略: {args.strategy}")

    # 建索引
    store = build_index(text, embedder, args.strategy)

    # 单次查询 / 交互
    if args.query:
        print(f"\n🔍 查询: \"{args.query}\"")
        qv = embedder.embed(args.query)
        results = store.search(qv, top_k=args.top_k)

        print(f"📊 Top {len(results)} 结果:")
        print("-" * 60)
        for i, (chunk, score) in enumerate(results, 1):
            print(f"  [{i}] {score*100:.1f}%")
            print(f"      {chunk[:200]}{'…' if len(chunk) > 200 else ''}")
            print()
    else:
        search_loop(store, embedder)


if __name__ == "__main__":
    main()
