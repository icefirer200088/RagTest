"""
rag_search.py — RAG 检索主入口

用法：
    # 建索引并立即查询
    python src/rag_search.py data/demo.txt "向量检索是什么" --mode onnx

    # 建索引并保存
    python src/rag_search.py data/demo.txt --mode onnx --save my_index.h5

    # 加载已有索引 + 查询
    python src/rag_search.py --load-index my_index.h5 "向量检索"

    # 端到端 RAG（检索 + LLM 回答）
    python src/rag_search.py data/demo.txt "向量检索" --mode onnx --generate

流程：
    1. 读文件 → 分块 → 算向量 → 建索引（或加载已有索引）
    2. 查询 → 检索 top-k
    3. (可选) 检索结果 + Prompt → LLM 生成回答
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import Embedder as ApiEmbedder
from src.embedder_local import LocalEmbedder
from src.chunk import chunk_by_sentence, chunk_by_size, chunk_by_paragraph
from src.vector_store import VectorStore


def build_index(text: str, embedder, store: VectorStore,
                strategy: str = "sentence"):
    """文件文本 → 分块 → 向量化 → 写入 store"""
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

    store.add_batch(chunks, vectors)
    print(f"✅ 索引完成，共 {store.size} 条记录")


def do_search(store: VectorStore, embedder, query: str, top_k: int):
    """执行检索，返回结果列表"""
    qv = embedder.embed(query)
    return store.search(qv, top_k=top_k)


def print_results(results: list[tuple[str, float]]):
    """打印检索结果"""
    for i, (chunk, score) in enumerate(results, 1):
        print(f"  [{i}] {score*100:.1f}%")
        print(f"      {chunk[:200]}{'…' if len(chunk) > 200 else ''}")
        print()


def generate_answer(results: list[tuple[str, float]], query: str):
    """
    检索结果 + 用户问题 → LLM 生成回答。

    用 DeepSeek Chat API（同 OpenClaw 使用的配置）。
    """
    import requests
    import json

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    # 从配置文件兜底
    if not api_key:
        for path in [
            "/root/.openclaw/workspace/credential/deepseek.md",
            "/root/clawd/credential/deepseek.md",
        ]:
            try:
                with open(path) as f:
                    api_key = f.read().strip()
                    break
            except (FileNotFoundError, OSError):
                continue

    if not api_key:
        print("⚠️  无 DEEPSEEK_API_KEY，跳过 LLM 回答")
        print("设置: export DEEPSEEK_API_KEY='sk-...'")
        return

    # 构建上下文
    context_parts = []
    for i, (chunk, score) in enumerate(results, 1):
        context_parts.append(f"[参考{i}] (相关度 {score*100:.1f}%)\n{chunk}")
    context = "\n\n".join(context_parts)

    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {
                "role": "system",
                "content": "你是一个智能问答助手。根据检索到的参考资料回答用户问题。"
                           "如果参考资料不足以支撑回答，请如实说明。"
                           "引用参考资料时用 [参考1] 标注来源。"
            },
            {
                "role": "user",
                "content": f"## 参考资料\n\n{context}\n\n## 问题\n{query}"
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]

    print("\n" + "=" * 60)
    print(f"🤖 LLM 回答:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="RagTest — RAG 检索实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/rag_search.py data/demo.txt "Agent 多角色对话"
  python src/rag_search.py data/demo.txt --mode onnx --save index.h5
  python src/rag_search.py data/demo.txt "向量检索" --generate
  python src/rag_search.py --load-index index.h5 "向量检索" --generate
        """,
    )
    parser.add_argument("file", nargs="?", type=str, help="文本文件建索引")
    parser.add_argument("query", nargs="?", type=str, help="查询文本（或使用 --query）")
    parser.add_argument("--query", type=str, dest="query2", help="查询文本（替代位置参数）")
    parser.add_argument("--mode", type=str, default="local",
                        choices=["local", "api", "onnx"],
                        help="embedding 模式")
    parser.add_argument("--strategy", type=str, default="sentence",
                        choices=["sentence", "size", "paragraph"])
    parser.add_argument("--save", type=str, help="保存索引到文件 (.h5)")
    parser.add_argument("--load-index", type=str, help="从文件加载索引 (.h5)")
    parser.add_argument("--generate", action="store_true",
                        help="端到端 RAG: 检索后让 LLM 生成回答")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()

    # ── Embedder ──
    embedder = None
    embedder_cls = ""

    if args.mode == "onnx":
        from src.embedder_onnx import ONNXEmbedder
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
        embedder = LocalEmbedder()
        embedder_cls = f"LocalEmbedder"

    print(f"🧠 Embedder: {embedder_cls}")

    # ── VectorStore ──
    store = None

    # 加载已有索引
    if args.load_index:
        store = VectorStore.load(args.load_index)
        idx_dim = store.get_dim()
        print(f"📂 加载索引: {args.load_index} ({store.size} 条, {idx_dim}维)")
        # 索引维度不匹配时自动切换到 onnx embedder
        if args.mode == "local" and idx_dim != 256:
            print(f"⚠️  索引为 {idx_dim}维，自动切换 onnx embedder")
            from src.embedder_onnx import ONNXEmbedder
            try:
                embedder = ONNXEmbedder()
                embedder_cls = embedder.name
            except FileNotFoundError:
                print(f"❌ 需要 ONNX 模型，先下载")
                sys.exit(1)

    # 从文件建索引
    if args.file:
        import os as _os
        path = _os.path.expanduser(args.file)
        if not _os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            sys.exit(1)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"📄 文件: {path} ({len(text)} 字符)")

        if store is None:
            store = VectorStore()

        build_index(text, embedder, store, args.strategy)

        if args.save:
            store.save(args.save)

    if store is None:
        parser.print_help()
        print("\n💡 提供 --file 建索引，或 --load-index 加载已有索引")
        sys.exit(1)

    # ── 查询 ──
    query = args.query or args.query2

    if args.interactive or (query is None and args.file):
        # 交互模式
        print("\n🔍 输入查询（空行退出）:")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
            print(f"\n🔍 \"{q}\"")
            results = do_search(store, embedder, q, args.top_k)
            print_results(results)

            if args.generate:
                generate_answer(results, q)

    elif query:
        print(f"\n🔍 \"{query}\"")
        results = do_search(store, embedder, query, args.top_k)
        print_results(results)

        if args.generate:
            generate_answer(results, query)
    else:
        print("💡 未提供查询，索引已建好。下次用 --load-index 加载后查询。")


if __name__ == "__main__":
    main()
