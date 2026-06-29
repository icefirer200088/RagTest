#!/usr/bin/env python3
"""
download_embedding_model.py — 下载 ONNX embedding 模型

用法：
    # 下载中文 text2vec（推荐，~380MB）
    python scripts/download_embedding_model.py text2vec-chinese

    # 下载英文 all-MiniLM（轻量，26MB）
    python scripts/download_embedding_model.py all-MiniLM-L6

    # 下载所有模型
    python scripts/download_embedding_model.py all
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

CACHE_DIR = Path(__file__).parent.parent / "models"

MODELS = {
    "text2vec-chinese": {
        "repo_id": "shibing624/text2vec-base-chinese",
        "files": [
            "onnx/model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
        "desc": "中文专用，768维，~380MB（推荐）",
    },
    "all-MiniLM-L6": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "files": [
            "onnx/model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
        "desc": "英文，384维，26MB",
    },
}


def download_model(name, info):
    """下载一个模型"""
    repo_id = info["repo_id"]
    print(f"\n📥 [{name}] {repo_id}")
    print(f"   {info['desc']}")
    for fname in info["files"]:
        print(f"  → {fname}")
        hf_hub_download(repo_id=repo_id, filename=fname, cache_dir=str(CACHE_DIR))
    print(f"  ✅ 完成")


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/download_embedding_model.py <模型名>")
        print(f"\n可用模型:")
        for name, info in MODELS.items():
            print(f"  {name} — {info['desc']}")
        print(f"\n示例:")
        print(f"  python scripts/download_embedding_model.py text2vec-chinese")
        print(f"  python scripts/download_embedding_model.py all-MiniLM-L6")
        sys.exit(1)

    target = sys.argv[1]
    if target == "all":
        for name, info in MODELS.items():
            download_model(name, info)
    elif target in MODELS:
        download_model(target, MODELS[target])
    else:
        print(f"未知模型: {target}")
        print(f"可选: {list(MODELS.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
