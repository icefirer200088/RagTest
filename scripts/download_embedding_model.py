#!/usr/bin/env python3
"""
download_embedding_model.py — 下载 ONNX embedding 模型

用法：
    python scripts/download_embedding_model.py

下载 sentence-transformers/all-MiniLM-L6-v2 的 ONNX 版本到 models/ 目录。
模型大小约 26MB，不需要 PyTorch，直接 ONNX Runtime 推理。
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
import sys

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).parent.parent / "models"

FILES = [
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "special_tokens_map.json",
]


def main():
    print(f"📥 下载 {MODEL_ID} 的 ONNX 版本到 {CACHE_DIR}")
    for fname in FILES:
        print(f"  → {fname}")
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=fname,
            cache_dir=str(CACHE_DIR),
        )
    print("✅ 模型下载完成！")
    print(f"  位置: {CACHE_DIR / 'models--sentence-transformers--all-MiniLM-L6-v2'}")


if __name__ == "__main__":
    main()
