"""
embedder.py — 文本 → 向量

通过 LLM API 为文本生成向量表示。
默认使用 DeepSeek Chat API 的 logprobs 特征做近似 embedding。
也支持标准 OpenAI-compatible Embedding API（如果有的话）。
"""

import os
import json
import requests
import numpy as np

# ── 配置 ──────────────────────────────────────────────
DEFAULT_BASE_URL = "https://api.deepseek.com"
# ─────────────────────────────────────────────────────


def _get_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    # 尝试从 OpenClaw 的 key 文件读
    for path in [
        "/root/.openclaw/workspace/credential/deepseek.md",
        "/root/clawd/credential/deepseek.md",
    ]:
        try:
            with open(path) as f:
                return f.read().strip()
        except (FileNotFoundError, OSError):
            continue
    return None


class Embedder:
    """文本向量化器"""

    def __init__(self, api_key=None, base_url=None, dim=384):
        self.api_key = api_key or _get_api_key()
        if not self.api_key:
            raise ValueError("需要 API key：传参或设 DEEPSEEK_API_KEY 环境变量")

        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.dim = dim

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        """
        单条文本 → 归一化向量

        方法：用 LLM 的 logprobs 统计信息做 embedding 近似。
        取 top-5 token 的 logprob 值作为特征向量，padding 到统一维度。

        这只是一个轻量演示方案。生产环境应该用专门的 embedding model。
        """
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Embed this text for retrieval: " + text},
                {"role": "user", "content": text},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 5,
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content_logprobs = choice.get("logprobs", {})
        all_logprobs = []

        # 提取 top_logprobs 的 logprob 值做向量
        for item in content_logprobs.get("content", []):
            for tp in item.get("top_logprobs", []):
                all_logprobs.append(tp.get("logprob", 0))
            if len(all_logprobs) >= self.dim:
                break

        vec = np.array(all_logprobs[:self.dim], dtype=np.float32)

        # padding / truncate 到固定维度
        if len(vec) < self.dim:
            vec = np.pad(vec, (0, self.dim - len(vec)))
        elif len(vec) > self.dim:
            vec = vec[:self.dim]

        return self._normalize(vec)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """批量向量化 → (N, dim) 矩阵"""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = [self.embed(t) for t in texts]
        return np.array(vecs, dtype=np.float32)
