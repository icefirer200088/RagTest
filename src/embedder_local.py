"""
embedder_local.py — 纯本地 embedding（不需要 API / 不需要 GPU）
 
用字符 n-gram 哈希 + 词袋方法生成固定维度向量。
虽然不如神经网络 embedding 语义强，但零依赖、可离线、可复现。
适合演示和实验。
"""

import numpy as np
import hashlib


class LocalEmbedder:
    """
    纯本地文本向量化器。

    思路：
      1. 提取文本的字符 2-gram、3-gram 特征
      2. 用哈希将 n-gram 映射到固定维度的特征向量
      3. 加位置加权（靠近开头的字词权重高一点）
    
    dim=256 对 demo 足够了，调高可以获得更好区分度。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _hash_feat(self, gram: str, seed: int = 0) -> int:
        """n-gram → 固定范围的 bin index"""
        h = hashlib.md5(f"{seed}:{gram}".encode()).digest()
        return int.from_bytes(h[:4], "big") % self.dim

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        """
        文本 → 归一化向量
        
        提取 bi-gram + tri-gram 哈希特征，带位置权重。
        """
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        text = text.lower().strip()
        vec = np.zeros(self.dim, dtype=np.float32)

        # 2-gram
        n = len(text)
        for i in range(n - 1):
            gram = text[i:i+2]
            idx = self._hash_feat(gram, seed=1)
            # 位置权重：越靠近开头权重略高
            pos_weight = 1.0 + 0.5 * (1.0 - i / max(n, 1))
            vec[idx] += pos_weight

        # 3-gram（增加区分度）
        for i in range(n - 2):
            gram = text[i:i+3]
            idx = self._hash_feat(gram, seed=2)
            pos_weight = 1.0 + 0.3 * (1.0 - i / max(n, 1))
            vec[idx] += pos_weight

        # 单字（提高对中文的覆盖）
        for i, ch in enumerate(text):
            if ch.strip():
                idx = self._hash_feat(ch, seed=3)
                pos_weight = 1.0 + 0.2 * (1.0 - i / max(n, 1))
                vec[idx] += pos_weight

        return self._normalize(vec)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = [self.embed(t) for t in texts]
        return np.array(vecs, dtype=np.float32)
