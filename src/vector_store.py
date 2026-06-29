"""
vector_store.py — 向量存储 + 最近邻检索

纯 numpy 实现，不需要数据库。
支持余弦距离（归一化后 = 点积）、top-k 检索。
"""

import numpy as np


class VectorStore:
    """
    内存向量存储。

    存：
        store.add("文本片段", vector)
    查：
        results = store.search(query_vector, top_k=5)
        → [(chunk, score), ...]
    """

    def __init__(self):
        self.chunks: list[str] = []
        # 用 list 累积，检索时一次性转矩阵
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunk: str, vector: np.ndarray):
        """添加一条记录"""
        self.chunks.append(chunk)
        self._vectors.append(vector.ravel().astype(np.float32))
        self._matrix = None  # 标记脏

    def add_batch(self, chunks: list[str], vectors: np.ndarray):
        """批量添加 chunks → (N, dim), vectors → (N, dim)"""
        self.chunks.extend(chunks)
        for i in range(vectors.shape[0]):
            self._vectors.append(vectors[i].ravel().astype(np.float32))
        self._matrix = None

    def _build_matrix(self):
        """把全部 vectors 变矩阵（按需重建）"""
        if self._matrix is None and self._vectors:
            self._matrix = np.array(self._vectors, dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """
        检索 top_k 最相似的 chunk。

        Returns:
            [(chunk_text, similarity_score), ...]
            score 范围 [-1, 1]，1 = 完全相同
        """
        if self.size == 0:
            return []

        self._build_matrix()
        qv = query_vec.ravel().astype(np.float32).reshape(1, -1)

        # 余弦相似度（向量已归一化 → 点积）
        scores = np.dot(self._matrix, qv.T).ravel()

        # 取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    def clear(self):
        """清空所有数据"""
        self.chunks.clear()
        self._vectors.clear()
        self._matrix = None
