"""
vector_store.py — 向量存储 + 最近邻检索 + HDF5 持久化

纯 numpy 实现，不需要数据库。
支持余弦距离（归一化后 = 点积）、top-k 检索。
支持 save/load 到 HDF5 文件（用 h5py）。
"""

import json
import numpy as np
from pathlib import Path


class VectorStore:
    """
    内存向量存储 + HDF5 持久化。

    存：
        store.add("文本片段", vector)
    查：
        results = store.search(query_vector, top_k=5)
        → [(chunk, score), ...]
    存盘：
        store.save("my_index.h5")
    读盘：
        store = VectorStore.load("my_index.h5")
    """

    def __init__(self):
        self.chunks: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunk: str, vector: np.ndarray):
        """添加一条记录"""
        self.chunks.append(chunk)
        self._vectors.append(vector.ravel().astype(np.float32))
        self._matrix = None

    def add_batch(self, chunks: list[str], vectors: np.ndarray):
        """批量添加 chunks → (N,), vectors → (N, dim)"""
        self.chunks.extend(chunks)
        for i in range(vectors.shape[0]):
            self._vectors.append(vectors[i].ravel().astype(np.float32))
        self._matrix = None

    def _build_matrix(self):
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
        """
        if self.size == 0:
            return []

        self._build_matrix()
        qv = query_vec.ravel().astype(np.float32).reshape(1, -1)

        # 余弦相似度（向量已归一化 → 点积）
        scores = np.dot(self._matrix, qv.T).ravel()

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

    # ── 持久化 ────────────────────────────────────────

    def save(self, path: str | Path):
        """
        将索引保存到 HDF5 文件。

        结构:
          /chunks      — 字符串数组
          /vectors     — float32 矩阵 (N, dim)
          /metadata    — JSON 字符串（统计信息）
        """
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._build_matrix()

        # 字符串需要编码为字节
        chunk_bytes = [c.encode("utf-8") for c in self.chunks]
        dt = h5py.string_dtype()

        with h5py.File(str(path), "w") as f:
            dt_chunks = h5py.special_dtype(vlen=str)
            f.create_dataset("chunks", data=self.chunks, dtype=dt_chunks)
            f.create_dataset("vectors", data=self._matrix, dtype=np.float32)

            meta = {
                "size": self.size,
                "dim": self._matrix.shape[1] if self._matrix is not None else 0,
            }
            f.attrs["metadata"] = json.dumps(meta)

        print(f"💾 索引已保存: {path} ({self.size} 条)")

    @staticmethod
    def load(path: str | Path) -> "VectorStore":
        """从 HDF5 文件加载索引"""
        import h5py

        path = Path(path)
        store = VectorStore()

        with h5py.File(str(path), "r") as f:
            chunks = f["chunks"][:].tolist()
            # 兼容 bytes 和 str
            store.chunks = [c.decode("utf-8") if isinstance(c, bytes) else c for c in chunks]
            store._matrix = f["vectors"][:].astype(np.float32)

        return store

    def get_chunks(self) -> list[str]:
        """获取所有文本块"""
        return self.chunks.copy()

    def get_dim(self) -> int:
        """获取向量维度"""
        if self._matrix is not None:
            return self._matrix.shape[1]
        if self._vectors:
            return self._vectors[0].shape[0]
        return 0
