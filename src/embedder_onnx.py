"""
embedder_onnx.py — 用 ONNX Runtime 跑真实 embedding 模型

支持多模型源，自动检测已缓存的 ONNX 模型。
默认优先使用 shibing624/text2vec-base-chinese（中文专用，768维）。

用法:
    from embedder_onnx import ONNXEmbedder
    emb = ONNXEmbedder()  # 自动找本地缓存或默认模型
    vec = emb.embed("你好")
"""

import json
import os
import numpy as np
from pathlib import Path

import onnxruntime as ort
from tokenizers import Tokenizer


# ── 支持的模型 ─────────────────────────────────────────
MODELS = {
    "text2vec-chinese": {
        "repo_id": "shibing624/text2vec-base-chinese",
        "onnx_path": "onnx/model.onnx",
        "dim": 768,
        "desc": "中文（推荐，~380MB）",
    },
    "all-MiniLM-L6": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "onnx_path": "onnx/model.onnx",
        "dim": 384,
        "desc": "英文（26MB，中文也能用但差一些）",
    },
}
# ──────────────────────────────────────────────────────


def _find_model_in_cache(repo_id, model_root) -> Path | None:
    """在 HF 缓存目录中寻找已下载的 ONNX 模型"""
    # HF 缓存格式: models/models--{org}--{name}/snapshots/{hash}/
    repo_dir = repo_id.replace("/", "--")
    snap_dir = model_root / f"models--{repo_dir}" / "snapshots"
    if not snap_dir.exists():
        return None

    for snapshot in snap_dir.iterdir():
        onnx_file = snapshot / "onnx" / "model.onnx"
        if onnx_file.exists():
            return snapshot  # 返回 snapshot 根目录
        # 直接检查
        onnx_file2 = snapshot / "model.onnx"
        if onnx_file2.exists():
            return snapshot

    return None


def _guess_chinese_best(model_root) -> tuple:
    """智能选择可用的最佳中文模型"""
    for name, info in MODELS.items():
        snap = _find_model_in_cache(info["repo_id"], model_root)
        if snap:
            onnx_path = snap / info["onnx_path"]
            if onnx_path.exists():
                return snap, onnx_path, info["dim"]

    # fallback: 直接目录
    for name, info in MODELS.items():
        onnx_path = model_root / info["repo_id"] / info["onnx_path"]
        if onnx_path.exists():
            return model_root / info["repo_id"], onnx_path, info["dim"]

    return None, None, None


class ONNXEmbedder:
    """
    使用 ONNX 运行真实 embedding 模型。

    自动检测本地缓存的 ONNX 模型，按中文优先策略加载。
    输出向量已 L2 归一化，可用于余弦相似度（点积）。
    """

    def __init__(self, model_root=None, model_name=None):
        self.model_root = Path(model_root) if model_root else self._default_model_root()

        if model_name:
            info = MODELS.get(model_name)
            if not info:
                raise ValueError(f"未知模型: {model_name}，可选: {list(MODELS.keys())}")
            self._load(info)
        else:
            # 自动检测
            snap, onnx_path, dim = _guess_chinese_best(self.model_root)
            if onnx_path:
                self.dim = dim
                self._load_model(onnx_path)
                self._load_tokenizer(snap)
            else:
                raise FileNotFoundError(
                    f"找不到缓存的 ONNX 模型。先运行:\n"
                    f"  python scripts/download_embedding_model.py"
                )

    def _default_model_root(self) -> Path:
        return Path(__file__).parent.parent / "models"

    def _load(self, info):
        self.dim = info["dim"]
        # 搜索缓存
        snap = _find_model_in_cache(info["repo_id"], self.model_root)
        if snap:
            onnx_path = snap / info["onnx_path"]
            if onnx_path.exists():
                self._load_model(onnx_path)
                self._load_tokenizer(snap)
                return

        # 直接路径
        onnx_path = self.model_root / info["repo_id"] / info["onnx_path"]
        if onnx_path.exists():
            self._load_model(onnx_path)
            self._load_tokenizer(onnx_path.parent.parent)
            return

        raise FileNotFoundError(f"模型 {info['repo_id']} 未下载")

    def _load_model(self, onnx_path: Path):
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.max_length = 512 if "text2vec" in str(onnx_path) else 256
        print(f"✅ ONNX 模型: {onnx_path.parent.parent.name} (dim={self.dim})")

    def _load_tokenizer(self, model_dir: Path):
        """加载 tokenizer"""
        # 尝试多个路径
        candidates = [
            model_dir / "tokenizer.json",
            model_dir / "onnx" / "tokenizer.json",
            model_dir.parent / "tokenizer.json",
            model_dir.parent / "onnx" / "tokenizer.json",
        ]
        tk_path = None
        for p in candidates:
            if p.exists():
                tk_path = p
                break

        if tk_path:
            self.tokenizer = Tokenizer.from_file(str(tk_path))
        else:
            self.tokenizer = None

        if self.tokenizer:
            # 设置默认参数
            self.tokenizer.enable_truncation(self.max_length)
            self.tokenizer.enable_padding(
                pad_id=0,
                pad_token="[PAD]",
                length=self.max_length,
            )
            print(f"✅ Tokenizer 加载完成 (max_length={self.max_length})")
        else:
            print("⚠️  未找到 tokenizer.json，将使用 fallback tokenization")

    def _tokenize(self, text: str) -> dict:
        """文本 → ONNX 输入"""
        if self.tokenizer:
            encoding = self.tokenizer.encode(text)
            input_ids = encoding.ids
            attention_mask = encoding.attention_mask
            token_type_ids = encoding.type_ids if hasattr(encoding, 'type_ids') else [0] * len(input_ids)

            # padding
            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [0] * pad_len
                attention_mask += [0] * pad_len
                token_type_ids += [0] * pad_len
        else:
            # fallback
            input_ids = [101] + [ord(c) % 30000 for c in text.strip()[:self.max_length - 2]] + [102]
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [0] * pad_len
                attention_mask += [0] * pad_len
                token_type_ids += [0] * pad_len

        feed = {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array([token_type_ids], dtype=np.int64)

        return feed

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        """文本 → 归一化向量"""
        feed = self._tokenize(text)
        outputs = self.session.run(None, feed)
        # [CLS] token 的输出作为句子向量
        emb = outputs[0][0, 0, :].astype(np.float32)
        return self._normalize(emb)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = [self.embed(t) for t in texts]
        return np.array(vecs, dtype=np.float32)

    @property
    def name(self):
        return f"ONNX (dim={self.dim})"
