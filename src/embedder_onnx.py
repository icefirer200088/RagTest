"""
embedder_onnx.py — 用 ONNX Runtime 跑真实 embedding 模型

使用 sentence-transformers/all-MiniLM-L6-v2 的 ONNX 版本。
纯 CPU 推理，不需要 PyTorch，体积 26MB。
"""

import json
import os
import numpy as np
from pathlib import Path

# ONNX Runtime
import onnxruntime as ort


# ── 配置 ──────────────────────────────────────────────
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = Path(__file__).parent.parent / "models"

# ─────────────────────────────────────────────────────


class ONNXEmbedder:
    """
    使用 ONNX 加载真实 sentence-transformer embedding 模型。
    输出 384 维向量（all-MiniLM-L6-v2 的标准输出维度），已 L2 归一化。
    """

    def __init__(self, model_dir=None, dim=384):
        self.dim = dim
        self.model_dir = Path(model_dir) if model_dir else self._find_model()

        # 加载 ONNX session
        onnx_path = self.model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            # 尝试候补路径
            candidates = list(self.model_dir.rglob("*.onnx"))
            if candidates:
                onnx_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"找不到 ONNX 模型文件。模型应下载到 {self.model_dir}，"
                    "包含 onnx/model.onnx"
                )

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        # 加载 tokenizer config
        self._load_tokenizer()

        # 模型配置
        self.input_name = self.session.get_inputs()[0].name
        self.max_length = 256  # all-MiniLM 默认 max_seq_length

        print(f"✅ ONNX 模型加载完成: {onnx_path.name} ({self.dim}维)")

    def _find_model(self):
        """自动查找模型缓存目录"""
        # 优先已下载的 snapshot
        snapshots = list(MODEL_CACHE.glob("models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*/"))
        if snapshots:
            return snapshots[0]

        # 尝试直接路径
        if MODEL_CACHE.exists():
            return MODEL_CACHE

        raise FileNotFoundError(
            "模型未下载。在 RagTest/ 根目录运行:\n"
            "  python scripts/download_embedding_model.py"
        )

    def _load_tokenizer(self):
        """加载 tokenizer 配置（简单规则分词，不用依赖 tokenizers 库）"""
        # 找 tokenizer.json
        tk_paths = list(self.model_dir.rglob("tokenizer.json"))
        tk_config_paths = list(self.model_dir.rglob("tokenizer_config.json"))

        if tk_config_paths:
            with open(tk_config_paths[0]) as f:
                self.tk_config = json.load(f)
        else:
            self.tk_config = {}

        # vocab
        if tk_paths:
            with open(tk_paths[0]) as f:
                tk_data = json.load(f)
            self.vocab = tk_data.get("model", {}).get("vocab", {})
        else:
            self.vocab = {}

        # 默认特殊 token
        self.cls_token_id = 101  # BERT-style [CLS]
        self.sep_token_id = 102  # [SEP]
        self.pad_token_id = 0    # [PAD]

    def _tokenize(self, text: str) -> dict:
        """
        用 vocab 做简单的 tokenization。
        不是完美的 BPE tokenizer（完整 tokenizer 需要 tokenizers 库），
        但足够做 embedding demo。

        真实场景建议: pip install tokenizers
        """
        if not self.vocab:
            return self._basic_tokenize(text)

        # 简化 tokenization: 空格分词 + subword fallback
        words = text.strip().lower().split()
        input_ids = [self.cls_token_id]

        for word in words:
            if word in self.vocab:
                input_ids.append(self.vocab[word])
            else:
                # fallback: 字符级 subword
                for ch in word:
                    if ch in self.vocab:
                        input_ids.append(self.vocab[ch])

        input_ids.append(self.sep_token_id)

        # truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length - 1] + [self.sep_token_id]

        # attention mask
        attention_mask = [1] * len(input_ids)

        # padding
        # token_type_ids（0 = sentence A）
        token_type_ids = [0] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len

        return {
            self.input_name: np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.array([token_type_ids], dtype=np.int64),
        }

    def _basic_tokenize(self, text: str) -> dict:
        """无 vocab 时的兜底 tokenization（用字符 ID）"""
        input_ids = [101] + [ord(c) % 30000 for c in text.strip()[:self.max_length - 2]] + [102]
        attention_mask = [1] * len(input_ids)

        token_type_ids = [0] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len

        return {
            self.input_name: np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.array([token_type_ids], dtype=np.int64),
        }

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        """文本 → 归一化 384 维向量"""
        inputs = self._tokenize(text)
        # ONNX 推理
        outputs = self.session.run(None, inputs)
        # all-MiniLM 的输出是 (batch, seq_len, hidden)，取 [CLS] token
        emb = outputs[0][0, 0, :].astype(np.float32)
        return self._normalize(emb)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = [self.embed(t) for t in texts]
        return np.array(vecs, dtype=np.float32)

    @property
    def name(self):
        return f"ONNX/{MODEL_ID} (dim={self.dim})"
