"""
RAG 检索引擎（有索引版）
==========================
- 索引产物：agent/knowledge/.index/{chunks.json, vectors.npy, meta.json}
- 启动时若索引不存在或源文件 mtime 变化 → 自动 rebuild，否则直接 load
- 优先 sentence-transformers 语义检索，未安装则回退 TF-IDF
- 相关度阈值：embedding 0.35 / TF-IDF 0.05，低于阈值返回「知识库未覆盖」
- 检索结果带 source_file / section / anchors，供 LLM 标注 [来源: 文件#章节]
"""

import os
import json
import time
from typing import Optional

import numpy as np

from agent.knowledge.build_index import (
    ensure_index, build_index, load_index, is_stale, mtime_snapshot,
    load_embedding_model, build_tfidf_matrix, _tokenize,
)

DEFAULT_MIN_SCORE = {"embedding": 0.35, "tfidf": 0.05}
CONTENT_LIMIT = 600


class MedicalRAG:
    def __init__(self, auto_build: bool = True):
        self.chunks: list = []
        self.vectors: Optional[np.ndarray] = None
        self.meta: dict = {}
        self._model = None
        self._vocab: dict = {}
        self._idf = None
        self._mode = "empty"
        self._last_check = 0.0
        if auto_build:
            self.refresh()

    # ---------------- 索引 ----------------

    def refresh(self, force: bool = False):
        chunks, vectors, meta = ensure_index(force=force)
        self.chunks = chunks or []
        self.vectors = vectors
        self.meta = meta or {}
        self._mode = self.meta.get("mode", "empty")
        if self._mode == "embedding":
            if self._model is None:
                self._model = load_embedding_model()
            if self._model is None:      # 模型不可用 → 降级为 TF-IDF
                self._downgrade_to_tfidf()
        elif self._mode == "tfidf":
            self._load_tfidf_vocab()
        self._last_check = time.time()

    def _load_tfidf_vocab(self):
        path = os.path.join(self.meta.get("index_dir", ""), "tfidf_vocab.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._vocab = data.get("vocab", {})
            idf = data.get("idf") or []
            self._idf = np.array(idf, dtype=np.float32) if idf else None
        except Exception:
            self._vocab, self._idf = {}, None

    def _downgrade_to_tfidf(self):
        texts = [f"[{c['source_file']}] {c['section']}\n{c['text']}" for c in self.chunks]
        matrix, idf, vocab = build_tfidf_matrix(texts)
        self.vectors = matrix
        self._vocab, self._idf = vocab, idf
        self._mode = "tfidf"

    def maybe_rebuild(self):
        """监听 mtime：变了就重建（每次 search 前最多检查一次，间隔 5 秒）"""
        if time.time() - self._last_check < 5:
            return
        if is_stale(self.meta) or not self.chunks:
            self.refresh(force=True)
        else:
            self._last_check = time.time()

    # ---------------- 查询向量 ----------------

    def _query_vector(self, query: str) -> Optional[np.ndarray]:
        if self._mode == "embedding" and self._model is not None:
            return self._model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        if self._mode == "tfidf":
            if not self._vocab:
                self._load_tfidf_vocab()
            if not self._vocab:
                return None
            q_vec = np.zeros(len(self._vocab), dtype=np.float32)
            for tok in _tokenize(query):
                if tok in self._vocab:
                    q_vec[self._vocab[tok]] += 1
            if self._idf is not None and len(self._idf) == len(self._vocab):
                q_vec = q_vec * self._idf
            norm = np.linalg.norm(q_vec)
            if norm == 0:
                return None
            return q_vec / norm
        return None

    # ---------------- 检索 ----------------

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> dict:
        try:
            self.maybe_rebuild()
        except Exception:
            pass

        if not self.chunks or self.vectors is None:
            return {
                "success": False,
                "error": "知识库为空，请检查 agent/knowledge/ 目录",
                "results": [],
            }

        threshold = min_score if min_score > 0 else DEFAULT_MIN_SCORE.get(self._mode, 0.05)
        try:
            q_vec = self._query_vector(query)
            if q_vec is None:
                # 查询词完全不在词表 / 检索器不可用 → 按「未覆盖」处理，别报成引擎故障
                return {
                    "success": True,
                    "engine": self._mode,
                    "query": query,
                    "threshold": threshold,
                    "covered": False,
                    "results": [],
                    "message": (
                        "知识库未覆盖此问题（查询词与知识库没有任何重合）。"
                        "请直接告诉用户「知识库未覆盖」，不要凭记忆编造医学事实。"
                    ),
                }
            scores = np.dot(self.vectors, q_vec)
            top_indices = np.argsort(scores)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score < threshold:
                    continue
                chunk = self.chunks[idx]
                results.append({
                    "content": chunk.get("text", "")[:CONTENT_LIMIT],
                    "score": round(score, 3),
                    "source_file": chunk.get("source_file", ""),
                    "section": chunk.get("section", ""),
                    "anchors": chunk.get("anchors", []),
                    "citation": f"[来源: {chunk.get('source_file', '')}#{chunk.get('section', '')}]",
                    "index": int(idx),
                })

            if not results:
                return {
                    "success": True,
                    "engine": self._mode,
                    "query": query,
                    "results": [],
                    "threshold": threshold,
                    "covered": False,
                    "message": (
                        f"知识库未覆盖此问题（最高相关度低于阈值 {threshold}）。"
                        "请直接告诉用户「知识库未覆盖」，不要凭记忆编造医学事实。"
                    ),
                }

            return {
                "success": True,
                "engine": self._mode,
                "query": query,
                "threshold": threshold,
                "covered": True,
                "total_chunks": len(self.chunks),
                "results": results,
                # 让 LLM 在工具结果里就看到引用要求，而不只依赖 system prompt
                "citation_instruction": (
                    "回答时必须在每个来自知识库的结论后紧跟 citation 字段的原文，"
                    "格式 [来源: 文件名#章节]；一结论一标注，不要合并。"
                ),
            }
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}", "results": []}

    # ---------------- 属性 ----------------

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def engine(self) -> str:
        return self._mode

    @property
    def status(self) -> dict:
        return {
            "engine": self._mode,
            "chunk_count": len(self.chunks),
            "model": self.meta.get("model", ""),
            "dim": self.meta.get("dim", 0),
            "created_at": self.meta.get("created_at", ""),
            "index_dir": self.meta.get("index_dir", ""),
            "sources": sorted({c.get("source_file", "") for c in self.chunks}),
        }


# 全局单例
_rag: Optional[MedicalRAG] = None


def get_rag() -> MedicalRAG:
    global _rag
    if _rag is None:
        _rag = MedicalRAG()
    return _rag


def rebuild_index() -> dict:
    """供设置页 / 命令行手动重建索引"""
    global _rag
    meta = build_index(force=True)
    if _rag is not None:
        _rag.refresh(force=True)
    return meta


if __name__ == "__main__":
    rag = get_rag()
    print("RAG 状态:", json.dumps(rag.status, ensure_ascii=False, indent=2))
    for q in ["MCI 诊断标准", "海马与 AD 的关系", "多久复查一次",
              "Lecanemab 适应症", "aMCI 与 naMCI 的区别"]:
        r = rag.search(q)
        print(f"\n查询: {q}")
        if not r.get("results"):
            print(f"  → {r.get('message') or r.get('error', '无结果')}")
            continue
        for item in r["results"]:
            print(f"  [{item['score']:.3f}] {item['citation']} {item['content'][:70]}...")
