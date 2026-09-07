"""
知识库索引构建
================
把 knowledge/*.md 切成 chunks，算 embeddings，产物落在 .index/：
    chunks.json  — [{id, source_file, section, anchors, text}]
    vectors.npy  — (N, D) float32，已 L2 归一化
    meta.json    — {mode, model, dim, created_at, files:{name: mtime}, chunk_count}

启动时若 .index 不存在（或源文件 mtime 变了）才重建，否则直接 load。
打包后 knowledge/ 只读时，索引写到 ~/.智影agent/index/。
"""

import os
import re
import sys
import json
import time

import numpy as np

_KNOWLEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIRNAME = ".index"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _fallback_index_dir() -> str:
    from agent.security import CONFIG_DIR
    return os.path.join(CONFIG_DIR, "index")


def index_dir() -> str:
    """优先 knowledge/.index；不可写则退回用户目录"""
    primary = os.path.join(_KNOWLEDGE_DIR, INDEX_DIRNAME)
    try:
        os.makedirs(primary, exist_ok=True)
        probe = os.path.join(primary, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("1")
        os.remove(probe)
        return primary
    except Exception:
        d = _fallback_index_dir()
        os.makedirs(d, exist_ok=True)
        return d


def chunks_path() -> str:
    return os.path.join(index_dir(), "chunks.json")


def vectors_path() -> str:
    return os.path.join(index_dir(), "vectors.npy")


def meta_path() -> str:
    return os.path.join(index_dir(), "meta.json")


# ==================== 分块 ====================

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def clean_text(text: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", text)
    return text.strip()


def extract_anchors(text: str) -> list:
    """从 chunk 正文里抽关键词（加粗术语 + 英文缩写），作为检索锚点"""
    anchors = set()
    for term in re.findall(r"\*\*(.+?)\*\*", text):
        if 1 < len(term) <= 20:
            anchors.add(term.strip())
    for term in re.findall(r"\b[A-Z]{2,6}\b", text):
        anchors.add(term)
    for term in re.findall(r"[\u4e00-\u9fff]{2,6}(?:病|症|脑区|回|核|叶|标准|检测)", text):
        anchors.add(term)
    return sorted(anchors)[:12]


def chunk_markdown(text: str, source_file: str) -> list:
    """按 #/##/### 标题切块，保留章节路径"""
    chunks = []
    sections = re.split(r"\n(?=#{1,6}\s)", text)
    stack: dict = {}
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        first_line = sec.split("\n")[0].lstrip("#").strip()
        level = len(sec.split("\n")[0]) - len(sec.split("\n")[0].lstrip("#"))
        level = min(max(level, 1), 6)
        stack[level] = first_line
        for deeper in list(stack.keys()):
            if deeper > level:
                stack.pop(deeper, None)
        path = [stack[k] for k in sorted(stack) if k <= level and stack.get(k)]
        section_path = " > ".join(path)
        body = clean_text(sec)
        if len(body) < 20:
            continue
        # 超长块再按段落二次切分，避免语义被稀释
        if len(body) > 1200:
            parts, buf = [], []
            for para in re.split(r"\n\s*\n", sec):
                buf.append(para)
                if sum(len(b) for b in buf) > 900:
                    parts.append("\n".join(buf))
                    buf = []
            if buf:
                parts.append("\n".join(buf))
        else:
            parts = [sec]

        for idx, part in enumerate(parts):
            content = clean_text(part)
            if len(content) < 20:
                continue
            chunks.append({
                "id": f"{source_file}::{len(chunks)}",
                "source_file": source_file,
                "section": section_path + (f" [{idx + 1}]" if len(parts) > 1 else ""),
                "anchors": extract_anchors(part),
                "text": content,
            })
    return chunks


def scan_knowledge() -> list:
    chunks = []
    if not os.path.exists(_KNOWLEDGE_DIR):
        return chunks
    for fname in sorted(os.listdir(_KNOWLEDGE_DIR)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(_KNOWLEDGE_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()
        chunks.extend(chunk_markdown(text, fname))
    return chunks


def mtime_snapshot() -> dict:
    snap = {}
    if not os.path.exists(_KNOWLEDGE_DIR):
        return snap
    for fname in sorted(os.listdir(_KNOWLEDGE_DIR)):
        if fname.endswith(".md"):
            snap[fname] = round(os.path.getmtime(os.path.join(_KNOWLEDGE_DIR, fname)), 3)
    return snap


# ==================== Embedding / TF-IDF ====================

def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        return None


def embed_texts(model, texts: list) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)


def _tokenize(text: str) -> list:
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}|\d+', text.lower())
    # 滑窗二元组（有重叠）：非重叠切分会把「轻度认知障碍」错位成「度认」，
    # 患者口吻的自然句（如"什么是轻度认知障碍"）会因此检索不到
    tokens += re.findall(r'(?=([\u4e00-\u9fff]{2}))', text)
    return tokens


def build_tfidf_matrix(texts: list) -> tuple:
    """返回 (matrix, idf, vocab)；纯 numpy，零依赖回退"""
    vocab: dict = {}
    for doc in texts:
        for tok in _tokenize(doc):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    if not vocab:
        return np.zeros((len(texts), 1), dtype=np.float32), None, vocab
    tf = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, doc in enumerate(texts):
        for tok in _tokenize(doc):
            if tok in vocab:
                tf[i, vocab[tok]] += 1
    df = np.sum(tf > 0, axis=0)
    idf = np.log((len(texts) + 1) / (df + 1)) + 1
    mat = tf * idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (mat / norms).astype(np.float32), idf, vocab


# ==================== 索引构建 / 加载 ====================

def build_index(force: bool = False) -> dict:
    """构建索引；返回 meta。force=False 且已是最新时直接返回现有 meta。"""
    meta_file = meta_path()
    if not force and os.path.exists(meta_file) and os.path.exists(chunks_path()):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("files") == mtime_snapshot() and meta.get("chunk_count", 0) > 0:
                return meta
        except Exception:
            pass

    chunks = scan_knowledge()
    if not chunks:
        meta = {
            "mode": "empty", "model": "", "dim": 0, "chunk_count": 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": mtime_snapshot(), "index_dir": index_dir(),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    texts = [f"[{c['source_file']}] {c['section']}\n{c['text']}" for c in chunks]
    model = load_embedding_model()

    if model is not None:
        vectors = embed_texts(model, texts)
        mode, model_name = "embedding", EMBEDDING_MODEL
        np.save(vectors_path(), vectors)
    else:
        matrix, idf, vocab = build_tfidf_matrix(texts)
        np.save(vectors_path(), matrix)
        with open(os.path.join(index_dir(), "tfidf_vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"vocab": vocab, "idf": idf.tolist() if idf is not None else []}, f)
        mode, model_name = "tfidf", ""

    with open(chunks_path(), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    meta = {
        "mode": mode,
        "model": model_name,
        "dim": int(np.load(vectors_path()).shape[1]),
        "chunk_count": len(chunks),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": mtime_snapshot(),
        "index_dir": index_dir(),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def load_index() -> tuple:
    """返回 (chunks, vectors, meta)；索引不存在返回 (None, None, None)"""
    if not (os.path.exists(chunks_path()) and os.path.exists(meta_path())
            and os.path.exists(vectors_path())):
        return None, None, None
    try:
        with open(chunks_path(), "r", encoding="utf-8") as f:
            chunks = json.load(f)
        with open(meta_path(), "r", encoding="utf-8") as f:
            meta = json.load(f)
        vectors = np.load(vectors_path())
        if len(chunks) != vectors.shape[0]:
            return None, None, None
        return chunks, vectors, meta
    except Exception:
        return None, None, None


def is_stale(meta: dict) -> bool:
    return not meta or meta.get("files") != mtime_snapshot()


def ensure_index(force: bool = False) -> tuple:
    """保证索引可用：不存在或过期则重建。返回 (chunks, vectors, meta)"""
    chunks, vectors, meta = load_index()
    if chunks is None or is_stale(meta) or force:
        meta = build_index(force=True)
        chunks, vectors, meta = load_index()
    return chunks, vectors, meta


if __name__ == "__main__":
    meta = build_index(force=True)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
