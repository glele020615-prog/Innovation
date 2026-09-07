"""
脱敏：落盘前把所有绝对路径 / 患者 ID 换成占位符
=================================================
会话 JSON、日志在写入磁盘前必须过 redact()。
原始值存到 ~/.智影agent/redaction_map.json（仅本地，600 权限），用于回放。

占位符格式：**带短哈希 token**
    F:/Innovation/Data/pMCI/FC/matr_002_S_0729.mat -> <病例路径#3f2a1b>
    002_S_0729                                    -> <ID#9c8e77>
这样既能让 LLM / 阅读者看懂语义，又能按 token 精确还原（unredact_text）。
"""

import os
import re
import json
import hashlib

from agent.security import CONFIG_DIR

MAP_PATH = os.path.join(CONFIG_DIR, "redaction_map.json")

PH_PATH = "病例路径"
PH_ID = "ID"
# 形如 <病例路径#3f2a1b> / <ID#9c8e77>（6~64 位：兼容早期 10 位 token 的历史数据）
_TOKEN_PLACEHOLDER_RE = re.compile(r"<(病例路径|ID)#([0-9a-f]{6,64})>")

MAX_MAP_ENTRIES = 5000      # 映射表上限，超出丢最旧的，避免无限增长


def _placeholder(kind: str, key: str) -> str:
    """统一的占位符构造：redact / reveal_candidates 必须走同一处，否则 token 长度会打架"""
    label = PH_PATH if kind == "path" else PH_ID
    return f"<{label}#{key}>"

# 中文标点也要当分隔符，否则 "xxx.mat，报告存到 F:/yyy" 会被整段吞成一个"路径"
_CJK_PUNCT = "，。；：！？、（）【】《》“”‘’…—·"
_PATH_STOP = r"\\\/\s:*?\"<>|\n\r" + _CJK_PUNCT

# Windows: F:\xxx\yyy ; C:/Users/...
_WIN_PATH_RE = re.compile(
    r"(?i)\b[A-Za-z]:[\\/](?:[^" + _PATH_STOP + r"]+[\\/])*[^" + _PATH_STOP + r"]*")
# Unix: /Users/... /home/... /data/... 等绝对路径（至少两级）
_UNIX_PATH_RE = re.compile(
    r"/(?:Users|home|root|data|mnt|media|var|tmp|opt|srv)(?:/[^" + _PATH_STOP + r"]*)+")
# 患者标识
_PATIENT_RES = [
    re.compile(r"\b\d{3}_S_\d{4}\b"),            # ADNI: 002_S_0729
    re.compile(r"\bPAT[_-]?[A-Za-z0-9]+\b", re.I),
    re.compile(r"\b\d{3}_S_\d{4}_\d{4}-\d{2}-\d{2}\b"),
]

_map_cache: dict = {}
_dirty = False


def _load_map() -> dict:
    global _map_cache
    if _map_cache:
        return _map_cache
    try:
        if os.path.exists(MAP_PATH):
            with open(MAP_PATH, "r", encoding="utf-8") as f:
                _map_cache = json.load(f)
    except Exception:
        _map_cache = {}
    return _map_cache


def _save_map():
    global _dirty
    if not _dirty:
        return
    try:
        with open(MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(_map_cache, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(MAP_PATH, 0o600)
        except Exception:
            pass
        _dirty = False
    except Exception:
        pass


def _token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6]


def _remember(raw: str, placeholder_kind: str) -> str:
    """记住原文，返回带 token 的占位符（同一原文永远得到同一个 token）"""
    mapping = _load_map()
    key = _token(raw)
    if key not in mapping:
        mapping[key] = {"kind": placeholder_kind, "value": raw}
        global _dirty
        _dirty = True
        _trim_map(mapping)
    return _placeholder(placeholder_kind, key)


def _trim_map(mapping: dict):
    """映射表超过上限时丢掉最早写入的条目（dict 保持插入顺序）"""
    if len(mapping) <= MAX_MAP_ENTRIES:
        return
    overflow = len(mapping) - MAX_MAP_ENTRIES
    for key in list(mapping.keys())[:overflow]:
        mapping.pop(key, None)


def redact_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    def sub_path(m):
        raw = m.group(0)
        if len(raw) < 4:
            return raw
        return _remember(raw, "path")

    out = _WIN_PATH_RE.sub(sub_path, text)
    out = _UNIX_PATH_RE.sub(sub_path, out)

    def sub_pid(m):
        return _remember(m.group(0), "patient_id")

    for rx in _PATIENT_RES:
        out = rx.sub(sub_pid, out)

    _save_map()
    return out


def redact(obj, _depth: int = 0):
    """递归脱敏任意结构（dict / list / str）"""
    if _depth > 8:
        return obj
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v, _depth + 1) for v in obj]
    return obj


def unredact_text(text: str) -> str:
    """回放：按 token 精确还原占位符（path 与 ID 都还原）。

    占位符形如 <病例路径#3f2a1b> / <ID#9c8e77>；映射表里查不到的 token 原样保留。
    """
    if not isinstance(text, str) or not text:
        return text
    mapping = _load_map()

    def _sub(m):
        item = mapping.get(m.group(2))
        if not item:
            return m.group(0)
        return str(item.get("value", m.group(0)))

    return _TOKEN_PLACEHOLDER_RE.sub(_sub, text)


def reveal_map() -> dict:
    """完整的 token → 原文映射（合规回放用，含患者信息，勿外传）"""
    return _load_map()


def reveal(token: str) -> str:
    """按 token（或带尖括号的占位符）查原文；查不到返回空串。"""
    if not token:
        return ""
    token = str(token).strip().strip("<>").split("#")[-1]
    item = _load_map().get(token)
    return str(item.get("value", "")) if item else ""


def reveal_candidates(kind: str = "") -> list:
    """列出某一类（path / patient_id）的所有原文，供医生挑认：
    "这个 <ID#9c8e77> 当时是哪位患者" """
    items = []
    for key, item in _load_map().items():
        if kind and item.get("kind") != kind:
            continue
        items.append({"token": key, "kind": item.get("kind", ""),
                      "value": item.get("value", ""),
                      "placeholder": _placeholder(item.get("kind", ""), key)})
    return items
