"""
通用工具：JSON 安全序列化 / 路径 / 时间
=======================================
Agent 链路里所有要交给 LLM 或落盘的 dict 都必须先过 jsonable()，
否则 numpy 标量、Path、set 会在 json.dumps 时炸掉。
"""

import os
import re
import sys
import math
import time
import datetime
from pathlib import Path

import numpy as np

# jsonable 对超长序列的截断阈值（防止 116x116 矩阵灌爆上下文）
MAX_SEQ = 512


def _project_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PROJECT_ROOT = _project_root()


# ==================== 用户数据目录（可迁移 / 可便携） ====================
#
# 解析优先级：
#   1) 环境变量 AGENT_HOME            —— 临时指定、脚本/多实例场景
#   2) 项目根/exe 目录的 agent_home.txt —— 便携模式：整个目录拷到别的机器，
#                                         配置/日志/会话跟着一起走（内容是一个路径）
#   3) ~/.智影agent                    —— 默认
#
# 目录下的东西：config.json(密钥) / logs / sessions.json / redaction_map.json
#               / allowed_roots.json / user_memory.json / index

AGENT_HOME_ENV = "AGENT_HOME"
AGENT_HOME_MARKER = "agent_home.txt"
DEFAULT_DIRNAME = ".智影agent"
FALLBACK_DIRNAME = ".zhiying_agent"


def default_agent_home() -> str:
    """默认位置：用户主目录。C 盘不是必须的，可用 AGENT_HOME 改到任意盘。"""
    home = os.path.expanduser("~")
    primary = os.path.join(home, DEFAULT_DIRNAME)
    # 中文目录名在少数环境（旧版 Windows / 非 UTF-8 文件系统）可能创建失败
    try:
        os.makedirs(primary, exist_ok=True)
        return primary
    except Exception:
        fallback = os.path.join(home, FALLBACK_DIRNAME)
        os.makedirs(fallback, exist_ok=True)
        return fallback


def agent_home_marker_path() -> str:
    return os.path.join(PROJECT_ROOT, AGENT_HOME_MARKER)


def resolve_config_dir(create: bool = True) -> str:
    """按优先级解析用户数据目录"""
    # 1) 环境变量
    override = os.getenv(AGENT_HOME_ENV, "").strip().strip('"')
    if override:
        base = os.path.abspath(override)
        if create:
            try:
                os.makedirs(base, exist_ok=True)
            except Exception:
                pass
        return base

    # 2) 便携模式：项目根/exe 目录下的 agent_home.txt
    marker = agent_home_marker_path()
    if os.path.exists(marker):
        try:
            with open(marker, "r", encoding="utf-8") as f:
                value = f.read().strip().strip('"')
            if value:
                # 相对路径按"程序所在目录"解析 —— 这样写 "AgentData" 就能做 U 盘便携版
                base = value if os.path.isabs(value) else os.path.join(PROJECT_ROOT, value)
                base = os.path.abspath(base)
                if create:
                    os.makedirs(base, exist_ok=True)
                return base
        except Exception:
            pass

    # 3) 默认
    return default_agent_home() if create else os.path.join(
        os.path.expanduser("~"), DEFAULT_DIRNAME)


# 进程内唯一的数据目录（所有模块都从这里取，保证一致）
CONFIG_DIR = resolve_config_dir()


def set_agent_home(path: str, portable: bool = True) -> dict:
    """把用户数据目录改到指定位置。

    portable=True  → 写入 项目根/agent_home.txt，配置跟着程序目录走，
                     整目录拷到别的机器依然生效（推荐，便于迁移）。
    portable=False → 只提示用环境变量 AGENT_HOME（不落任何文件）。
    """
    target = os.path.abspath(str(path or "").strip())
    if not target:
        return {"success": False, "error": "路径不能为空"}
    try:
        os.makedirs(target, exist_ok=True)
        probe = os.path.join(target, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("1")
        os.remove(probe)
    except Exception as e:
        return {"success": False, "error": f"目录不可写: {target} ({e})"}

    if portable:
        try:
            with open(agent_home_marker_path(), "w", encoding="utf-8") as f:
                f.write(target)
        except Exception as e:
            return {"success": False, "error": f"无法写入 {agent_home_marker_path()}: {e}"}

    return {
        "success": True,
        "path": target,
        "portable": portable,
        "marker": agent_home_marker_path() if portable else "",
        "note": "重启程序后生效（新目录会在下次启动时使用）",
    }


def clear_agent_home() -> dict:
    """恢复默认位置（删掉便携标记）"""
    marker = agent_home_marker_path()
    try:
        if os.path.exists(marker):
            os.remove(marker)
        os.environ.pop(AGENT_HOME_ENV, None)
        return {"success": True, "path": default_agent_home()}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def agent_home_info() -> dict:
    """给 UI 用：当前数据目录 + 来源 + 各子项大小"""
    source = "default"
    if os.getenv(AGENT_HOME_ENV, "").strip():
        source = "env"
    elif os.path.exists(agent_home_marker_path()):
        source = "portable"

    path = resolve_config_dir()
    items = {}
    total = 0
    for name in ("config.json", "logs", "sessions.json", "redaction_map.json",
                 "allowed_roots.json", "user_memory.json", "index"):
        p = os.path.join(path, name)
        if not os.path.exists(p):
            continue
        size = _dir_size(p) if os.path.isdir(p) else os.path.getsize(p)
        items[name] = round(size / 1024, 1)
        total += size

    return {
        "path": path,
        "source": source,
        "source_label": {"env": "环境变量 AGENT_HOME",
                         "portable": f"便携模式（{AGENT_HOME_MARKER}）",
                         "default": "默认（用户主目录）"}.get(source, source),
        "marker_path": agent_home_marker_path(),
        "env_var": AGENT_HOME_ENV,
        "items_kb": items,
        "total_kb": round(total / 1024, 1),
    }


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except Exception:
                pass
    return total


def migrate_agent_home(new_dir: str, portable: bool = True) -> dict:
    """把现有数据目录整体搬到新位置（配置/日志/会话/脱敏表/索引）。

    先拷后删：目标写成功才动源目录，中途失败不会丢数据。
    """
    import shutil

    old = resolve_config_dir()
    new = os.path.abspath(str(new_dir or "").strip())
    if not new:
        return {"success": False, "error": "目标路径不能为空"}
    if os.path.normcase(os.path.normpath(old)) == os.path.normcase(os.path.normpath(new)):
        return {"success": False, "error": "目标与当前位置相同"}

    try:
        os.makedirs(new, exist_ok=True)
        moved = []
        for name in sorted(os.listdir(old)):
            src = os.path.join(old, name)
            dst = os.path.join(new, name)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    # 目录已存在：合并（逐个文件覆盖）
                    for sub_root, _d, files in os.walk(src):
                        rel = os.path.relpath(sub_root, src)
                        target_dir = os.path.join(dst, rel) if rel != "." else dst
                        os.makedirs(target_dir, exist_ok=True)
                        for f in files:
                            shutil.copy2(os.path.join(sub_root, f),
                                         os.path.join(target_dir, f))
                else:
                    shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            moved.append(name)
    except Exception as e:
        return {"success": False, "error": f"迁移失败: {type(e).__name__}: {e}",
                "note": "源目录未改动，数据未丢失"}

    result = set_agent_home(new, portable=portable)
    if not result.get("success"):
        return result
    result.update({"moved": moved, "old_path": old})
    try:
        shutil.rmtree(old)
        result["old_removed"] = True
    except Exception:
        result["old_removed"] = False
        result["note"] = (f"新位置已生效，但旧目录未能自动删除：{old}\n"
                          f"确认新目录内容无误后可手动删除。")
    return result


def jsonable(obj, max_seq: int = MAX_SEQ, _depth: int = 0):
    """把任意对象递归转换为 JSON 安全类型。

    - numpy 标量 -> python 标量（nan/inf -> None）
    - numpy 数组 / torch tensor -> list（超长截断）
    - Path -> str
    - set/tuple -> list
    - datetime -> ISO 字符串
    - 不可识别对象 -> str(obj)
    """
    if _depth > 8:
        return str(obj)[:200]

    if obj is None or isinstance(obj, (bool, str)):
        return obj

    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.complexfloating, complex)):
        return str(obj)

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return jsonable(obj.item(), max_seq, _depth + 1)
        flat = obj.reshape(-1)
        truncated = flat.shape[0] > max_seq
        head = flat[:max_seq]
        data = [jsonable(v, max_seq, _depth + 1) for v in head.tolist()]
        return {
            "_type": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "truncated": truncated,
            "data": data,
        } if truncated else data

    # torch tensor（避免硬依赖 torch，用鸭子类型判断）
    if type(obj).__module__.startswith("torch") and hasattr(obj, "detach"):
        try:
            return jsonable(obj.detach().cpu().numpy(), max_seq, _depth + 1)
        except Exception:
            return str(obj)[:200]

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset, tuple)):
        return jsonable(list(obj), max_seq, _depth + 1)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            try:
                out[key] = jsonable(v, max_seq, _depth + 1)
            except Exception as e:  # pragma: no cover - 兜底
                out[key] = f"<unserializable: {e}>"
        return out
    if isinstance(obj, (list,)):
        truncated = len(obj) > max_seq
        seq = [jsonable(v, max_seq, _depth + 1) for v in (obj[:max_seq] if truncated else obj)]
        if truncated:
            return {"_type": "list", "length": len(obj), "truncated": True, "data": seq}
        return seq
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")[:2000]

    return str(obj)[:2000]


def dumps_safe(obj, **kwargs) -> str:
    """json.dumps 的安全包装：先 jsonable 再序列化，绝不抛异常。"""
    import json
    kwargs.setdefault("ensure_ascii", False)
    try:
        return json.dumps(jsonable(obj), **kwargs)
    except Exception:
        return json.dumps({"success": False, "error": "结果序列化失败"}, ensure_ascii=False)


def truncate_text(text: str, limit: int = 400) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + f"...(共{len(text)}字)"


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def today() -> str:
    return time.strftime("%Y-%m-%d")


# ==================== 受试者 / 病例 ID 的提取 ====================

_ADNI_RE = re.compile(r"(?<!\d)\d{3}_S_\d{4}", re.IGNORECASE)   # matr_002_S_0729_2011-08-16
_PAT_RE = re.compile(r"\bPAT_[A-Za-z0-9]+\b", re.IGNORECASE)
_SUB_RE = re.compile(r"\bsub-[A-Za-z0-9]+", re.IGNORECASE)


def extract_subject_id(path: str) -> str:
    """从文件名里提取受试者标识：ADNI 式 > sub-xx > 文件名主干"""
    if not path:
        return "unknown"
    base = os.path.basename(str(path))
    stem = re.sub(r"\.(nii\.gz|tar\.gz)$", "", base, flags=re.IGNORECASE)
    stem = os.path.splitext(stem)[0]
    m = _ADNI_RE.search(stem)
    if m:
        return m.group(0).upper()      # ADNI 编号统一大写：002_S_0729
    m = _SUB_RE.search(stem)
    if m:
        return m.group(0)
    # 去掉常见的 _bold / _fc / matr_ 前后缀
    stem = re.sub(r"^(matr_|fc_|bold_)", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(_bold|_fc|_preproc)$", "", stem, flags=re.IGNORECASE)
    return stem or "unknown"


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(text)).strip("_") or "case"


def resolve_fuzzy_path(path: str, search_root: str = "", prefer: str = "") -> str:
    """把 LLM 常给的模糊路径补全成真实路径。

    支持三类模糊输入：
      1. Data/pMCI/.../matr_002_S_0729.mat（省略号 / 通配符）
      2. 只有文件名
      3. 文件名缺后缀（matr_002_S_0729.mat vs matr_002_S_0729_2011-08-16.mat）
    prefer="fc"/"bold" 时，多个候选中优先路径含该关键字的那个。
    找不到时原样返回（交给调用方报错）。
    """
    if not path:
        return path
    text = str(path).strip().strip('"').strip("'")
    if os.path.exists(text):
        return text

    abs_text = text if os.path.isabs(text) else os.path.join(PROJECT_ROOT, text)
    if os.path.exists(abs_text):
        return abs_text

    # 1) 省略号 / 通配符
    if "..." in abs_text or "*" in abs_text or "?" in abs_text:
        import glob as _glob
        pattern = abs_text.replace("...", "*")
        matches = [m for m in _glob.glob(pattern, recursive=True) if os.path.isfile(m)]
        if matches:
            return _pick(matches, prefer)

    # 2) 按文件名找（含"缺日期后缀"的前缀匹配）
    name = os.path.basename(text)
    if name:
        import glob as _glob
        roots = [search_root] if search_root else [
            os.path.join(PROJECT_ROOT, "Data"), PROJECT_ROOT]
        stem = os.path.splitext(name)[0]
        exact, prefix = [], []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for current, _dirs, files in os.walk(root):
                for f in files:
                    full = os.path.join(current, f)
                    if f == name:
                        exact.append(full)
                    elif stem and f.startswith(stem):
                        prefix.append(full)
            if exact or prefix:
                break
        for bucket in (exact, prefix):
            if bucket:
                return _pick(bucket, prefer)
    return text


def _pick(candidates: list, prefer: str = "") -> str:
    """多个候选时：prefer 命中优先，否则取排序第一个"""
    if not candidates:
        return ""
    if prefer:
        low = prefer.lower()
        hits = [c for c in candidates if low in os.path.basename(c).lower()
                or low in c.lower().replace("\\", "/").split("/")[-2:][0].lower()]
        if hits:
            return sorted(hits)[0]
    return sorted(candidates)[0]
