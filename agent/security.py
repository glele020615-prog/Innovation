"""
路径安全：白名单 + 越界拒绝
=============================
所有 Agent 工具的入参路径在执行前必须过 safe_path()。
默认只放行 PROJECT_ROOT 与用户配置目录，其余目录需显式写入
~/.智影agent/allowed_roots.json 授权。
"""

import os
import re
import sys
import json

from agent.util import PROJECT_ROOT, resolve_fuzzy_path, CONFIG_DIR, resolve_config_dir

# ==================== 用户数据目录 ====================
# 位置可迁移：AGENT_HOME 环境变量 > 项目根 agent_home.txt（便携）> ~/.智影agent
# 唯一真源在 agent/util.py:CONFIG_DIR，这里只是转发，避免两处各自解析

ALLOWED_ROOTS_FILE = os.path.join(CONFIG_DIR, "allowed_roots.json")

# ==================== 黑名单 ====================

_UNC_RE = re.compile(r"^\\\\+")            # \\server\share
_WIN_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")
_UNIX_HOME_RE = re.compile(r"^/(Users|home|root|etc|proc|sys|dev|var|boot)(/|$)")

_FORBIDDEN = [
    r"C:\Windows", r"C:\Program Files", r"C:\Program Files (x86)",
    r"C:\ProgramData", r"C:\$Recycle.Bin", r"C:\System32",
    "/Windows", "/System", "/Library", "/etc", "/proc", "/sys", "/dev", "/boot",
]


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def load_allowed_roots() -> list[str]:
    roots = [PROJECT_ROOT, CONFIG_DIR]
    try:
        if os.path.exists(ALLOWED_ROOTS_FILE):
            with open(ALLOWED_ROOTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                roots.extend([str(p) for p in data if isinstance(p, str)])
    except Exception:
        pass
    # 去重并保持顺序
    seen, out = set(), []
    for r in roots:
        n = _norm(r)
        if n not in seen:
            seen.add(n)
            out.append(r)
    return out


def save_allowed_roots(roots: list[str]):
    try:
        with open(ALLOWED_ROOTS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(set(roots)), f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def authorize_directory(path: str) -> dict:
    """把目录加入白名单（供设置页调用，不给 LLM 暴露）"""
    if not path or not os.path.isdir(path):
        return {"success": False, "error": f"目录不存在: {path}"}
    roots = load_allowed_roots()
    if _norm(path) not in [_norm(r) for r in roots]:
        roots.append(os.path.abspath(path))
        save_allowed_roots(roots)
    return {"success": True, "authorized": load_allowed_roots()}


def is_unc(path: str) -> bool:
    return bool(_UNC_RE.match(str(path)))


def safe_path(path, must_exist: bool = False, allow_missing: bool = True,
              prefer: str = "") -> tuple:
    """校验路径是否在白名单内。

    prefer="fc"/"bold"：模糊解析有多个候选时优先选含该关键字的。
    返回 (ok: bool, resolved: str | error_message: str)
    """
    if path is None:
        return False, "路径为空"
    text = str(path).strip().strip('"').strip("'")
    if not text:
        return False, "路径为空"

    if is_unc(text):
        return False, "路径越界：拒绝访问 UNC / 网络共享路径"

    # 相对路径按 PROJECT_ROOT 解析（LLM 常给 Data/xxx 这种相对形式）
    if not os.path.isabs(text) and not _WIN_ABS_RE.match(text):
        text = os.path.join(PROJECT_ROOT, text)

    # 模糊路径补全：Data/pMCI/.../matr_002_S_0729.mat
    if not os.path.exists(text):
        text = resolve_fuzzy_path(text, prefer=prefer)

    resolved = _norm(text)

    for bad in _FORBIDDEN:
        if resolved.startswith(_norm(bad)) or resolved.lower().startswith(bad.lower()):
            return False, f"路径越界：禁止访问系统目录（{bad}）"

    roots = load_allowed_roots()
    for root in roots:
        root_n = _norm(root)
        if resolved == root_n or resolved.startswith(root_n + os.sep):
            if must_exist and not os.path.exists(resolved):
                return False, f"路径不存在: {text}"
            return True, resolved

    if allow_missing:
        # 不在白名单但也不是系统目录：明确提示需要授权
        return False, (
            f"路径越界：{text} 不在授权目录内。"
            f"请在「设置」中授权该目录，或把数据放到项目目录下（当前白名单：{roots}）"
        )
    return False, f"路径越界：{text}"


def safe_path_or_error(path, **kwargs) -> str:
    """便捷版：越界直接抛 ValueError，交给 BaseTool.execute 转成结构化错误"""
    ok, value = safe_path(path, **kwargs)
    if not ok:
        raise ValueError(value)
    return value
