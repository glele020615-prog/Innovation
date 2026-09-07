"""
AAL116 脑区名称与知识 —— 单一数据源
=====================================
数据来自 agent/knowledge/brain_regions.json（唯一来源）。
报告渲染（individual_report_page）与 Agent 的 get_region_info / RAG 共用同一份数据，
避免中文名与 AD 知识出现两套互相打架的硬编码。
"""

import os
import sys
import json

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_FILE = os.path.join(_PROJECT_ROOT, "agent", "knowledge", "brain_regions.json")


def _resource_file() -> str:
    base = getattr(sys, "_MEIPASS", _PROJECT_ROOT)
    bundled = os.path.join(base, "agent", "knowledge", "brain_regions.json")
    if os.path.exists(_DATA_FILE):
        return _DATA_FILE
    return bundled


_cache: dict = {}


def load_brain_regions() -> dict:
    """返回完整数据：{schema_version, name_map, regions}"""
    global _cache
    if _cache:
        return _cache
    path = _resource_file()
    try:
        with open(path, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    except Exception:
        _cache = {"schema_version": 1, "name_map": {}, "regions": {}}
    return _cache


def reload_brain_regions() -> dict:
    global _cache
    _cache = {}
    return load_brain_regions()


# 向后兼容：individual_report_page 直接用 NODE_NAME_CN
NODE_NAME_CN: dict = load_brain_regions().get("name_map", {})


def base_name(name: str) -> str:
    """'HIP.L' -> 'HIP'"""
    if not name:
        return ""
    base = str(name)
    if "." in base:
        base, _suffix = base.rsplit(".", 1)
    return base


def region_side(name: str) -> str:
    if not name or "." not in str(name):
        return ""
    suffix = str(name).rsplit(".", 1)[-1]
    return "左侧" if suffix == "L" else ("右侧" if suffix == "R" else "")


def to_cn_region(name: str) -> str:
    """'HIP.L' -> '左侧海马'；未知则原样返回"""
    if not name:
        return name
    side = region_side(name)
    base = base_name(name)
    cn = NODE_NAME_CN.get(base, base)
    return f"{side}{cn}" if side else cn


def find_region(query: str) -> tuple:
    """按 缩写 / 中文名 / 英文名 / 别名 查找脑区。返回 (abbr, info_dict)。"""
    if not query:
        return "", {}
    data = load_brain_regions()
    regions = data.get("regions", {})
    name_map = data.get("name_map", {})

    base = base_name(str(query).strip())
    if base in regions:
        return base, regions[base]

    lowered = base.lower()
    for abbr, info in regions.items():
        if abbr.lower() == lowered:
            return abbr, info
        if str(info.get("cn_name", "")).lower() == lowered:
            return abbr, info
        if str(info.get("en_name", "")).lower() == lowered:
            return abbr, info
        for alias in info.get("aliases", []) or []:
            if str(alias).lower() == lowered:
                return abbr, info

    # 只认识中文名（无详细知识条目）时，至少返回中文名
    cn = name_map.get(base)
    if cn:
        return base, {"abbr": base, "cn_name": cn, "ad_relevance": "暂无详细 AD 关联知识"}
    for abbr, cn_name in name_map.items():
        if cn_name == base:
            return abbr, {"abbr": abbr, "cn_name": cn_name, "ad_relevance": "暂无详细 AD 关联知识"}
    return "", {}


def region_display(name: str) -> dict:
    """返回 {name, abbr, cn_name, side, ad_relevance, functions, ...}"""
    base = base_name(name)
    abbr, info = find_region(name)
    if not info:
        return {"name": name, "abbr": base, "cn_name": name_map_fallback(name),
                "side": region_side(name), "ad_relevance": "暂无详细 AD 关联知识"}
    out = dict(info)
    out["name"] = name
    out["side"] = region_side(name)
    return out


def name_map_fallback(name: str) -> str:
    return NODE_NAME_CN.get(base_name(name), base_name(name) or name)


__all__ = ["NODE_NAME_CN", "to_cn_region", "find_region", "region_display",
           "load_brain_regions", "reload_brain_regions", "base_name", "region_side"]
