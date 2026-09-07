"""
医学知识子系统：RAG 检索 + AAL116 脑区查询
============================================
两者共用 agent/knowledge/brain_regions.json，消除"硬编码 + md"双源。
"""

import os

from agent.skills.base import BaseTool
from brain_region_names import (
    load_brain_regions, find_region, to_cn_region, base_name, region_side,
)

_KNOWLEDGE_HINT = (
    "知识库覆盖：NIA-AA 诊断标准（ATN 分类、MCI/AD 诊断标准、进展风险分层）、"
    "AAL116 脑区与 AD 关联、临床随访与干预建议。"
)


def search_knowledge(query: str, top_k: int = 3, min_score: float = 0.0) -> dict:
    """检索知识库；低于相关度阈值返回空 + 明确提示。"""
    try:
        from agent.rag import get_rag
        rag = get_rag()
        return rag.search(query, top_k=top_k, min_score=min_score)
    except ImportError:
        return {"success": False, "error": "RAG 模块未加载"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def get_region_info(region_name: str) -> dict:
    """查询脑区中文名 + AD 关联（数据来自 brain_regions.json）"""
    name = str(region_name or "").strip()
    if not name:
        return {"success": False, "error": "region_name 不能为空"}
    abbr, info = find_region(name)
    if not info:
        return {
            "success": True,
            "found": False,
            "name": name,
            "cn_name": "未收录",
            "ad_relevance": f"暂无 {name} 与 AD 直接关联的详细知识，可参考 AAL116 图谱文档。",
            "source": "agent/knowledge/brain_regions.json",
        }
    out = {
        "success": True,
        "found": True,
        "name": name,
        "abbr": abbr,
        "cn_name": info.get("cn_name", to_cn_region(name)),
        "side": region_side(name),
        "en_name": info.get("en_name", ""),
        "functions": info.get("functions", ""),
        "ad_relevance": info.get("ad_relevance", "暂无详细 AD 关联知识"),
        "source": "agent/knowledge/brain_regions.json",
    }
    if info.get("aal_index"):
        out["aal_index"] = info["aal_index"]
    if info.get("rs_fmri"):
        out["rs_fmri"] = info["rs_fmri"]
    if info.get("clinical"):
        out["clinical"] = info["clinical"]
    return out


def list_regions(category: str = "") -> dict:
    """列出已收录详细知识的脑区（供 LLM 知道能查什么）"""
    data = load_brain_regions()
    regions = data.get("regions", {})
    items = [
        {"abbr": abbr, "cn_name": info.get("cn_name", ""),
         "has_detail": bool(info.get("functions") or info.get("rs_fmri"))}
        for abbr, info in sorted(regions.items())
    ]
    return {
        "success": True,
        "total": len(items),
        "regions": items,
        "note": "未列出的脑区也可用 get_region_info 查中文名（详细 AD 知识可能为空）",
        "source": "agent/knowledge/brain_regions.json",
    }


# ==================== 工具 ====================

class SearchKnowledgeTool(BaseTool):
    name = "search_knowledge"
    category = "knowledge"
    description = (
        "在 AD/MCI 医学知识库中检索权威内容。" + _KNOWLEDGE_HINT +
        "回答时须标注 [来源: 文件#章节]；若返回 '知识库未覆盖'，"
        "应直接说明未覆盖，不要用记忆编造医学事实。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索语句，如「MCI 随访频率建议」「海马与 AD 的关系」"},
            "top_k": {"type": "integer", "description": "返回结果条数，默认 3", "default": 3},
        },
        "required": ["query"],
    }
    timeout = 60

    def run(self, query: str, top_k: int = 3) -> dict:
        return search_knowledge(query, top_k=top_k)

    def summarize(self, result: dict) -> str:
        if not result.get("success"):
            return f"❌ {str(result.get('error', ''))[:60]}"
        n = len(result.get("results", []) or [])
        return f"命中 {n} 条（{result.get('engine', '?')}）" if n else "知识库未覆盖"


class GetRegionInfoTool(BaseTool):
    name = "get_region_info"
    category = "knowledge"
    description = (
        "查询 AAL116 脑区的中文名、功能及与阿尔茨海默症的已知关联"
        "（数据源：agent/knowledge/brain_regions.json）。"
        "当用户或工具输出提到具体脑区（如 HIP.L、PCG.R、AMYG.L、PreCG.L）时调用，"
        "用返回的知识解释该脑区为何重要。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "region_name": {"type": "string",
                            "description": "脑区名，支持缩写(HIP.L)、中文名(海马)、英文名(Hippocampus)"},
        },
        "required": ["region_name"],
    }

    def run(self, region_name: str) -> dict:
        return get_region_info(region_name)

    def summarize(self, result: dict) -> str:
        return f"{result.get('cn_name', '-')}（{result.get('abbr', '-')}）"


class ListRegionsTool(BaseTool):
    name = "list_regions"
    category = "knowledge"
    description = "列出知识库中已收录详细 AD 关联知识的 AAL116 脑区（缩写 + 中文名）。"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs) -> dict:
        return list_regions()

    def summarize(self, result: dict) -> str:
        return f"共 {result.get('total', 0)} 个脑区"


knowledge_tools = [SearchKnowledgeTool(), GetRegionInfoTool(), ListRegionsTool()]

__all__ = ["knowledge_tools", "search_knowledge", "get_region_info", "list_regions"]
