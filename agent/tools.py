"""
Agent 工具注册表（只做汇总，不写实现）
=======================================
新增/移除工具 = 改对应 skill 模块的 *_tools 列表，本文件无需改动。

约定：
  TOOLS        -> BaseTool 实例列表（唯一真源）
  TOOL_MAP     -> {name: BaseTool}
  TOOL_SCHEMAS -> 给 LLM function calling 的 schema 列表
"""

from functools import lru_cache

from agent.skills.base import build_map, tool_schemas
from agent.skills.scan import ScanStore, scan_tools
from agent.skills.preprocess import preprocess_tools
from agent.skills.feature_extract import feature_tools
from agent.skills.predict import predict_tools
from agent.skills.report import report_tools
from agent.skills.knowledge import knowledge_tools
from agent.skills.pipeline import guide_tools

# ==================== 注册表 ====================

TOOLS = (
    scan_tools
    + preprocess_tools
    + feature_tools
    + predict_tools
    + report_tools
    + knowledge_tools
    + guide_tools
)

TOOL_MAP = build_map(TOOLS)
TOOL_SCHEMAS = tool_schemas(TOOLS)

GROUP_LABELS = {
    "scan": "数据加载（扫描记录）",
    "preprocess": "预处理",
    "feature": "特征提取",
    "predict": "预测与可解释性",
    "report": "报告生成",
    "knowledge": "医学知识",
    "guide": "流程引导",
}


def grouped_tools(visible_only: bool = True) -> dict:
    """按子系统分组：{group: [tool, ...]}"""
    groups: dict = {}
    for tool in TOOLS:
        if visible_only and not tool.visible:
            continue
        groups.setdefault(tool.category, []).append(tool)
    return groups


@lru_cache(maxsize=1)
def describe_groups() -> str:
    """给 system prompt 用的"先选大类，再列具名"目录

    TOOLS 是模块级常量、构造后不变，所以结果可以缓存；
    若运行时动态改过 TOOLS，调用 describe_groups.cache_clear() 失效即可。
    """
    lines = []
    for group, tools in grouped_tools().items():
        names = "、".join(t.name for t in tools)
        lines.append(f"- {GROUP_LABELS.get(group, group)}：{names}")
    return "\n".join(lines)


def get_tool(name: str):
    return TOOL_MAP.get(name)


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """统一执行入口（供 runtime 与旧代码调用）"""
    tool = TOOL_MAP.get(tool_name)
    if tool is None:
        return {"success": False, "error": f"未知工具: {tool_name}",
                "error_type": "unknown_tool",
                "hint": f"可用工具: {', '.join(TOOL_MAP)}"}
    return tool.execute(arguments or {})


# ==================== 向后兼容的函数式入口 ====================

def _call(name: str, **kwargs) -> dict:
    return execute_tool(name, kwargs)


def list_scans(**kwargs) -> dict:
    return _call("list_scans", **kwargs)


def describe_pipeline(**kwargs) -> dict:
    return _call("describe_pipeline", **kwargs)


def get_region_info(region_name: str) -> dict:
    return _call("get_region_info", region_name=region_name)


def check_ready(fc_path: str = "", bold_path: str = "") -> dict:
    return _call("check_ready", fc_path=fc_path, bold_path=bold_path)


def search_knowledge(query: str, top_k: int = 3) -> dict:
    return _call("search_knowledge", query=query, top_k=top_k)


def predict_ad_risk(fc_path: str, bold_path: str, fc_var_name: str = "",
                    bold_var_name: str = "") -> dict:
    return _call("predict_ad_risk", fc_path=fc_path, bold_path=bold_path,
                 fc_var_name=fc_var_name, bold_var_name=bold_var_name)


__all__ = [
    "TOOLS", "TOOL_MAP", "TOOL_SCHEMAS", "execute_tool", "get_tool",
    "grouped_tools", "describe_groups", "GROUP_LABELS", "ScanStore",
    "list_scans", "describe_pipeline", "get_region_info", "check_ready",
    "search_knowledge", "predict_ad_risk",
]
