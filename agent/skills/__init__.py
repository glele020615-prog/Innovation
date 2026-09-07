"""
Agent Skills —— 按子系统切分的工具集
=====================================
每个子模块导出一个 *_tools 列表（元素为 BaseTool 子类实例），
agent/tools.py 只负责汇总，不写任何实现。
"""

from agent.skills.base import BaseTool, FunctionTool

# 子系统 → 模块（导入顺序即工具在 prompt 中的分组顺序）
SUBSYSTEMS = [
    ("scan", "数据加载 / 扫描记录"),
    ("preprocess", "预处理"),
    ("feature_extract", "特征提取"),
    ("predict", "预测与可解释性"),
    ("report", "报告生成"),
    ("knowledge", "医学知识检索"),
    ("guide", "流程引导"),
]

__all__ = ["BaseTool", "FunctionTool", "SUBSYSTEMS"]
