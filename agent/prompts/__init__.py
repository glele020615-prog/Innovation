"""
Prompt 模板（按语言拆分）
==========================
用法：from agent.prompts import get_prompt
     prompt = get_prompt("zh", tool_groups=describe_groups())

新增语言 = 加一个 xx.py（提供 SYSTEM_PROMPT 常量），无需改调用方。
"""

from typing import Optional

DEFAULT_LANG = "zh"
SUPPORTED = ("zh", "en")


def get_prompt(lang: str = DEFAULT_LANG, tool_groups: str = "", **kwargs) -> str:
    lang = (lang or DEFAULT_LANG).lower()
    if lang not in SUPPORTED:
        lang = DEFAULT_LANG
    module = __import__(f"agent.prompts.{lang}", fromlist=["SYSTEM_PROMPT"])
    template = getattr(module, "SYSTEM_PROMPT", "")
    if "{tool_groups}" in template:
        template = template.replace("{tool_groups}", tool_groups or "")
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def available_languages() -> tuple:
    return SUPPORTED


__all__ = ["get_prompt", "available_languages", "DEFAULT_LANG", "SUPPORTED"]
