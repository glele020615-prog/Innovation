"""
BaseTool 协议
==============
每个工具自称一体：name / description / parameters(JSON Schema) / run() -> dict。
工具自己负责把 numpy 序列化成 list（execute 统一过 jsonable），
并可通过 validate() 做入参校验（含 safe_path 越界检查）。

新增工具 = 在一个 skill 模块里写一个 BaseTool 子类 + 加到该模块的 *_tools 列表，
agent/tools.py 自动汇总，不需要再改注册表以外的代码。
"""

import time
import traceback
from typing import Any, Optional

from agent.util import jsonable, dumps_safe


class BaseTool:
    name: str = ""
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}, "required": []}
    category: str = "general"     # 工具分组：scan / preprocess / feature / predict / report / knowledge / guide
    timeout: int = 30             # 秒，交给 runtime 的 ToolExecutor
    visible: bool = True          # False = 注册但不暴露给 LLM

    # ---------------- 子类实现 ----------------

    def validate(self, args: dict) -> tuple:
        """返回 (cleaned_args, error_message)。默认只做 required / 未知参数检查。"""
        cleaned, error = self._check_schema(args)
        if error:
            return cleaned, error
        return self.validate_args(cleaned)

    def validate_args(self, args: dict) -> tuple:
        """子类覆写：语义级校验 + 类型转换"""
        return args, None

    def run(self, **kwargs) -> dict:
        raise NotImplementedError

    # ---------------- 框架调用 ----------------

    def execute(self, args: Optional[dict] = None, **kwargs) -> dict:
        """统一入口：校验 -> 执行 -> 序列化，任何异常都变成结构化 dict。"""
        started = time.time()
        try:
            merged = dict(args or {})
            merged.update(kwargs)
            cleaned, error = self.validate(merged)
            if error:
                return {"success": False, "error": error, "error_type": "validation",
                        "tool": self.name}
            result = self.run(**cleaned)
            if result is None:
                result = {}
            if not isinstance(result, dict):
                result = {"success": True, "result": result}
            result.setdefault("success", True)
            result.setdefault("tool", self.name)
            result["elapsed_s"] = round(time.time() - started, 3)
            return jsonable(result)
        except Exception as e:
            return {
                "success": False,
                "tool": self.name,
                "error": f"{type(e).__name__}: {e}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()[-1500:],
                "elapsed_s": round(time.time() - started, 3),
            }

    def schema(self) -> dict:
        """OpenAI / DeepSeek function calling 的 schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "parameters": self.parameters,
            },
        }

    def summarize(self, result: dict) -> str:
        """给 UI 的一行摘要（子类可覆写）"""
        if not result.get("success"):
            return f"❌ {str(result.get('error', '失败'))[:60]}"
        for key in ("summary", "prediction_label", "message"):
            if result.get(key):
                return str(result[key])[:60]
        return "✅ 完成"

    # ---------------- 内部 ----------------

    def _check_schema(self, args: dict) -> tuple:
        if not isinstance(args, dict):
            return {}, f"{self.name}: 参数必须是 object"
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        missing = [k for k in required if k not in args or args[k] in (None, "")]
        if missing:
            return args, f"缺少必填参数: {', '.join(missing)}"
        unknown = [k for k in args if k not in props]
        if unknown:
            # 容错：未知参数直接丢弃而不是报错（LLM 偶尔会多给）
            args = {k: v for k, v in args.items() if k in props}
        return args, None


def build_map(tools: list) -> dict:
    """把 BaseTool 列表变成 {name: tool}"""
    return {t.name: t for t in tools}


def tool_schemas(tools: list) -> list:
    return [t.schema() for t in tools if t.visible]


def execute_tool(tools: dict, name: str, args: dict) -> dict:
    tool = tools.get(name)
    if tool is None:
        return {"success": False, "error": f"未知工具: {name}", "error_type": "unknown_tool"}
    return tool.execute(args or {})


class FunctionTool(BaseTool):
    """把一个普通函数快速包成工具（用于已有纯函数的迁移）"""

    def __init__(self, func, name: str = "", description: str = "",
                 parameters: Optional[dict] = None, category: str = "general",
                 timeout: int = 30):
        self._func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        self.parameters = parameters or {"type": "object", "properties": {}, "required": []}
        self.category = category
        self.timeout = timeout

    def run(self, **kwargs) -> dict:
        return self._func(**kwargs)  # type: ignore[operator]


__all__ = ["BaseTool", "FunctionTool", "build_map", "tool_schemas", "execute_tool", "dumps_safe"]
