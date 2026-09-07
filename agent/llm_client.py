"""
统一 LLM 客户端
=================
- Key 从 agent.keyring 解析（环境变量 > 用户配置 > 项目配置）
- Key 仍是占位符时**不直接 raise**：标记 unavailable，由 UI 弹友好提示
- 支持 stream=True 与 usage 统计
"""

from typing import Optional

from openai import OpenAI

from agent.keyring import (
    resolve_api_key, is_placeholder, require_api_key, MissingAPIKeyError,
    get_provider_config, ensure_user_config, USER_CONFIG,
)


# ==================== 模型上下文预算 ====================
# 静态表而非启动时探测 API：省一次网络往返，本地/离线模型同样可用。
# 未列出的模型走 _default——宁可估保守（少喂历史），也不要撑爆上下文。
MODEL_CONTEXT_WINDOWS = {
    # DeepSeek
    "deepseek-chat": 64000,
    "deepseek-reasoner": 64000,
    "deepseek-coder": 64000,
    # OpenAI
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 16385,
    # Qwen / 本地
    "qwen": 8000,
    "qwen2.5-7b": 32000,
    "qwen2.5-72b": 128000,
    "qwen-long": 1000000,
    # 兜底
    "_default": 32000,
}

DEFAULT_OUTPUT_RESERVE = 4096      # 常规模型给输出留的额度
REASONER_OUTPUT_RESERVE = 16384    # reasoner 类模型思维链长，得多留
DEFAULT_SAFETY_MARGIN = 1024       # 估算误差 / schema 抖动的安全冗余


def lookup_context_window(model: str) -> int:
    """查表：精确匹配优先，其次取**最长**前缀。

    必须取最长而不是第一个命中，否则 qwen2.5-7b-instruct 会先撞上 qwen（8K）
    而不是 qwen2.5-7b（32K），把 32K 的模型当 8K 用。
    """
    key = str(model or "").strip().lower()
    if not key:
        return MODEL_CONTEXT_WINDOWS["_default"]
    if key in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[key]
    best_size, best_len = 0, 0
    for name, size in MODEL_CONTEXT_WINDOWS.items():
        if name == "_default":
            continue
        if key.startswith(name) and len(name) > best_len:
            best_size, best_len = size, len(name)
    return best_size if best_size else MODEL_CONTEXT_WINDOWS["_default"]


class LLMUnavailableError(RuntimeError):
    """Key 未配置；UI 应弹「请到设置页填写 Key」而不是显示堆栈。"""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


class LLMClient:
    """支持任意 OpenAI 兼容接口的 LLM 客户端"""

    def __init__(self, provider: str = "", auto_check: bool = True):
        cfg = get_provider_config(provider)
        self.provider = cfg.get("provider", "")
        self.model = cfg.get("model", "")
        self.base_url = cfg.get("base_url", "")
        self.api_key, self.key_source = resolve_api_key(provider)
        self.available = not is_placeholder(self.api_key)
        self.client: Optional[OpenAI] = None
        self._apply_model_meta()
        if self.available:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif auto_check:
            # 首次使用：确保用户目录下已生成配置文件，方便用户直接去填
            ensure_user_config()

    def _apply_model_meta(self):
        """按模型名刷新上下文预算（构造与 refresh 后都要调，配置改了模型要跟着变）"""
        self.context_window = lookup_context_window(self.model)
        self.max_output_reserve = (
            REASONER_OUTPUT_RESERVE if "reasoner" in str(self.model or "").lower()
            else DEFAULT_OUTPUT_RESERVE
        )

    # ---------------- 状态 ----------------

    @property
    def input_budget(self) -> int:
        """本次调用可用的 input tokens 上限"""
        return self.context_window - self.max_output_reserve - DEFAULT_SAFETY_MARGIN

    @property
    def status(self) -> dict:
        masked = (self.api_key[:8] + "..." + self.api_key[-4:]) \
            if len(self.api_key) > 12 else ("***" if self.api_key else "未设置")
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_masked": masked,
            "key_source": self.key_source,
            "available": self.available,
            "config_path": USER_CONFIG,
        }

    def _guard(self):
        if not self.available or self.client is None:
            raise LLMUnavailableError(
                "尚未配置可用的 API Key，无法调用大模型。",
                hint=f"点击对话页左下角「⚙ API 配置」填写，"
                     f"或直接编辑：{USER_CONFIG}",
            )

    def refresh(self):
        """配置变更后重建客户端"""
        cfg = get_provider_config(self.provider)
        self.model = cfg.get("model", self.model)
        self.base_url = cfg.get("base_url", self.base_url)
        self.api_key, self.key_source = resolve_api_key(self.provider)
        self.available = not is_placeholder(self.api_key)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.available else None
        self._apply_model_meta()

    # ---------------- 调用 ----------------

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ):
        self._guard()
        from agent.config import TEMPERATURE, MAX_COMPLETION_TOKENS

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": TEMPERATURE if temperature is None else temperature,
            "max_tokens": MAX_COMPLETION_TOKENS if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if stream:
            return self.client.chat.completions.create(**kwargs)
        return self.client.chat.completions.create(**kwargs).model_dump()


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm_client(refresh: bool = False) -> LLMClient:
    global _llm_client
    if _llm_client is None or refresh:
        _llm_client = LLMClient()
    return _llm_client


def llm_status() -> dict:
    """给 UI 用：不抛异常，返回当前可用性"""
    try:
        return get_llm_client().status
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}", "config_path": USER_CONFIG}


def assert_ready() -> tuple:
    """返回 (ok, message)；供 UI 在发消息前检查"""
    st = llm_status()
    if st.get("available"):
        return True, ""
    return False, (
        "尚未配置可用的 API Key。\n\n"
        "点击对话页左下角「⚙ API 配置」即可打开配置文件填写，\n"
        "保存后**再点一次该按钮**生效，无需重启软件。\n\n"
        f"配置文件：{st.get('config_path', USER_CONFIG)}"
    )


__all__ = ["LLMClient", "get_llm_client", "llm_status", "assert_ready",
           "LLMUnavailableError", "MissingAPIKeyError"]
