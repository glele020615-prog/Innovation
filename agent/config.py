"""
Agent 配置（统一入口）
========================
优先级：环境变量 > 用户配置 ~/.智影agent/config.json > 项目根 agent_config.json
真实 Key 不进仓库：仓库里只有 agent_config.example.json。

打包后的行为：exe 同目录下的 agent_config.json 仍作为兜底读取。
"""

import os
import sys

from agent.keyring import (
    get_provider_config, resolve_api_key, is_placeholder,
    load_user_config, ensure_user_config, USER_CONFIG, CONFIG_DIR,
)

# ==================== 标准路径（兼容 PyInstaller） ====================

def _get_project_root() -> str:
    """开发版返回项目根目录，打包版返回 exe 所在目录"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_resource_path(*parts: str) -> str:
    """兼容 PyInstaller _MEIPASS 的资源路径"""
    base = getattr(sys, "_MEIPASS", _get_project_root())
    return os.path.join(base, *parts)


PROJECT_ROOT = _get_project_root()
RESOURCE_ROOT = getattr(sys, "_MEIPASS", PROJECT_ROOT)
USER_DIR = CONFIG_DIR

# ==================== LLM 配置 ====================

_cfg = get_provider_config()

LLM_PROVIDER = os.getenv("AGENT_LLM_PROVIDER", _cfg.get("provider", "deepseek"))
LLM_API_KEY = os.getenv("AGENT_API_KEY", _cfg.get("api_key", ""))
LLM_BASE_URL = os.getenv("AGENT_BASE_URL", _cfg.get("base_url", "https://api.deepseek.com"))
LLM_MODEL = os.getenv("AGENT_MODEL", _cfg.get("model", "deepseek-chat"))

# ==================== 通用参数 ====================

_user_cfg = load_user_config()
_params = _user_cfg.get("parameters", {})
MAX_COMPLETION_TOKENS = int(_params.get("max_tokens", 2048))
TEMPERATURE = float(_params.get("temperature", 0.3))

# ==================== Agent 循环参数 ====================

_agent_cfg = _user_cfg.get("agent", {})


def _cli_lang() -> str:
    """支持 python 项目.py --lang=en 这种启动参数"""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--lang" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return ""


MAX_STEPS = int(_agent_cfg.get("max_steps", 8))
TOOL_TIMEOUT = int(_agent_cfg.get("tool_timeout", 30))
LANGUAGE = _cli_lang() or os.getenv("AGENT_LANG", _agent_cfg.get("language", "zh"))

# ==================== 便捷方法 ====================

def get_available_providers() -> list:
    providers = set()
    for cfg in (_user_cfg,):
        providers.update((cfg.get("llm", {}) or {}).get("providers", {}).keys())
    return sorted(providers) or ["deepseek"]


def get_current_config_info() -> dict:
    """返回当前配置信息（Key 已脱敏）"""
    key = LLM_API_KEY
    masked = (key[:8] + "..." + key[-4:]) if len(key) > 12 else ("***" if key else "未设置")
    return {
        "provider": LLM_PROVIDER,
        "api_key_masked": masked,
        "base_url": LLM_BASE_URL,
        "model": LLM_MODEL,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "temperature": TEMPERATURE,
        "max_steps": MAX_STEPS,
        "tool_timeout": TOOL_TIMEOUT,
        "language": LANGUAGE,
        "key_ready": not is_placeholder(key),
        "available_providers": get_available_providers(),
        "user_config_path": USER_CONFIG,
        "project_root": PROJECT_ROOT,
    }


def ensure_config() -> str:
    """确保用户配置存在（首次启动调用）"""
    return ensure_user_config()
