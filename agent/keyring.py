"""
密钥管理：环境变量 > 用户配置 > 项目配置 > 占位符
==================================================
仓库里只保留 agent_config.example.json（占位 key），
真实 key 存在 ~/.智影agent/config.json。

注意：本模块**不能**命名为 agent/secrets.py —— 一旦 agent/ 出现在 sys.path 里，
它会遮蔽 Python 标准库的 secrets 模块，导致 numpy.random 等依赖直接 ImportError。
"""

import os
import json
import shutil

from agent.util import PROJECT_ROOT
from agent.security import CONFIG_DIR

USER_CONFIG = os.path.join(CONFIG_DIR, "config.json")
LEGACY_CONFIG = os.path.join(PROJECT_ROOT, "agent_config.json")
EXAMPLE_CONFIG = os.path.join(PROJECT_ROOT, "agent_config.example.json")

PLACEHOLDER_PREFIXES = ("your-", "sk-your", "YOUR_", "<", "${")

DEFAULT_CONFIG = {
    "llm": {
        "provider": "deepseek",
        "providers": {
            "deepseek": {
                "api_key": "your-deepseek-api-key-here",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
            },
            "openai": {
                "api_key": "your-openai-api-key-here",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
            },
            "local": {
                "api_key": "EMPTY",
                "base_url": "http://localhost:8000/v1",
                "model": "qwen",
            },
        },
    },
    "parameters": {"max_tokens": 2048, "temperature": 0.3},
}


class MissingAPIKeyError(RuntimeError):
    """Key 未配置 / 仍是占位符 —— UI 捕获后给友好提示，而不是堆栈。"""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def is_placeholder(key: str) -> bool:
    if not key:
        return True
    return key.startswith(PLACEHOLDER_PREFIXES)


def ensure_user_config() -> str:
    """首次启动时从 example 或遗留配置拷贝一份到用户目录"""
    if os.path.exists(USER_CONFIG):
        return USER_CONFIG
    src = None
    for candidate in (LEGACY_CONFIG, EXAMPLE_CONFIG):
        if os.path.exists(candidate):
            src = candidate
            break
    data = DEFAULT_CONFIG
    if src:
        try:
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = DEFAULT_CONFIG
    save_user_config(data)
    return USER_CONFIG


def load_user_config() -> dict:
    try:
        with open(USER_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_user_config(cfg: dict):
    with open(USER_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(USER_CONFIG, 0o600)
    except Exception:
        pass


def _iter_configs():
    """按优先级产出配置来源：(来源名, 配置dict)"""
    yield "user", load_user_config()
    if os.path.exists(LEGACY_CONFIG):
        try:
            with open(LEGACY_CONFIG, "r", encoding="utf-8") as f:
                yield "project", json.load(f)
        except Exception:
            pass


def get_provider_config(provider: str = "") -> dict:
    """返回 {api_key, base_url, model, provider, source}"""
    provider = provider or os.getenv("AGENT_LLM_PROVIDER", "")
    out = {"provider": provider, "api_key": "", "base_url": "", "model": "", "source": ""}
    for source, cfg in _iter_configs():
        llm = cfg.get("llm", {})
        if not provider:
            provider = llm.get("provider", "deepseek")
            out["provider"] = provider
        providers = llm.get("providers", {})
        p = providers.get(provider) or providers.get("deepseek") or {}
        for field in ("api_key", "base_url", "model"):
            if not out[field] and p.get(field):
                out[field] = p[field]
                out["source"] = source
    return out


def resolve_api_key(provider: str = "") -> tuple:
    """返回 (key, source)。环境变量优先级最高。"""
    env_key = os.getenv("AGENT_API_KEY", "")
    if env_key:
        return env_key, "env"
    cfg = get_provider_config(provider)
    key = cfg.get("api_key", "")
    return key, cfg.get("source", "")


def require_api_key(provider: str = "") -> str:
    key, source = resolve_api_key(provider)
    if is_placeholder(key):
        raise MissingAPIKeyError(
            "尚未配置可用的 API Key。",
            hint=f"请填写 Key，或直接编辑配置文件：{USER_CONFIG}",
        )
    return key


def set_api_key(key: str, provider: str = "deepseek") -> dict:
    cfg = load_user_config()
    cfg.setdefault("llm", {}).setdefault("providers", {}).setdefault(provider, {})
    cfg["llm"]["providers"][provider]["api_key"] = key
    cfg["llm"]["provider"] = provider
    save_user_config(cfg)
    return {"success": True, "config_path": USER_CONFIG}


def config_status() -> dict:
    cfg = get_provider_config()
    key = cfg.get("api_key", "")
    masked = (key[:8] + "..." + key[-4:]) if len(key) > 12 else ("***" if key else "未设置")
    return {
        "provider": cfg.get("provider", ""),
        "model": cfg.get("model", ""),
        "base_url": cfg.get("base_url", ""),
        "api_key_masked": masked,
        "key_ready": not is_placeholder(key),
        "source": cfg.get("source", ""),
        "user_config_path": USER_CONFIG,
    }
