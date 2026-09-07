"""
会话管理器 —— 多对话支持
============================
- schema_version：便于后续迁移（v1 为旧格式，无版本号）
- 落盘消息只保留 role / content / tool_call_id / timestamp，不存 numpy 引用
- 落盘前统一过 redact()：绝对路径 → <病例路径>，患者 ID → <ID>
- 单条消息长度上限，保证 100 条消息的 JSON 远小于 200KB
"""

import os
import json
import uuid
import time
from typing import Optional

from agent.redaction import redact, redact_text
from agent.util import CONFIG_DIR

# 会话含病历信息，存在用户数据目录（可用「⚙ 设置」换盘），不留在项目目录里
_STORAGE = os.path.join(CONFIG_DIR, "sessions.json")
# 旧版本存在 agent/ 下，启动时自动搬一次
_LEGACY_STORAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sessions.json")

SCHEMA_VERSION = 2
MAX_STORED_CHARS = 4000     # 单条消息落盘上限
MAX_MESSAGES = 200          # 单会话最多保留的消息数

_KEEP_KEYS = ("role", "content", "tool_call_id", "timestamp")


def _clean_message(msg: dict) -> dict:
    """只保留能安全序列化且对回放有意义的字段"""
    out = {}
    for key in _KEEP_KEYS:
        if key not in msg:
            continue
        value = msg[key]
        if key == "content":
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, default=str)
            if len(value) > MAX_STORED_CHARS:
                value = value[:MAX_STORED_CHARS] + f"...(截断，共{len(value)}字)"
        out[key] = value
    if not out.get("content") and out.get("role") != "assistant":
        out["content"] = ""
    return out


class Session:
    """单个对话会话"""

    def __init__(self, session_id: str, name: str = ""):
        self.id = session_id
        self.name = name or "New Chat"
        self.created_at = time.strftime("%Y-%m-%d %H:%M")
        self.schema_version = SCHEMA_VERSION
        self.messages: list[dict] = []
        self.system_prompt = ""

    def add_message(self, role: str, content: str, tool_calls: Optional[list] = None,
                    tool_call_id: str = ""):
        entry = {"role": role, "content": content or ""}
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        if tool_calls:
            # 只存工具名与参数摘要，避免把完整返回值塞进历史
            entry["tool_calls"] = [
                {"name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in tool_calls
            ]
        self.messages.append(entry)
        if len(self.messages) > MAX_MESSAGES:
            self.messages = self.messages[-MAX_MESSAGES:]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "system_prompt": self.system_prompt,
            "messages": [_clean_message(m) for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        s = cls(d["id"], d.get("name", ""))
        s.created_at = d.get("created_at", "")
        s.schema_version = int(d.get("schema_version", 1))
        s.messages = [_clean_message(m) for m in d.get("messages", [])]
        s.system_prompt = d.get("system_prompt", "")
        return s

    # ---------------- 导出 ----------------

    def export_text(self, redact: bool = True) -> str:
        """导出 Markdown。默认**强制脱敏**——导出的文件会离开本机，不能带患者信息。"""
        header = [f"# {self.name}  (会话ID: {self.id})", f"创建时间: {self.created_at}",
                  "> 本文件已脱敏：绝对路径 → <病例路径#xxxxxx>，患者 ID → <ID#xxxxxx>"]
        lines = list(header) if redact else header[:2]
        lines.append("")
        for m in self.messages:
            role = {"user": "用户", "assistant": "助手", "tool": "工具返回"}.get(
                m.get("role", ""), m.get("role", ""))
            content = m.get("content", "")
            if redact:
                content = redact_text(content)
            lines.append(f"**{role}**：{content}")
            lines.append("")
        return "\n".join(lines)

    def export_json(self, redact: bool = True) -> str:
        data = self.to_dict()
        return json.dumps(redact(data) if redact else data, ensure_ascii=False, indent=2)


class SessionManager:
    """管理所有会话"""

    def __init__(self, max_sessions: int = 50, storage: str = ""):
        self.max_sessions = max_sessions
        self._storage = storage or _STORAGE
        self.schema_version = SCHEMA_VERSION
        self._sessions: dict[str, Session] = {}
        self._active_id: Optional[str] = None
        self._migrate_legacy()
        self._load()

    def _migrate_legacy(self):
        """旧版本把会话存在 agent/sessions.json，搬一次到用户数据目录"""
        import shutil

        if self._storage != _STORAGE or not os.path.exists(_LEGACY_STORAGE):
            return
        if os.path.exists(_STORAGE):
            return
        try:
            shutil.move(_LEGACY_STORAGE, _STORAGE)
        except Exception:
            pass

    # ==================== CRUD ====================

    def create_session(self, name: str = "") -> Session:
        sid = uuid.uuid4().hex[:12]
        session = Session(sid, name)
        self._sessions[sid] = session
        self._active_id = sid
        self._save()
        if len(self._sessions) > self.max_sessions:
            oldest = sorted(self._sessions.values(), key=lambda s: s.created_at)[0]
            del self._sessions[oldest.id]
        return session

    def delete_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
            if self._active_id == session_id:
                self._active_id = None
            self._save()

    def rename_session(self, session_id: str, new_name: str):
        if session_id in self._sessions:
            self._sessions[session_id].name = new_name
            self._save()

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    @property
    def active_session(self) -> Optional[Session]:
        if self._active_id and self._active_id in self._sessions:
            return self._sessions[self._active_id]
        return None

    @active_session.setter
    def active_session(self, session: Session):
        self._active_id = session.id

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    def switch_to(self, session_id: str) -> Optional[Session]:
        if session_id in self._sessions:
            self._active_id = session_id
            self._save()
            return self._sessions[session_id]
        return None

    def list_sessions(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    # ==================== 持久化（脱敏） ====================

    def _save(self):
        try:
            data = [s.to_dict() for s in self._sessions.values()]
            with open(self._storage, "w", encoding="utf-8") as f:
                json.dump(redact(data), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self):
        if not os.path.exists(self._storage):
            return
        try:
            with open(self._storage, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):        # v1 兼容：{id: {...}}
                data = list(data.values())
            for item in data:
                s = Session.from_dict(item)
                self._sessions[s.id] = s
            if self._sessions:
                last = sorted(self._sessions.values(), key=lambda s: s.created_at)[-1]
                self._active_id = last.id
        except Exception:
            pass

    def export_session(self, session_id: str, path: str, fmt: str = "md",
                       redact: bool = True) -> dict:
        """导出完整会话（UI 里的「导出会话」按钮用）。

        redact=True（默认）会脱敏绝对路径与患者 ID；
        redact=False 仅用于本机留档，UI 必须二次确认后才允许。
        """
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": f"会话不存在: {session_id}"}
        try:
            if fmt.lower() == "json":
                content = session.export_json(redact=redact)
            else:
                content = session.export_text(redact=redact)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "path": path, "redacted": redact}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    def storage_size_kb(self) -> float:
        try:
            return round(os.path.getsize(self._storage) / 1024, 1)
        except Exception:
            return 0.0
