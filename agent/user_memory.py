"""
长期记忆：跨会话记住用户偏好
==============================
存 ~/.智影agent/user_memory.json：
    user_role            医生 / 家属 / 研究员 / 未知
    preferred_detail_level  brief / normal / detailed
    output_format          报告格式偏好（pdf / json）
    last_case_id           最近处理的病例
    notes                  其他稳定偏好

新会话开始时用 as_system_hint() 生成一条 system 消息插入。
"""

import os
import re
import json
import time

from agent.security import CONFIG_DIR

MEMORY_FILE = os.path.join(CONFIG_DIR, "user_memory.json")

ROLE_KEYWORDS = {
    "医生": "医生", "临床": "医生", "大夫": "医生", "神内": "医生",
    "家属": "家属", "患者家属": "家属", "陪护": "家属",
    "研究员": "研究员", "研究生": "研究员", "科研": "研究员", "学生": "研究员",
    "doctor": "医生", "clinician": "医生",
    "researcher": "研究员", "student": "研究员",
}

DETAIL_KEYWORDS = {
    "简单": "brief", "简短": "brief", "一句话": "brief", "简要": "brief",
    "详细": "detailed", "展开讲": "detailed", "具体": "detailed", "深入": "detailed",
}


class UserMemory:
    def __init__(self, path: str = ""):
        self.path = path or MEMORY_FILE
        self.data = self._load()

    # ---------------- IO ----------------

    def _load(self) -> dict:
        default = {
            "schema_version": 1,
            "user_role": "未知",
            "preferred_detail_level": "normal",
            "output_format": "pdf",
            "last_case_id": "",
            "notes": [],
            "updated_at": "",
        }
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    default.update(loaded)
        except Exception:
            pass
        return default

    def save(self):
        self.data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------------- 读写偏好 ----------------

    def set(self, key: str, value):
        self.data[key] = value
        self.save()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def remember(self, note: str):
        notes = self.data.get("notes", [])
        note = str(note).strip()
        if note and note not in notes:
            notes.append(note)
            self.data["notes"] = notes[-20:]
            self.save()

    def observe(self, text: str) -> list:
        """从用户语句里抽取稳定偏好，返回本次更新了哪些字段"""
        if not text:
            return []
        changed = []
        lowered = text.lower()

        for kw, role in ROLE_KEYWORDS.items():
            if kw in lowered:
                if self.data.get("user_role") != role:
                    self.data["user_role"] = role
                    changed.append(f"user_role={role}")
                break

        for kw, level in DETAIL_KEYWORDS.items():
            if kw in lowered:
                if self.data.get("preferred_detail_level") != level:
                    self.data["preferred_detail_level"] = level
                    changed.append(f"detail_level={level}")
                break

        if "json" in lowered and "报告" in text:
            if self.data.get("output_format") != "json":
                self.data["output_format"] = "json"
                changed.append("output_format=json")

        pref = re.search(r"(?:我叫|称呼我|我是)\s*([\u4e00-\u9fffA-Za-z]{1,12})", text)
        if pref:
            self.remember(f"用户称呼：{pref.group(1)}")
            changed.append("notes+=name")

        if changed:
            self.save()
        return changed

    def note_case(self, case_id: str):
        if case_id and self.data.get("last_case_id") != case_id:
            self.data["last_case_id"] = case_id
            self.save()

    # ---------------- 注入 prompt ----------------

    def as_system_hint(self) -> str:
        role = self.data.get("user_role", "未知")
        level = self.data.get("preferred_detail_level", "normal")
        fmt = self.data.get("output_format", "pdf")
        last_case = self.data.get("last_case_id", "")
        notes = self.data.get("notes", []) or []

        level_desc = {"brief": "简洁（要点式，少解释）",
                      "normal": "适中",
                      "detailed": "详尽（展开机制与依据）"}.get(level, "适中")
        lines = [
            "【用户画像 · 长期记忆】",
            f"- 身份：{role}；讲解深度：{level_desc}；报告格式偏好：{fmt}",
        ]
        if last_case:
            lines.append(f"- 最近处理的病例：{last_case}（用户再说【这个病例】时指它）")
        if notes:
            lines.append("- 其他偏好：" + "；".join(str(n) for n in notes[-5:]))
        if role == "家属":
            lines.append("- 对方是家属：避免过多术语，结论先行，多给可执行建议。")
        elif role == "医生":
            lines.append("- 对方是医生：可直接用术语、给数值与依据，注明来源。")
        elif role == "研究员":
            lines.append("- 对方是研究员：补充方法学细节（图谱、阈值、模型结构）与局限性。")
        return "\n".join(lines)


_memory: UserMemory = None


def get_memory() -> UserMemory:
    global _memory
    if _memory is None:
        _memory = UserMemory()
    return _memory


__all__ = ["UserMemory", "get_memory", "MEMORY_FILE"]
