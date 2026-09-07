"""
患者端 AI 健康问答页
====================
与医生端 AgentChatPage 的区别：
- **没有任何工具调用**：不接 TOOLS/AgentLoop，不能跑诊断流程、不能碰数据。
- 只做「RAG 检索 + LLM 归纳」：先检索 agent/knowledge 知识库，把命中片段
  连同患者版 system prompt 一起发给 LLM 做纯对话补全，并强制标注来源。
- 会话独立存储（patient_sessions.json），与医生端会话互相不可见。
"""

import re
import sys
import os
import traceback
import unicodedata

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PySide6.QtCore import Qt, QThread, Signal, QEvent
from PySide6.QtGui import QColor, QTextCursor, QTextBlockFormat, QTextCharFormat
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QTextBrowser,
    QPushButton, QLabel, QFrame, QSplitter, QListWidget, QListWidgetItem,
    QMenu, QMessageBox,
)

from agent.llm_client import get_llm_client, assert_ready, LLMUnavailableError
from agent.rag import get_rag
from agent.config import LLM_PROVIDER, LLM_MODEL
from agent.session_manager import SessionManager
from agent.keyring import CONFIG_DIR

# 患者端会话与医生端分开落盘，避免患者看到医生的诊断会话
_PATIENT_STORAGE = os.path.join(CONFIG_DIR, "patient_sessions.json")

PATIENT_SYSTEM_PROMPT = """你是「智影健康助手」——面向 MCI/AD 患者及家属的健康科普问答助手。

## 身份与边界
- 你**只能**依据下方提供的知识库片段回答问题；知识库没有的内容，如实回答
  「这个问题我暂时无法回答，建议咨询您的主治医生」，严禁凭记忆编造医学事实。
- 不给诊断结论、不推荐具体药物与剂量、不做治疗决策；涉及就医、用药时引导用户咨询主治医生。
- 影像诊断、预测、报告生成等系统功能在医生端，由医生操作；用户问起时说明这一点即可。

## 回答要求
- 语言温和通俗，少用专业术语，必要的术语用一句话解释。
- 回答简明分点，一般不超过 200 字。
- 凡来自知识库的结论，必须紧跟引用标签，格式为 `[来源: 文件名#章节]`，
  原样照抄检索结果的 citation 字段，不得改写或省略。"""


def _patient_system_prompt(rag_result: dict) -> str:
    """把检索片段拼进 system prompt；未命中时明确告知未覆盖。"""
    if not rag_result.get("results"):
        note = rag_result.get("message") or rag_result.get("error") or "知识库未覆盖此问题。"
        return (PATIENT_SYSTEM_PROMPT
                + "\n\n【知识库检索结果】\n本次检索未命中任何相关内容。"
                + "请直接按边界要求告知用户无法回答。检索说明：" + note)

    parts = ["\n\n【知识库检索结果】（回答只能基于这些内容）"]
    for item in rag_result["results"]:
        parts.append(f"\n--- {item['citation']}（相关度 {item['score']}）\n{item['content']}")
    return PATIENT_SYSTEM_PROMPT + "\n".join(parts)


def shorten_for_title(text: str, max_width: int = 24) -> str:
    if not text:
        return "New Chat"
    text = " ".join(text.split())
    width = 0
    cut = 0
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if width + w > max_width:
            break
        width += w
        cut += 1
    return text[:cut] + ("..." if cut < len(text) else "")


# [来源: nia_aa_guidelines.md#MCI 进展风险分层] → 浅色引用标签
_CITE_RE = re.compile(r"\[来源:\s*([^\]#]+)#([^\]]+)\]")


def _to_html(text: str) -> str:
    if not text:
        return ""
    html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = re.sub(r'\*\*(.+?)\*\*', r'<b style="color:#5FE2FF;">\1</b>', html)
    html = _CITE_RE.sub(
        r'<span style="color:#7FD1FF;font-size:12px;background:rgba(42,150,255,0.12);'
        r'border-radius:4px;padding:1px 5px;">📎 来源: \1#\2</span>', html)
    return html.replace('\n', '<br>')


# ==================== 后台线程：检索 + 纯对话 ====================

class PatientChatWorker(QThread):
    """检索知识库 → 不带 tools 的流式对话补全。只做信号转发。"""

    stream_chunk = Signal(str)
    completed = Signal(str)
    error = Signal(str, str)          # (message, hint)

    def __init__(self, history: list, parent=None):
        super().__init__(parent)
        self.history = history or []

    def run(self):
        try:
            ok, msg = assert_ready()
            if not ok:
                self.error.emit(msg, "")
                return

            query = next((m.get("content", "") for m in reversed(self.history)
                          if m.get("role") == "user"), "")
            rag_result = get_rag().search(query, top_k=4)

            llm = get_llm_client()
            messages = [{"role": "system", "content": _patient_system_prompt(rag_result)}]
            messages += [{"role": m["role"], "content": m["content"]}
                         for m in self.history if m.get("role") in ("user", "assistant")]

            full = ""
            for ev in llm.chat(messages=messages, stream=True):
                if self.isInterruptionRequested():
                    break
                if not getattr(ev, "choices", None):
                    continue
                delta = getattr(ev.choices[0].delta, "content", None)
                if delta:
                    full += delta
                    self.stream_chunk.emit(delta)

            self.completed.emit(full)
        except LLMUnavailableError as e:
            self.error.emit(str(e), getattr(e, "hint", ""))
        except Exception as e:
            self.error.emit(f"问答服务内部错误: {type(e).__name__}: {e}",
                            traceback.format_exc()[-800:])


# ==================== 患者端对话页 ====================

class PatientChatPage(QWidget):
    """患者端问答页：会话列表 + 聊天区，无任何工具入口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: PatientChatWorker | None = None
        self._session_mgr = SessionManager(storage=_PATIENT_STORAGE)
        self._ai_text_buffer = ""
        self.setup_ui()
        self._ensure_active_session()
        self._refresh_session_list()

    # ==================== UI ====================

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(38)
        header.setStyleSheet("background:rgba(6,20,42,0.85); border-bottom:1px solid rgba(96,190,255,0.15);")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(14, 0, 10, 0)
        h_layout.addWidget(QLabel("🌿 AI 健康问答助手"))
        h_layout.addStretch()
        self.status_label = QLabel(f"{LLM_PROVIDER}/{LLM_MODEL} | 知识库问答")
        self.status_label.setStyleSheet("color:#89A7C4;font-size:11px;")
        h_layout.addWidget(self.status_label)
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background:rgba(96,190,255,0.12); width:1px; }")

        self._build_session_sidebar(splitter)

        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        self.chat_browser = QTextBrowser()
        self.chat_browser.setOpenExternalLinks(True)
        self.chat_browser.setStyleSheet("""
            QTextBrowser {
                background: rgba(4,14,28,0.55); border: none;
                padding: 12px; color: #D6E8FF; font-size: 14px; line-height: 1.7;
            }
            QScrollBar:vertical { background:rgba(6,18,40,0.5); width:7px; border-radius:3px; }
            QScrollBar::handle:vertical { background:rgba(91,183,255,0.3); border-radius:3px; min-height:30px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        """)
        chat_layout.addWidget(self.chat_browser, 1)

        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background:rgba(6,18,40,0.5); border-top:1px solid rgba(74,164,235,0.15); }")
        in_layout = QHBoxLayout(input_frame)
        in_layout.setContentsMargins(8, 6, 8, 6)
        in_layout.setSpacing(6)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("想了解什么？例如「什么是轻度认知障碍」...")
        self.input_edit.setMaximumHeight(60)
        self.input_edit.setStyleSheet("""
            QTextEdit { background:rgba(4,14,30,0.68); border:1px solid rgba(82,179,241,0.18);
                border-radius:8px; padding:6px 10px; color:#E0EDFF; font-size:13px; }
            QTextEdit:focus { border:1px solid rgba(105,215,255,0.45); }
        """)
        self.input_edit.installEventFilter(self)

        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedSize(64, 36)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setStyleSheet("""
            QPushButton { background:rgba(8,80,165,0.78); color:#fff; border:1px solid rgba(91,204,255,0.42);
                border-radius:8px; font-weight:700; }
            QPushButton:hover { background:rgba(11,101,201,0.9); }
            QPushButton:disabled { background:rgba(30,50,80,0.5); color:#5A7390; }
        """)
        self.send_btn.clicked.connect(self._send_message)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setFixedSize(56, 36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet("""
            QPushButton { background:rgba(120,40,40,0.7); color:#FFD9D9; border:1px solid rgba(255,120,120,0.35);
                border-radius:8px; font-weight:600; }
            QPushButton:hover { background:rgba(160,50,50,0.85); }
            QPushButton:disabled { background:rgba(40,40,50,0.4); color:#6A6A78; }
        """)
        self.stop_btn.clicked.connect(self._stop_worker)

        in_layout.addWidget(self.input_edit, 1)
        in_layout.addWidget(self.stop_btn)
        in_layout.addWidget(self.send_btn)
        chat_layout.addWidget(input_frame)

        quick_row = QHBoxLayout()
        quick_row.setContentsMargins(8, 4, 8, 4)
        quick_row.setSpacing(6)
        self._add_quick_btn("🧠 什么是 MCI", "什么是轻度认知障碍（MCI）？", quick_row)
        self._add_quick_btn("🔍 如何早期发现", "出现哪些表现要警惕认知障碍？", quick_row)
        self._add_quick_btn("🥗 日常护脑", "日常生活中怎么做对大脑健康有帮助？", quick_row)
        self._add_quick_btn("📅 随访复查", "多长时间需要复查一次？", quick_row)
        quick_row.addStretch()
        chat_layout.addLayout(quick_row)

        splitter.addWidget(chat_panel)
        splitter.setSizes([180, 520])
        root.addWidget(splitter, 1)

    def _build_session_sidebar(self, parent: QSplitter):
        panel = QFrame()
        panel.setMinimumWidth(150)
        panel.setStyleSheet("QFrame { background:rgba(4,10,24,0.7); border-right:1px solid rgba(96,190,255,0.12); }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(6)

        new_btn = QPushButton("+ 新建对话")
        new_btn.setCursor(Qt.PointingHandCursor)
        new_btn.setStyleSheet("""
            QPushButton { background:rgba(8,80,165,0.5); color:#E0EDFF; border:1px solid rgba(91,204,255,0.2);
                border-radius:6px; padding:6px; font-size:12px; font-weight:600; }
            QPushButton:hover { background:rgba(11,101,201,0.75); }
        """)
        new_btn.clicked.connect(self._new_session)
        layout.addWidget(new_btn)

        self.session_list = QListWidget()
        self.session_list.setStyleSheet("""
            QListWidget { background:transparent; border:none; color:#C8DDF0; font-size:12px; }
            QListWidget::item { padding:7px 10px; border-radius:5px; margin:1px 0; }
            QListWidget::item:selected { background:rgba(42,150,255,0.2); color:#fff; }
            QListWidget::item:hover { background:rgba(42,150,255,0.1); }
        """)
        self.session_list.itemClicked.connect(self._on_session_clicked)
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._on_session_context_menu)
        layout.addWidget(self.session_list, 1)

        tip = QLabel("💡 我只能回答知识库内的\n健康科普问题，不能替代\n医生诊断哦")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#6D88A6;font-size:10px;padding:2px 4px;")
        layout.addWidget(tip)

        parent.addWidget(panel)

    def _add_quick_btn(self, label: str, msg: str, layout: QHBoxLayout):
        btn = QPushButton(label)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton { background:rgba(10,40,80,0.35); color:#fff; border:1px solid rgba(82,179,241,0.15);
                border-radius:6px; padding:4px 10px; font-size:11px; }
            QPushButton:hover { background:rgba(20,70,140,0.45); color:#E0EDFF; }
        """)
        btn.clicked.connect(lambda: self.input_edit.setPlainText(msg))
        layout.addWidget(btn)

    # ==================== 会话管理 ====================

    def _ensure_active_session(self):
        if self._session_mgr.active_session is None:
            self._session_mgr.create_session()
        self._render_session_messages()

    def new_session(self):
        """对外入口（悬浮球标题栏「＋ 新对话」用）"""
        self._new_session()

    def stop_worker(self):
        """对外入口（弹窗关闭时调用）"""
        self._stop_worker()

    def _new_session(self):
        self._session_mgr.create_session()
        self._refresh_session_list()
        self.chat_browser.clear()
        self._append_welcome()

    def _on_session_clicked(self, item: QListWidgetItem):
        sid = item.data(Qt.UserRole)
        if sid and sid != self._session_mgr.active_id:
            self._session_mgr.switch_to(sid)
            self._render_session_messages()

    def _on_session_context_menu(self, pos):
        item = self.session_list.itemAt(pos)
        if not item:
            return
        sid = item.data(Qt.UserRole)
        menu = QMenu()
        menu.setStyleSheet("QMenu { background:#0a1628; color:#D6E8FF; border:1px solid rgba(96,190,255,0.2); }"
                           "QMenu::item { padding:5px 20px; }"
                           "QMenu::item:selected { background:rgba(42,150,255,0.25); }")
        rename_act = menu.addAction("✏ 重命名")
        del_act = menu.addAction("🗑 删除")
        action = menu.exec(self.session_list.mapToGlobal(pos))
        if action == rename_act:
            from PySide6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(self, "重命名", "新名称:", text=item.text())
            if ok and name.strip():
                self._session_mgr.rename_session(sid, name.strip())
                self._refresh_session_list()
        elif action == del_act:
            if len(self._session_mgr.list_sessions()) <= 1:
                QMessageBox.information(self, "提示", "至少保留一个会话。")
                return
            self._session_mgr.delete_session(sid)
            if self._session_mgr.active_session is None:
                self._session_mgr.create_session()
                self._render_session_messages()
            self._refresh_session_list()

    def _refresh_session_list(self):
        self.session_list.blockSignals(True)
        self.session_list.clear()
        active_id = self._session_mgr.active_id
        for s in self._session_mgr.list_sessions():
            item = QListWidgetItem(s.name)
            item.setData(Qt.UserRole, s.id)
            if s.id == active_id:
                item.setSelected(True)
            self.session_list.addItem(item)
        self.session_list.blockSignals(False)

    def _render_session_messages(self):
        self.chat_browser.clear()
        session = self._session_mgr.active_session
        if session is None:
            self._append_welcome()
            return
        for msg in session.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                self._append_message("user", escaped)
            elif role == "assistant" and content:
                self._append_message("assistant", _to_html(content))
        if not session.messages:
            self._append_welcome()

    # ==================== 消息显示 ====================

    def _append_welcome(self):
        self._append_message("assistant", """
<p style="font-size:15px;margin:0;">👋 您好！我是<b style="color:#5FE2FF;">智影健康助手</b>。</p>
<p style="margin:8px 0 4px;">我可以回答认知健康相关的科普问题：</p>
<ul style="margin:4px 0; color:#B7D1ED;">
<li>什么是轻度认知障碍（MCI）、阿尔茨海默病</li>
<li>日常护脑、饮食运动、随访复查建议</li>
<li>认知训练的意义与做法</li>
</ul>
<p style="margin-top:8px; color:#A9BEDA; font-size:13px;">
💡 我只基于知识库回答，不能替代医生诊断；治疗和用药请咨询您的主治医生。<br>
📊 影像检查与风险预测由您的医生在医生端完成。
</p>
""")

    def _append_message(self, role: str, html: str):
        cursor = self.chat_browser.textCursor()
        cursor.movePosition(QTextCursor.End)
        block_fmt = QTextBlockFormat()
        block_fmt.setTopMargin(6)
        block_fmt.setBottomMargin(6)
        if role == "user":
            block_fmt.setAlignment(Qt.AlignRight)
            block_fmt.setLeftMargin(60)
            block_fmt.setRightMargin(0)
            char_fmt = QTextCharFormat()
            char_fmt.setBackground(QColor(8, 80, 165, 153))
            char_fmt.setForeground(QColor(224, 237, 255))
        else:
            block_fmt.setAlignment(Qt.AlignLeft)
            block_fmt.setLeftMargin(0)
            block_fmt.setRightMargin(60)
            char_fmt = QTextCharFormat()
            char_fmt.setBackground(QColor(6, 20, 38, 178))
            char_fmt.setForeground(QColor(214, 232, 255))
        cursor.insertBlock(block_fmt, char_fmt)
        cursor.insertHtml(html)
        self.chat_browser.setTextCursor(cursor)
        self._scroll_bottom()

    def _scroll_bottom(self):
        self.chat_browser.verticalScrollBar().setValue(
            self.chat_browser.verticalScrollBar().maximum())

    def _show_typing(self):
        self._append_message("assistant", '<span style="color:#89A7C4;">⏳ 正在查阅知识库...</span>')

    def _update_last_ai(self, html: str):
        cursor = self.chat_browser.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        self._append_message("assistant", html)

    # ==================== 发送 ====================

    def _history_for_llm(self) -> list:
        session = self._session_mgr.active_session
        if session is None:
            return []
        return [{"role": m["role"], "content": m.get("content", "")}
                for m in session.messages if m.get("role") in ("user", "assistant")]

    def _send_message(self):
        text = self.input_edit.toPlainText().strip()
        if not text or self._worker is not None:
            return

        session = self._session_mgr.active_session
        if session is None:
            return

        ok, msg = assert_ready()
        if not ok:
            QMessageBox.warning(self, "暂时无法回答", msg)
            return

        self.input_edit.clear()
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._append_message("user", escaped)
        self._show_typing()
        session.add_message("user", text)

        self._ai_text_buffer = ""
        self._worker = PatientChatWorker(self._history_for_llm())
        self._worker.stream_chunk.connect(self._on_stream)
        self._worker.completed.connect(self._on_completed)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop_worker(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.quit()
            self._worker.wait(2000)
        self._worker = None
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ==================== 回调 ====================

    def _on_stream(self, chunk: str):
        self._ai_text_buffer += chunk
        self._update_last_ai(_to_html(self._ai_text_buffer))

    def _on_completed(self, text: str):
        final = _to_html(self._ai_text_buffer or text)
        self._update_last_ai(final)

        session = self._session_mgr.active_session
        if session:
            content = self._ai_text_buffer or text
            session.add_message("assistant", content)
            self._session_mgr._save()

            if content and session.name == "New Chat":
                for msg in session.messages:
                    if msg.get("role") == "user":
                        q = msg.get("content", "").strip()
                        self._session_mgr.rename_session(session.id, shorten_for_title(q))
                        self._refresh_session_list()
                        break

        self._worker = None
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_edit.setFocus()

    def _on_error(self, err: str, hint: str):
        detail = _to_html(err)
        if hint:
            detail += f'<div style="color:#89A7C4;font-size:12px;">💡 {_to_html(hint)}</div>'
        self._update_last_ai(f'<span style="color:#EF4444;">❌ </span>{detail}')
        session = self._session_mgr.active_session
        if session:
            session.add_message("assistant", f"[错误] {err}")
            self._session_mgr._save()
        self._worker = None
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_edit.setFocus()

    # ==================== 键盘事件 ====================

    def eventFilter(self, obj, event):
        if obj is self.input_edit and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() & Qt.ShiftModifier:
                return False
            if event.key() == Qt.Key_Return:
                self._send_message()
                return True
        return super().eventFilter(obj, event)


# ==================== 快速自检 ====================
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("患者端 AI 问答自检")
    win.resize(680, 560)
    win.setStyleSheet("background:#040E1E;")
    win.setCentralWidget(PatientChatPage())
    win.show()
    sys.exit(app.exec())
