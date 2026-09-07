"""
智影Agent 对话页面 —— 嵌入 PySide6 桌面软件
===============================================
DeepSeek API + SA-STGCN Tool Calling（18 个工具）+ 多会话管理。
Agent 循环逻辑在 agent/runtime.py，本文件只负责 UI 与线程信号。
"""

import json
import sys
import os
import re
import time
import unicodedata
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PySide6.QtCore import Qt, QThread, Signal, QEvent
from PySide6.QtGui import QFont, QColor, QTextCursor, QTextBlockFormat, QTextCharFormat
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QTextBrowser,
    QPushButton, QLabel, QFrame, QApplication,
    QSplitter, QSizePolicy,
    QListWidget, QListWidgetItem, QMenu, QInputDialog, QMessageBox,
    QFileDialog,
)

from agent.llm_client import get_llm_client, assert_ready, LLMUnavailableError
from agent.tools import TOOLS, TOOL_MAP, describe_groups
from agent.config import LLM_PROVIDER, LLM_MODEL, MAX_STEPS, TOOL_TIMEOUT, LANGUAGE
from agent.session_manager import SessionManager
from agent.runtime import AgentLoop, make_plan, stream_chunks, Plan
from agent.prompts import get_prompt
from agent.user_memory import get_memory
from agent.keyring import USER_CONFIG
from agent.util import extract_subject_id


def shorten_for_title(text: str, max_width: int = 24) -> str:
    """会话列表标题截断：按**显示宽度**算（中文/全角算 2），
    避免按字符数截断时中文标题在 UI 上占两倍宽度、英文标题又过短。"""
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


# ==================== 后台执行线程 ====================

class AgentWorker(QThread):
    """后台跑 Agent 循环（LLM ⇄ 工具），只负责信号转发"""

    stream_chunk = Signal(str)
    tool_start = Signal(str, str)
    tool_done = Signal(str, str)
    completed = Signal(str, list)     # (final_text, tool_steps)
    error = Signal(str, str)          # (message, hint)

    def __init__(self, history: list, parent=None, session_id: str = "",
                 lang: str = "", plan: Plan = None):
        super().__init__(parent)
        self.history = history or []
        self.session_id = session_id
        self.lang = lang or LANGUAGE
        self.plan = plan
        self.tool_steps: list = []

    def run(self):
        try:
            ok, msg = assert_ready()
            if not ok:
                self.error.emit(msg, f"配置文件：{USER_CONFIG}")
                return

            llm = get_llm_client()
            system_prompt = get_prompt(self.lang, tool_groups=describe_groups())

            extra = get_memory().as_system_hint()
            if self.plan and self.plan.steps:
                extra += "\n\n【本轮执行计划（已与用户确认）】\n" + self.plan.to_text()

            loop = AgentLoop(
                llm=llm, tools=TOOLS, tool_map=TOOL_MAP,
                system_prompt=system_prompt,
                max_steps=MAX_STEPS, tool_timeout=TOOL_TIMEOUT,
                session_id=self.session_id,
                callbacks={
                    "tool_start": self._on_tool_start,
                    "tool_done": self._on_tool_done,
                },
            )

            # 带 plan 时走 Plan-as-tool-chain：计划当骨架，模型仍可临时插步
            result = loop.run(self.history, extra_system=extra, plan=self.plan)
            if not result.get("success"):
                self.error.emit(result.get("error", "Agent 执行失败"), "")
                return

            text = result.get("text", "") or ""
            # 分段流式输出：按句子切，而不是固定 10 字符
            for piece in stream_chunks(text):
                if self.isInterruptionRequested():
                    break
                self.stream_chunk.emit(piece)
                self.msleep(18)

            self.tool_steps = result.get("tool_steps", [])
            self.completed.emit(text, self.tool_steps)

        except LLMUnavailableError as e:
            self.error.emit(str(e), getattr(e, "hint", ""))
        except Exception as e:
            self.error.emit(f"Agent 内部错误: {type(e).__name__}: {e}",
                            traceback.format_exc()[-800:])

    # ---------------- 工具回调 ----------------

    def _on_tool_start(self, name: str, args: dict):
        short = {k: str(v)[:48] for k, v in (args or {}).items()}
        self.tool_start.emit(name, json.dumps(short, ensure_ascii=False))

    def _on_tool_done(self, name: str, result: dict, summary: str):
        self.tool_done.emit(name, summary)

    # [来源: nia_aa_guidelines.md#MCI 进展风险分层] → 渲染成浅色引用标签
    _CITE_RE = re.compile(r"\[来源:\s*([^\]#]+)#([^\]]+)\]")

    @staticmethod
    def _to_html(text: str) -> str:
        if not text:
            return ""
        html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = re.sub(r'\*\*(.+?)\*\*', r'<b style="color:#5FE2FF;">\1</b>', html)
        html = re.sub(r'\*(.+?)\*', r'<i style="color:#A9BEDA;">\1</i>', html)
        html = AgentWorker._CITE_RE.sub(
            r'<span style="color:#7FD1FF;font-size:12px;background:rgba(42,150,255,0.12);'
            r'border-radius:4px;padding:1px 5px;">📎 来源: \1#\2</span>', html)
        return html.replace('\n', '<br>')


# ==================== 对话页面（含侧边栏） ====================

class AgentChatPage(QWidget):
    """嵌入主窗口的 AI 对话页面，左侧会话列表 + 右侧聊天区"""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._worker: AgentWorker | None = None
        self._session_mgr = SessionManager()
        self._ai_text_buffer = ""
        self._pending_plan: Plan = None
        self.setup_ui()
        self._ensure_active_session()
        self._refresh_session_list()

    # ==================== UI ====================

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- 标题栏 ---
        header = QFrame()
        header.setFixedHeight(38)
        header.setStyleSheet("background:rgba(6,20,42,0.85); border-bottom:1px solid rgba(96,190,255,0.15);")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(14, 0, 10, 0)
        h_layout.addWidget(QLabel("🤖 AI 智能诊断助手"))
        h_layout.addStretch()
        cfg_info = f"{LLM_PROVIDER}/{LLM_MODEL} | SA-STGCN"
        self.status_label = QLabel(cfg_info)
        self.status_label.setStyleSheet("color:#89A7C4;font-size:11px;")
        h_layout.addWidget(self.status_label)
        root.addWidget(header)

        # --- 主区域：QSplitter ---
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

        # 输入区
        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background:rgba(6,18,40,0.5); border-top:1px solid rgba(74,164,235,0.15); }")
        in_layout = QHBoxLayout(input_frame)
        in_layout.setContentsMargins(8, 6, 8, 6)
        in_layout.setSpacing(6)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("输入消息...按 Enter 发送（Shift+Enter 换行）")
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

        # 快捷按钮
        quick_row = QHBoxLayout()
        quick_row.setContentsMargins(8, 4, 8, 4)
        quick_row.setSpacing(6)
        self._add_quick_btn("📊 预测流程", "请介绍一下完整的诊断流程", quick_row)
        self._add_quick_btn("📖 数据查看", "帮我看看有哪些扫描数据", quick_row)
        self._add_quick_btn("🧭 先出计划", "__plan__", quick_row)
        self._add_quick_btn("📤 导出会话", "__export__", quick_row)
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

        self.memory_label = QLabel("")
        self.memory_label.setWordWrap(True)
        self.memory_label.setStyleSheet("color:#6D88A6;font-size:10px;padding:2px 4px;")
        self._refresh_memory_label()
        layout.addWidget(self.memory_label)

        # 数据目录授权：默认只允许项目目录，跨盘数据需用户显式授权
        self.auth_btn = QPushButton("📂 授权数据目录")
        self.auth_btn.setCursor(Qt.PointingHandCursor)
        self.auth_btn.setToolTip("AI 助手默认只能访问项目目录；把外部数据目录加入白名单后才可读写")
        self.auth_btn.setStyleSheet("""
            QPushButton { background:rgba(10,40,80,0.35); color:#A9C8E8; border:1px solid rgba(82,179,241,0.15);
                border-radius:6px; padding:5px; font-size:11px; }
            QPushButton:hover { background:rgba(20,70,140,0.45); color:#E0EDFF; }
        """)
        self.auth_btn.clicked.connect(self._authorize_directory)
        layout.addWidget(self.auth_btn)

        # 设置：API 配置 + 数据目录（可换盘 / 便携迁移）
        self.settings_btn = QPushButton("⚙ 设置")
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.setToolTip("API 配置、测试连接、数据目录换盘")
        self.settings_btn.setStyleSheet("""
            QPushButton { background:rgba(10,40,80,0.35); color:#A9C8E8; border:1px solid rgba(82,179,241,0.15);
                border-radius:6px; padding:5px; font-size:11px; }
            QPushButton:hover { background:rgba(20,70,140,0.45); color:#E0EDFF; }
        """)
        self.settings_btn.clicked.connect(self._open_settings)
        layout.addWidget(self.settings_btn)

        parent.addWidget(panel)

    def _add_quick_btn(self, label: str, msg: str, layout: QHBoxLayout):
        btn = QPushButton(label)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton { background:rgba(10,40,80,0.35); color:#fff; border:1px solid rgba(82,179,241,0.15);
                border-radius:6px; padding:4px 10px; font-size:11px; }
            QPushButton:hover { background:rgba(20,70,140,0.45); color:#E0EDFF; }
        """)
        if msg == "__plan__":
            btn.clicked.connect(self._plan_then_send)
        elif msg == "__export__":
            btn.clicked.connect(self._export_session)
        else:
            btn.clicked.connect(lambda: self._quick_send(msg))
        layout.addWidget(btn)

    def _quick_send(self, msg: str):
        self.input_edit.setPlainText(msg)

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
        export_act = menu.addAction("📤 导出会话（脱敏）")
        del_act = menu.addAction("🗑 删除")
        action = menu.exec(self.session_list.mapToGlobal(pos))
        if action == rename_act:
            name, ok = QInputDialog.getText(self, "重命名", "新名称:", text=item.text())
            if ok and name.strip():
                self._session_mgr.rename_session(sid, name.strip())
                self._refresh_session_list()
        elif action == export_act:
            self._export_session(sid)
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

    def _refresh_memory_label(self):
        try:
            mem = get_memory()
            role = mem.get("user_role", "未知")
            level = mem.get("preferred_detail_level", "normal")
            self.memory_label.setText(f"👤 {role} · 深度 {level}")
        except Exception:
            self.memory_label.setText("")

    def _open_settings(self):
        """打开设置对话框（API 配置 + 数据目录）"""
        from agent.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self, on_applied=self._refresh_status_label)
        dialog.exec()

    def _refresh_status_label(self):
        """配置变更后刷新右上角状态栏"""
        try:
            from agent.llm_client import llm_status
            st = llm_status()
            provider = st.get("provider", "") or LLM_PROVIDER
            model = st.get("model", "") or LLM_MODEL
            if st.get("available"):
                self.status_label.setText(f"{provider}/{model} | SA-STGCN")
            else:
                self.status_label.setText(f"{provider}/{model} | ⚠ Key 未配置")
        except Exception:
            pass

    def _authorize_directory(self):
        """把外部数据目录加入白名单（默认只允许 PROJECT_ROOT）"""
        from agent.security import authorize_directory, load_allowed_roots
        from agent.config import PROJECT_ROOT

        directory = QFileDialog.getExistingDirectory(self, "选择要授权给 AI 助手访问的目录", PROJECT_ROOT)
        if not directory:
            return
        result = authorize_directory(directory)
        if result.get("success"):
            QMessageBox.information(
                self, "已授权",
                f"AI 助手现在可以访问：\n{directory}\n\n"
                f"当前白名单共 {len(load_allowed_roots())} 个目录。")
        else:
            QMessageBox.warning(self, "授权失败", result.get("error", "未知错误"))

    def _export_session(self, session_id: str = ""):
        """导出会话。默认脱敏；要导出含患者信息的原始版本需二次确认。"""
        sid = session_id or self._session_mgr.active_id
        if not sid:
            return
        session = self._session_mgr.get_session(sid)
        if not session:
            return

        default_name = re.sub(r'[\\/:*?"<>|]', "_", session.name or "session") + ".md"
        path, _ = QFileDialog.getSaveFileName(self, "导出会话", default_name,
                                              "Markdown (*.md);;JSON (*.json)")
        if not path:
            return
        fmt = "json" if path.lower().endswith(".json") else "md"

        box = QMessageBox(self)
        box.setWindowTitle("导出方式")
        box.setText("导出的文件会离开本机，请选择导出方式：")
        box.setInformativeText(
            "脱敏：绝对路径 → <病例路径#xxxxxx>，患者 ID → <ID#xxxxxx>（推荐）\n"
            "原始：保留真实路径与患者编号，仅供本机留档，请勿外传")
        safe_btn = box.addButton("脱敏导出（推荐）", QMessageBox.AcceptRole)
        raw_btn = box.addButton("原始导出（含患者信息）", QMessageBox.DestructiveRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(safe_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is raw_btn:
            confirm = QMessageBox.warning(
                self, "确认导出原始内容",
                "导出的文件将包含患者编号与完整路径，存在隐私泄露风险。\n\n"
                "仅在确需本机留档时继续。确定要导出原始内容吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if confirm != QMessageBox.Yes:
                return
            redact = False
        elif clicked is safe_btn:
            redact = True
        else:
            return

        result = self._session_mgr.export_session(sid, path, fmt=fmt, redact=redact)
        if result.get("success"):
            note = ("已脱敏：绝对路径 → <病例路径#xxxxxx>，患者 ID → <ID#xxxxxx>"
                    if redact else "⚠ 原始导出，包含患者信息，请勿外传")
            QMessageBox.information(self, "导出成功", f"已导出：\n{path}\n\n{note}")
        else:
            QMessageBox.warning(self, "导出失败", result.get("error", "未知错误"))

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
            elif role == "assistant":
                if content:
                    self._append_message("assistant", AgentWorker._to_html(content))
                for tc in (msg.get("tool_calls") or []):
                    self._append_tool_status("🔧", f"{tc.get('name', '')} 已调用")
            elif role == "tool":
                self._append_tool_status("📋", content[:70])

        if not session.messages:
            self._append_welcome()

    # ==================== 消息显示 ====================

    def _append_welcome(self):
        self._append_message("assistant", """
<p style="font-size:15px;margin:0;">👋 您好！我是<b style="color:#5FE2FF;">智影Agent</b>。</p>
<p style="margin:8px 0 4px;">我可以帮你跑完整条流程，不只是回答问题：</p>
<ul style="margin:4px 0; color:#B7D1ED;">
<li>导入影像 → 预处理 → <b>提取特征</b> → <b>SA-STGCN 预测</b> → <b>生成 PDF 报告</b></li>
<li>解释 <b>关键脑区</b>及其与 AD 的关联（带知识库来源）</li>
<li>查询 <b>诊断标准 / 随访建议 / 用药进展</b></li>
</ul>
<p style="margin-top:8px; color:#A9BEDA; font-size:13px;">
💡 直接说「帮我处理 Data 下某个样本，给出预测和报告」即可全自动完成。<br>
📂 也可以问"流程是什么""有哪些数据"来了解系统。
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

    def _append_tool_status(self, icon: str, text: str):
        cursor = self.chat_browser.textCursor()
        cursor.movePosition(QTextCursor.End)
        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(Qt.AlignLeft)
        block_fmt.setLeftMargin(20)
        block_fmt.setRightMargin(60)
        block_fmt.setTopMargin(6)
        block_fmt.setBottomMargin(4)
        char_fmt = QTextCharFormat()
        char_fmt.setForeground(QColor(251, 191, 36))
        cursor.insertBlock(block_fmt, char_fmt)
        cursor.insertHtml(f"{icon} {text}")
        self.chat_browser.setTextCursor(cursor)
        self._scroll_bottom()

    def _scroll_bottom(self):
        self.chat_browser.verticalScrollBar().setValue(
            self.chat_browser.verticalScrollBar().maximum())

    def _show_typing(self):
        self._append_message("assistant", '<span style="color:#89A7C4;">⏳ 正在分析中...</span>')

    def _update_last_ai(self, html: str):
        cursor = self.chat_browser.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        self._append_message("assistant", html)

    # ==================== 发送 / 计划 ====================

    def _history_for_llm(self) -> list:
        session = self._session_mgr.active_session
        if session is None:
            return []
        history = []
        for msg in session.messages:
            role = msg.get("role", "")
            if role in ("user", "assistant", "tool"):
                entry = {"role": role, "content": msg.get("content", "")}
                if msg.get("tool_call_id"):
                    entry["tool_call_id"] = msg["tool_call_id"]
                history.append(entry)
        return history

    def _send_message(self, plan: Plan = None):
        # QPushButton.clicked 会带一个 bool 进来，这里只认真正的 Plan
        if not isinstance(plan, Plan):
            plan = None
        text = self.input_edit.toPlainText().strip()
        if not text or self._worker is not None:
            return

        session = self._session_mgr.active_session
        if session is None:
            return

        ok, msg = assert_ready()
        if not ok:
            QMessageBox.warning(self, "需要配置 API Key",
                                f"{msg}\n\n配置后无需重启，直接重发即可。")
            return

        # 长期记忆：从用户输入里抽取稳定偏好
        try:
            get_memory().observe(text)
            self._refresh_memory_label()
        except Exception:
            pass

        self.input_edit.clear()
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._append_message("user", escaped)
        self._show_typing()
        session.add_message("user", text)

        self._ai_text_buffer = ""
        self._worker = AgentWorker(
            self._history_for_llm(),        # 含刚写入的这条 user 消息
            session_id=session.id,
            plan=plan,
        )
        self._worker.stream_chunk.connect(self._on_stream)
        self._worker.tool_start.connect(self._on_tool_start)
        self._worker.tool_done.connect(self._on_tool_done)
        self._worker.completed.connect(self._on_completed)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _plan_then_send(self):
        """先让 LLM 出计划，用户确认后再执行"""
        text = self.input_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "先在输入框里写下你的诉求，再点「先出计划」。")
            return
        ok, msg = assert_ready()
        if not ok:
            QMessageBox.warning(self, "需要配置 API Key", msg)
            return
        try:
            llm = get_llm_client()
            plan = make_plan(llm, text, describe_groups())
        except Exception as e:
            QMessageBox.warning(self, "计划生成失败", f"{type(e).__name__}: {e}")
            return

        box = QMessageBox(self)
        box.setWindowTitle("执行计划确认")
        box.setText("我打算这样执行：\n\n" + plan.to_text())
        box.setInformativeText("确认后我会按计划调用工具；你也可以取消后自己补充要求。")
        run_btn = box.addButton("按计划执行", QMessageBox.AcceptRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is not run_btn:
            return
        self._send_message(plan=plan)

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
        self._update_last_ai(AgentWorker._to_html(self._ai_text_buffer))

    def _on_tool_start(self, name: str, args_summary: str):
        self._append_tool_status("🔧", f"调用工具: <b>{name}</b>")

    def _on_tool_done(self, name: str, result_summary: str):
        summary = (result_summary or "").replace("<", "&lt;").replace(">", "&gt;")
        self._append_tool_status("📋", f"<b>{name}</b>: {summary}")

    def _on_completed(self, text: str, tool_steps: list):
        final = AgentWorker._to_html(self._ai_text_buffer or text)
        self._update_last_ai(final)

        session = self._session_mgr.active_session
        if session:
            content = self._ai_text_buffer or text
            session.add_message("assistant", content,
                                tool_calls=[{"name": ts.get("name"), "args": {}} for ts in tool_steps])
            for ts in tool_steps or []:
                session.add_message("tool", f"{ts.get('name')}: {ts.get('result_summary', '')}")
            # 记住最近处理过的病例（存病例 ID，不存路径——路径属于敏感信息）
            for ts in tool_steps or []:
                if ts.get("name") == "predict_ad_risk":
                    try:
                        get_memory().note_case(
                            extract_subject_id(str((ts.get("args") or {}).get("fc_path", ""))))
                    except Exception:
                        pass
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
        detail = AgentWorker._to_html(err)
        if hint:
            detail += f'<div style="color:#89A7C4;font-size:12px;">💡 {AgentWorker._to_html(hint)}</div>'
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

    # ==================== 外部对接 ====================

    def load_case_data(self, case_data: dict):
        if not case_data:
            return
        fc_path = case_data.get("fc_path", "")
        bold_path = case_data.get("bold_path", "")
        probs = case_data.get("probs")
        pred = case_data.get("prediction")

        hint = "📋 当前已加载诊断结果：<br>"
        if fc_path:
            hint += f"FC: {fc_path}<br>"
        if bold_path:
            hint += f"BOLD: {bold_path}<br>"
        if pred is not None and probs is not None:
            label = "认知稳定" if pred == 0 else "认知进展"
            hint += f"预测: {label} | 概率: {[f'{p * 100:.1f}%' for p in probs]}<br>"

        self._append_message("assistant",
                             f'<div style="font-size:13px;color:#FBBF24;">{hint}</div>'
                             f'<div style="font-size:13px;color:#A9BEDA;margin-top:4px;">'
                             f'可以就此提问："为什么预测这个结果？""关键脑区有哪些？"'
                             f'也可以直接让我生成 PDF 报告。</div>')
