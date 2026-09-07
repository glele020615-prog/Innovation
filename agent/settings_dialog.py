"""
设置对话框：API 配置 + 数据目录（可换盘 / 便携迁移）
======================================================
数据目录默认在 ~/.智影agent，可用本对话框改到任意盘：

  - 便携模式（推荐）：把位置写进项目根/exe 目录的 agent_home.txt，
    整个程序目录拷到别的机器，配置/日志/会话跟着一起走。
  - 环境变量 AGENT_HOME：临时覆盖，优先级最高，适合多实例/脚本。

改目录会执行迁移（先拷后删），中途失败不会丢数据。
"""

import os
import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog, QMessageBox,
    QGroupBox, QCheckBox, QTextEdit,
)

from agent.util import agent_home_info, migrate_agent_home, clear_agent_home
from agent.keyring import (
    load_user_config, save_user_config, USER_CONFIG, config_status,
)
from agent.llm_client import get_llm_client, llm_status


# ==================== API 页 ====================

class ApiTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = load_user_config()
        self._build()
        self._load()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(8)

        self.combo_provider = QComboBox()
        self.combo_provider.setEditable(True)
        self.combo_provider.currentTextChanged.connect(self._on_provider_changed)

        self.edit_base = QLineEdit()
        self.edit_base.setPlaceholderText("https://api.deepseek.com")

        self.edit_model = QLineEdit()
        self.edit_model.setPlaceholderText("deepseek-chat")

        key_row = QHBoxLayout()
        self.edit_key = QLineEdit()
        self.edit_key.setEchoMode(QLineEdit.Password)
        self.edit_key.setPlaceholderText("sk-...")
        self.btn_toggle = QPushButton("显示")
        self.btn_toggle.setFixedWidth(56)
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.toggled.connect(self._toggle_key)
        key_row.addWidget(self.edit_key, 1)
        key_row.addWidget(self.btn_toggle)

        form.addRow("服务商", self.combo_provider)
        form.addRow("接口地址", self.edit_base)
        form.addRow("模型", self.edit_model)
        form.addRow("API Key", key_row)
        layout.addLayout(form)

        hint = QLabel(
            "提示：也可以用环境变量覆盖（优先级最高）——\n"
            "AGENT_API_KEY / AGENT_BASE_URL / AGENT_MODEL / AGENT_LLM_PROVIDER")
        hint.setStyleSheet("color:#7C93B0; font-size:11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        self.btn_test = QPushButton("🔌 测试连接")
        self.btn_test.setCursor(Qt.PointingHandCursor)
        self.btn_test.clicked.connect(self._test_connection)
        self.btn_open_file = QPushButton("📝 打开配置文件")
        self.btn_open_file.setCursor(Qt.PointingHandCursor)
        self.btn_open_file.clicked.connect(self._open_config_file)
        row.addWidget(self.btn_test)
        row.addWidget(self.btn_open_file)
        row.addStretch()
        layout.addLayout(row)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setFixedHeight(90)
        self.result.setStyleSheet(
            "background:rgba(4,14,30,0.6); color:#A9C8E8; border:1px solid rgba(82,179,241,0.15);"
            "border-radius:6px; font-size:12px; padding:6px;")
        layout.addWidget(self.result, 1)

        layout.addStretch()

    # ---------------- 读写 ----------------

    def _providers(self) -> dict:
        return (self.cfg.get("llm", {}) or {}).get("providers", {}) or {}

    def _load(self):
        providers = self._providers()
        self.combo_provider.blockSignals(True)
        self.combo_provider.clear()
        self.combo_provider.addItems(sorted(providers.keys()) or ["deepseek"])
        current = (self.cfg.get("llm", {}) or {}).get("provider", "deepseek")
        idx = self.combo_provider.findText(current)
        if idx >= 0:
            self.combo_provider.setCurrentIndex(idx)
        else:
            self.combo_provider.setEditText(current)
        self.combo_provider.blockSignals(False)
        self._on_provider_changed(self.combo_provider.currentText())

    def _on_provider_changed(self, name: str):
        p = self._providers().get(name, {}) or {}
        self.edit_base.setText(p.get("base_url", ""))
        self.edit_model.setText(p.get("model", ""))
        self.edit_key.setText(p.get("api_key", ""))

    def _toggle_key(self, checked: bool):
        self.edit_key.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        self.btn_toggle.setText("隐藏" if checked else "显示")

    def apply(self) -> dict:
        """把界面上的值写回用户配置"""
        name = self.combo_provider.currentText().strip() or "deepseek"
        self.cfg.setdefault("llm", {}).setdefault("providers", {}).setdefault(name, {})
        self.cfg["llm"]["provider"] = name
        self.cfg["llm"]["providers"][name].update({
            "base_url": self.edit_base.text().strip(),
            "model": self.edit_model.text().strip(),
            "api_key": self.edit_key.text().strip(),
        })
        save_user_config(self.cfg)
        get_llm_client(refresh=True)
        return {"success": True, "provider": name}

    # ---------------- 动作 ----------------

    def _open_config_file(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(USER_CONFIG))

    def _test_connection(self):
        self.btn_test.setEnabled(False)
        self.result.setPlainText("正在连接…")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.edit_key.text().strip(),
                            base_url=self.edit_base.text().strip() or None,
                            timeout=20)
            resp = client.chat.completions.create(
                model=self.edit_model.text().strip(),
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4,
            )
            text = (resp.choices[0].message.content or "").strip()
            self.result.setPlainText(
                f"✅ 连接成功\n模型：{self.edit_model.text().strip()}\n回应：{text[:80]}")
        except Exception as e:
            self.result.setPlainText(f"❌ 连接失败\n{type(e).__name__}: {e}")
        finally:
            self.btn_test.setEnabled(True)


# ==================== 数据目录页 ====================

class DataDirTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.new_path: str = ""
        self._build()
        self._refresh()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(10)

        self.label_path = QLabel("-")
        self.label_path.setWordWrap(True)
        self.label_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label_path.setStyleSheet("color:#5FE2FF; font-size:12px;")

        self.label_source = QLabel("-")
        self.label_source.setStyleSheet("color:#7C93B0; font-size:11px;")

        self.label_usage = QLabel("-")
        self.label_usage.setStyleSheet("color:#A9C8E8; font-size:11px;")
        self.label_usage.setWordWrap(True)

        box = QGroupBox("当前数据目录")
        form = QFormLayout(box)
        form.setSpacing(8)
        form.addRow("位置", self.label_path)
        form.addRow("来源", self.label_source)
        form.addRow("占用", self.label_usage)
        layout.addWidget(box)

        btn_row = QHBoxLayout()
        self.btn_change = QPushButton("📁 更改位置")
        self.btn_change.setCursor(Qt.PointingHandCursor)
        self.btn_change.clicked.connect(self._change_dir)
        self.btn_open = QPushButton("📂 打开")
        self.btn_open.setCursor(Qt.PointingHandCursor)
        self.btn_open.clicked.connect(self._open_dir)
        self.btn_default = QPushButton("↩ 恢复默认")
        self.btn_default.setCursor(Qt.PointingHandCursor)
        self.btn_default.clicked.connect(self._reset_default)
        btn_row.addWidget(self.btn_change)
        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(self.btn_default)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.check_portable = QCheckBox("便携模式：把位置记录到程序目录（拷到别的机器依然生效）")
        self.check_portable.setChecked(True)
        self.check_portable.setStyleSheet("color:#A9C8E8; font-size:11px;")
        layout.addWidget(self.check_portable)

        note = QLabel(
            "目录里包含：API 密钥、会话记录、日志、脱敏映射表、知识库索引。\n"
            "换位置会执行迁移（先复制到新位置，确认成功后再删旧的），失败不会丢数据。\n"
            "换盘后需要重启程序才完全生效。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#7C93B0; font-size:11px;")
        layout.addWidget(note)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setFixedHeight(80)
        self.status.setStyleSheet(
            "background:rgba(4,14,30,0.6); color:#A9C8E8; border:1px solid rgba(82,179,241,0.15);"
            "border-radius:6px; font-size:12px; padding:6px;")
        layout.addWidget(self.status, 1)

    def _refresh(self):
        info = agent_home_info()
        self.label_path.setText(info["path"])
        self.label_source.setText(info["source_label"])
        items = info["items_kb"]
        if items:
            detail = "、".join(f"{k} {v}KB" for k, v in items.items())
            self.label_usage.setText(f"共 {info['total_kb']} KB（{detail}）")
        else:
            self.label_usage.setText("暂无数据")

    def _open_dir(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(agent_home_info()["path"]))

    def _change_dir(self):
        current = agent_home_info()["path"]
        path = QFileDialog.getExistingDirectory(self, "选择新的数据目录",
                                                os.path.dirname(current) or "")
        if not path:
            return
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(current):
            QMessageBox.information(self, "提示", "选择的目录与当前位置相同。")
            return

        # 选中的目录里已经有配置文件就直接用（比如换机器后指回旧位置），
        # 否则在其下建一个子目录，避免把配置散落到用户选的盘根目录
        if os.path.exists(os.path.join(path, "config.json")):
            target = path
        else:
            target = os.path.join(path, "AgentData")

        reply = QMessageBox.question(
            self, "确认迁移",
            f"将把数据目录迁移到：\n{target}\n\n"
            f"现有配置、会话、日志会一起搬过去。\n"
            f"换盘后需重启程序生效。是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        portable = self.check_portable.isChecked()
        result = migrate_agent_home(target, portable=portable)
        if result.get("success"):
            self.status.setPlainText(
                f"✅ 已迁移到：{result['path']}\n"
                f"搬移内容：{', '.join(result.get('moved', []))}\n"
                f"旧目录已删除：{'是' if result.get('old_removed') else '否'}\n"
                f"请重启程序使新位置完全生效。")
            self._refresh()
        else:
            self.status.setPlainText(f"❌ 迁移失败\n{result.get('error', '')}")
            QMessageBox.warning(self, "迁移失败", result.get("error", "未知错误"))

    def _reset_default(self):
        reply = QMessageBox.question(
            self, "恢复默认",
            "将清除自定义位置，回到用户主目录下的默认目录。\n"
            "（当前目录里的文件不会自动搬回，需要手动复制）\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        result = clear_agent_home()
        if result.get("success"):
            self.status.setPlainText(
                f"已恢复默认：{result['path']}\n请重启程序。\n"
                f"如需保留旧数据，请手动复制到该目录。")
            self._refresh()
        else:
            QMessageBox.warning(self, "失败", result.get("error", "未知错误"))


# ==================== 对话框 ====================

class SettingsDialog(QDialog):
    def __init__(self, parent=None, on_applied=None):
        super().__init__(parent)
        self.on_applied = on_applied
        self.setWindowTitle("⚙ 设置")
        self.resize(560, 480)
        self.setStyleSheet("""
            QDialog { background:#061428; color:#D6E8FF; }
            QLabel { color:#C8DDF0; }
            QLineEdit, QComboBox {
                background:rgba(4,14,30,0.7); border:1px solid rgba(82,179,241,0.18);
                border-radius:6px; padding:6px 8px; color:#E0EDFF; }
            QLineEdit:focus, QComboBox:focus { border:1px solid rgba(105,215,255,0.45); }
            QPushButton { background:rgba(8,80,165,0.6); color:#E0EDFF;
                border:1px solid rgba(91,204,255,0.22); border-radius:6px; padding:6px 12px; }
            QPushButton:hover { background:rgba(11,101,201,0.8); }
            QGroupBox { border:1px solid rgba(82,179,241,0.15); border-radius:6px;
                margin-top:10px; padding-top:10px; color:#9FC4E8; }
            QTabWidget::pane { border:1px solid rgba(82,179,241,0.15); }
            QTabBar::tab { background:rgba(8,26,48,0.6); color:#A9C8E8;
                padding:7px 18px; border-top-left-radius:6px; border-top-right-radius:6px; }
            QTabBar::tab:selected { background:rgba(11,101,201,0.55); color:#fff; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QTabWidget()
        self.api_tab = ApiTab(self)
        self.data_tab = DataDirTab(self)
        tabs.addTab(self.api_tab, "🔑 API")
        tabs.addTab(self.data_tab, "📁 数据目录")
        layout.addWidget(tabs, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(14, 10, 14, 12)
        footer.setSpacing(8)
        self.label_status = QLabel("")
        self.label_status.setStyleSheet("color:#7C93B0; font-size:11px;")
        self.btn_save = QPushButton("保存")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.clicked.connect(self._save)
        self.btn_close = QPushButton("关闭")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.clicked.connect(self.accept)
        footer.addWidget(self.label_status, 1)
        footer.addWidget(self.btn_save)
        footer.addWidget(self.btn_close)
        layout.addLayout(footer)

    def _save(self):
        result = self.api_tab.apply()
        if result.get("success"):
            st = llm_status()
            self.label_status.setText(
                f"已保存 · 当前 {st.get('provider', '')}/{st.get('model', '')} · "
                f"{'可用' if st.get('available') else 'Key 未配置'}")
            if self.on_applied:
                try:
                    self.on_applied()
                except Exception:
                    pass
        else:
            self.label_status.setText(f"保存失败: {result.get('error', '')}")
