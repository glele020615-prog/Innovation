"""
悬浮球 + 对话弹窗
==================
在任何页面点击悬浮球即可呼出 AI 诊断助手。
"""

import sys, os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PySide6.QtCore import Qt, QPoint, QSize, QPropertyAnimation, QEasingCurve, QTimer, Signal
from PySide6.QtGui import QFont, QColor, QPainter, QPainterPath, QBrush, QPen, QMouseEvent
from PySide6.QtWidgets import (
    QPushButton, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGraphicsDropShadowEffect, QApplication, QWidget, QSizeGrip,
)


# ==================== 对话弹窗 ====================

class ChatDialog(QDialog):
    """包含 AI 对话界面的模式对话框（WA_DeleteOnClose=False，关闭即复用）

    mode="doctor"  → 医生端 AgentChatPage（工具调用 + RAG）
    mode="patient" → 患者端 PatientChatPage（仅知识库问答，无工具）
    """

    closed = Signal()

    _TITLES = {"doctor": "🤖 AI 智能诊断助手", "patient": "🌿 AI 健康问答助手"}

    def __init__(self, parent=None, main_window=None, mode: str = "doctor"):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle(self._TITLES.get(mode, self._TITLES["doctor"]))
        self.setMinimumSize(640, 520)
        self.resize(680, 560)
        self.setWindowFlags(
            Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint
        )
        self.setAttribute(Qt.WA_DeleteOnClose, False)  # 复用实例

        # 整体风格
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #040E1E, stop:0.5 #081A32, stop:1 #041020);
                border: 1px solid rgba(80, 180, 255, 0.25);
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 自定义标题栏
        title_bar = QFrame()
        title_bar.setFixedHeight(42)
        title_bar.setStyleSheet("""
            QFrame {
                background: rgba(6, 20, 42, 0.85);
                border-bottom: 1px solid rgba(96, 190, 255, 0.18);
            }
        """)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(14, 0, 8, 0)
        tb_layout.setSpacing(8)

        title_icon = QLabel("🧠" if mode == "doctor" else "🌿")
        title_icon.setFixedSize(26, 26)
        title_icon.setAlignment(Qt.AlignCenter)

        title_text = QLabel(self._TITLES.get(mode, "AI 助手").split(" ", 1)[-1])
        title_text.setStyleSheet("color:#E7F5FF; font-size:14px; font-weight:700; background:transparent;")

        self._status = QLabel("就绪")
        self._status.setStyleSheet("color:#89A7C4; font-size:11px; background:transparent;")

        tb_layout.addWidget(title_icon)
        tb_layout.addWidget(title_text)
        tb_layout.addStretch()
        tb_layout.addWidget(self._status)

        # 「新对话」：复用同一个弹窗实例时，给用户一个清空历史的入口
        self._new_chat_btn = QPushButton("＋ 新对话")
        self._new_chat_btn.setCursor(Qt.PointingHandCursor)
        self._new_chat_btn.setToolTip("保留当前会话到左侧列表，另起一个干净对话")
        self._new_chat_btn.setStyleSheet("""
            QPushButton { background:rgba(8,80,165,0.5); color:#E0EDFF;
                border:1px solid rgba(91,204,255,0.22); border-radius:6px;
                padding:3px 10px; font-size:11px; }
            QPushButton:hover { background:rgba(11,101,201,0.75); }
        """)
        self._new_chat_btn.clicked.connect(self._start_new_chat)
        tb_layout.addWidget(self._new_chat_btn)

        layout.addWidget(title_bar)

        # 内嵌对话页（按模式惰性导入：患者端不必加载整个医生端 Agent 栈）
        if mode == "patient":
            from patient_chat_page import PatientChatPage
            self.chat_page = PatientChatPage()
        else:
            from agent_chat_page import AgentChatPage
            self.chat_page = AgentChatPage(main_window=main_window)
        # 覆盖 AgentChatPage 的样式让它在弹窗中更紧凑
        self.chat_page.setStyleSheet("")
        layout.addWidget(self.chat_page, 1)

        # 监听工具调用状态
        default_status = ("引擎: DeepSeek | 诊断: SA-STGCN" if mode == "doctor"
                          else "引擎: DeepSeek | 仅知识库问答")
        self.chat_page.status_label.setText(default_status)
        self.chat_page.status_label.setStyleSheet(
            "color:#89A7C4; font-size:11px;"
            "background:rgba(8,52,128,0.35); border:1px solid rgba(96,190,255,0.18);"
            "border-radius:8px; padding:2px 10px;"
        )

    def push_case_data(self, case_data: dict):
        """接收来自主窗口的诊断数据（仅医生端；患者端没有工具链，忽略）。"""
        if self.mode != "patient" and case_data:
            self.chat_page.load_case_data(case_data)

    def set_status(self, text: str):
        self._status.setText(text)

    def _start_new_chat(self):
        self.chat_page.new_session()

    def closeEvent(self, event):
        """关闭 ≠ 销毁（WA_DeleteOnClose=False）：隐藏并通知悬浮球，历史保留"""
        self.closed.emit()
        super().closeEvent(event)


# ==================== 悬浮球 ====================

class FloatingBall(QPushButton):
    """可拖拽的悬浮球，始终浮在所有页面之上"""

    BALL_SIZE = 56
    ICON_SIZE = 28

    def __init__(self, parent: QWidget, main_window=None, mode: str = "doctor"):
        super().__init__(parent)
        self.main_window = main_window
        self.mode = mode
        self._dialog: ChatDialog | None = None
        self._pending_case: dict | None = None   # 弹窗还没建好时先存着
        self._dragging = False
        self._drag_start = QPoint()
        self._ball_pos = QPoint()
        self._entered = False
        self._pulse_anim: QPropertyAnimation | None = None

        self.setFixedSize(self.BALL_SIZE, self.BALL_SIZE)
        self.setCursor(Qt.PointingHandCursor)
        if mode == "patient":
            self.setToolTip("🌿 点击打开 AI 健康问答助手\n拖拽可移动 | 右键隐藏")
        else:
            self.setToolTip("🅥 点击打开 AI 诊断助手\n拖拽可移动 | 右键隐藏")

        self.clicked.connect(self._on_click)

        # 位置：右下角
        self._reposition()

        # 入场动画 —— 从边缘弹入
        self._animate_enter()

    def _ensure_dialog(self) -> ChatDialog:
        """惰性创建弹窗，并把暂存的病例数据补进去"""
        if self._dialog is None:
            self._dialog = ChatDialog(self.parentWidget(), main_window=self.main_window, mode=self.mode)
            self._dialog.closed.connect(self._on_dialog_closed)
            if self._pending_case:
                self._dialog.push_case_data(self._pending_case)
                self._pending_case = None
        return self._dialog

    def _on_dialog_closed(self):
        """弹窗关闭：实例保留（历史与会话不丢），但停掉仍在跑的任务，避免后台线程空转"""
        if self._dialog is not None:
            try:
                self._dialog.chat_page.stop_worker()
            except Exception:
                pass

    def _on_click(self):
        """点击弹出对话窗口"""
        dialog = self._ensure_dialog()
        # 相对主窗口居中
        if self.main_window:
            mw_geo = self.main_window.geometry()
            dx = (mw_geo.width() - dialog.width()) // 2
            dy = (mw_geo.height() - dialog.height()) // 2
            dialog.move(mw_geo.x() + dx, mw_geo.y() + dy)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def push_case_data(self, data: dict):
        """主窗口推送诊断数据。弹窗还没建好就先存着，用户点开时再注入，
        不再为了推送数据而在后台偷偷创建一个窗口。"""
        if not data:
            return
        if self._dialog is None:
            self._pending_case = data
            return
        self._dialog.push_case_data(data)

    def set_status(self, text: str):
        if self._dialog:
            self._dialog.set_status(text)

    # ==================== 位置管理 ====================

    def _reposition(self):
        """将悬浮球定位到父窗口右下角"""
        pw = self.parentWidget()
        if pw:
            x = pw.width() - self.BALL_SIZE - 20
            y = pw.height() - self.BALL_SIZE - 24
            self.move(x, y)

    def update_position(self):
        """外部调用（窗口 resize 时）"""
        self._reposition()

    def _animate_enter(self):
        """弹入动画"""
        anim = QPropertyAnimation(self, b"pos")
        anim.setDuration(400)
        anim.setEasingCurve(QEasingCurve.OutBack)
        target = self.pos()
        anim.setStartValue(QPoint(target.x(), target.y() + 80))
        anim.setEndValue(target)
        anim.start()

    # ==================== 拖拽 ====================

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start = event.globalPosition().toPoint()
            self._ball_pos = self.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = event.globalPosition().toPoint() - self._drag_start
            new_x = self._ball_pos.x() + delta.x()
            new_y = self._ball_pos.y() + delta.y()
            pw = self.parentWidget()
            if pw:
                new_x = max(0, min(new_x, pw.width() - self.BALL_SIZE))
                new_y = max(0, min(new_y, pw.height() - self.BALL_SIZE))
            self.move(new_x, new_y)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = (event.globalPosition().toPoint() - self._drag_start).manhattanLength()
            self._dragging = False
            self.setCursor(Qt.PointingHandCursor)
            # 轻微拖动不计为拖拽，仍触发点击
            if delta > 5:
                return  # 阻止 clicked 信号，仅拖拽
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        self._entered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._entered = False
        self.update()
        super().leaveEvent(event)

    # ==================== 绘制 ====================

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w, h = self.width(), self.height()
        center = w // 2
        radius = center - 2

        # 外发光
        glow_radius = radius + 6 if self._entered else radius + 4
        glow = QPainterPath()
        glow.addEllipse(QPoint(center, center), glow_radius, glow_radius)
        if self._entered:
            glow_color = QColor(80, 180, 255, 60)
        else:
            glow_color = QColor(60, 150, 240, 30)
        painter.fillPath(glow, glow_color)

        # 主体圆
        body = QPainterPath()
        body.addEllipse(QPoint(center, center), radius, radius)

        # 渐变填充
        from PySide6.QtGui import QRadialGradient
        grad = QRadialGradient(center - 5, center - 8, radius * 1.2)
        if self._entered:
            grad.setColorAt(0, QColor(70, 170, 255))
            grad.setColorAt(0.6, QColor(20, 70, 160))
            grad.setColorAt(0.9, QColor(8, 30, 80))
        else:
            grad.setColorAt(0, QColor(50, 140, 230))
            grad.setColorAt(0.6, QColor(15, 55, 130))
            grad.setColorAt(0.9, QColor(6, 22, 60))
        painter.fillPath(body, grad)

        # 边框
        border_color = QColor(100, 200, 255, 180) if self._entered else QColor(70, 160, 230, 120)
        painter.setPen(QPen(border_color, 1.5))
        painter.drawPath(body)

        # 图标文字
        painter.setPen(QColor(255, 255, 255, 230))
        font = self.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(0, -1, 0, 0), Qt.AlignCenter, "AI")

        painter.end()

    def showEvent(self, event):
        self.raise_()
        super().showEvent(event)


# ==================== 快速测试 ====================
if __name__ == "__main__":
    from PySide6.QtWidgets import QMainWindow, QLabel
    app = QApplication(sys.argv)

    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("悬浮球测试")
            self.resize(800, 500)
            self.setStyleSheet("background:#040E1E;")

            label = QLabel("← 点击右下角悬浮球")
            label.setStyleSheet("color:#89A7C4;font-size:18px;")
            label.setAlignment(Qt.AlignCenter)
            self.setCentralWidget(label)

            self.ball = FloatingBall(self)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self.ball.update_position()

    win = TestWindow()
    win.show()
    sys.exit(app.exec())
