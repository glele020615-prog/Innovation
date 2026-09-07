"""
登录窗口。
视觉沿用医生端深色 QSS，登录成功按 role 字段分发到主窗口或患者窗口。
"""
import os
import sys
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QFontDatabase, QPainter, QPixmap, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

import auth


def _resource_path(*parts):
    """兼容开发环境与打包环境的资源路径解析。"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


class _BackgroundWidget(QWidget):
    """以一张图片铺满整个窗口的容器：等比放大至覆盖、超出部分居中裁剪。"""

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap(image_path)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


def _font(pt_size=10):
    fams = QFontDatabase.families()
    for f in ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Segoe UI"):
        if f in fams:
            return QFont(f, pt_size)
    return QFont("Microsoft YaHei UI", pt_size)


LOGIN_QSS = """
QWidget#loginRoot {
    background-color: #07152B;
}
QLabel#brandTitle {
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#brandSub {
    color: #9EC4F5;
    font-size: 13px;
    letter-spacing: 4px;
}
QLabel#hint {
    color: #6F86A8;
    font-size: 12px;
}
QLabel#roleHint {
    color: #82B1E8;
    font-size: 13px;
}
QLabel#err {
    color: #FF8A8A;
    font-size: 12px;
}
QLineEdit {
    background-color: #04101F;
    color: #FFFFFF;
    border: 1px solid rgba(82,179,241,0.30);
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 14px;
    selection-background-color: #1258A8;
}
QLineEdit:focus {
    border: 1px solid rgba(130,220,255,0.65);
}
QPushButton#primaryBtn {
    background-color: rgba(12,100,195,0.95);
    color: #FFFFFF;
    border: 1px solid rgba(140,225,255,0.55);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 15px;
    font-weight: 700;
    min-height: 40px;
}
QPushButton#primaryBtn:hover {
    background-color: rgba(20,120,225,1.0);
    border-color: rgba(160,230,255,0.85);
}
QPushButton#primaryBtn:disabled {
    background-color: #1B3A5C;
    color: #6F86A8;
    border-color: #1F4A75;
}
QFrame#card {
    background-color: #0B1E3A;
    border: 1px solid rgba(82,179,241,0.18);
    border-radius: 12px;
}
"""


class LoginWindow(QMainWindow):
    """登录窗口。按角色回调到 on_login_success(role, display_name)。"""

    def __init__(self, on_login_success):
        super().__init__()
        self.on_login_success = on_login_success
        self.setWindowTitle("智影识微 · 登录")
        self.setFixedSize(900, 560)
        self.setFont(_font(10))

        root = _BackgroundWidget(_resource_path("assets", "login.png"))
        root.setObjectName("loginRoot")
        self.setCentralWidget(root)

        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 左侧品牌区
        left = QFrame()
        left.setStyleSheet(
            "background-color: rgba(4,16,31,150);"
            "border-right: 1px solid rgba(82,179,241,0.15);"
        )
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(40, 40, 40, 40)
        left_lay.setSpacing(14)
        left_lay.addStretch(1)

        title = QLabel("智影识微")
        title.setObjectName("brandTitle")
        sub = QLabel("基于 MRI 影像的 MCI 风险智能预测平台")
        sub.setObjectName("brandSub")
        sub.setWordWrap(True)
        left_lay.addWidget(title)
        left_lay.addWidget(sub)

        # 角色提示（视觉引导，不强制）
        # role_tip = QLabel("医生账号 → 进入影像分析与诊断界面\n患者账号 → 进入报告与认知训练界面")
        # role_tip.setObjectName("roleHint")
        # role_tip.setWordWrap(True)
        # left_lay.addWidget(role_tip)

        left_lay.addStretch(2)

        tip = QLabel("登录账号由系统分配，请向管理员获取")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        left_lay.addWidget(tip)

        outer.addWidget(left, 5)

        # 右侧登录卡
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(40, 60, 40, 40)
        right_lay.setSpacing(18)

        right_lay.addStretch(1)

        card = QFrame(); card.setObjectName("card")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(28, 28, 28, 28)
        card_lay.setSpacing(14)

        card_title = QLabel("账号登录")
        card_title.setStyleSheet("color:#FFFFFF; font-size:20px; font-weight:700;")
        card_sub = QLabel("请输入您的账号与密码")
        card_sub.setStyleSheet("color:#82B1E8; font-size:12px;")

        self.edt_user = QLineEdit(); self.edt_user.setPlaceholderText("账号")
        self.edt_pass = QLineEdit(); self.edt_pass.setPlaceholderText("密码")
        self.edt_pass.setEchoMode(QLineEdit.Password)
        self.edt_pass.returnPressed.connect(self._try_login)

        self.lbl_err = QLabel("")
        self.lbl_err.setObjectName("err")
        self.lbl_err.setVisible(False)

        self.btn_login = QPushButton("登 录")
        self.btn_login.setObjectName("primaryBtn")
        self.btn_login.setCursor(Qt.PointingHandCursor)
        self.btn_login.clicked.connect(self._try_login)

        for w in (card_title, card_sub, self.edt_user, self.edt_pass, self.lbl_err, self.btn_login):
            card_lay.addWidget(w)

        # 给卡片加点阴影
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 160))
        card.setGraphicsEffect(shadow)

        right_lay.addWidget(card)
        right_lay.addStretch(1)

        outer.addWidget(right, 4)

        self.setStyleSheet(LOGIN_QSS)
        self.edt_user.setFocus()

    def _try_login(self):
        u = self.edt_user.text().strip()
        p = self.edt_pass.text()
        if not u or not p:
            self._show_err("请输入账号和密码")
            return
        self.btn_login.setEnabled(False)
        QApplication.processEvents()
        ok, role, name, err = auth.verify(u, p)
        self.btn_login.setEnabled(True)
        print(f"[login] verify ok={ok} role={role} name={name}", flush=True)
        if not ok:
            self._show_err(err or "登录失败")
            self.edt_pass.clear()
            self.edt_pass.setFocus()
            return
        # 关闭登录窗，外部负责接住角色
        try:
            print("[login] dispatching on_login_success ...", flush=True)
            self.on_login_success(role=role, username=u, display_name=name or u)
            print("[login] on_login_success returned", flush=True)
        except Exception:
            import traceback
            print("[login] EXCEPTION in on_login_success:", flush=True)
            traceback.print_exc()
            self.btn_login.setEnabled(True)

    def _show_err(self, msg):
        self.lbl_err.setText(msg)
        self.lbl_err.setVisible(True)


if __name__ == "__main__":
    # 自检：仅打开窗口，不进任何业务界面
    app = QApplication(sys.argv)
    app.setFont(_font(10))

    def cb(**kw):
        print("login ok:", kw)
        app.quit()

    w = LoginWindow(on_login_success=cb)
    w.show()
    sys.exit(app.exec())