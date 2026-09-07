import sys
import os

_SI_KEY = "ZhyingShiWei.SingleInstance"


def _early_single_instance_channel():
    """单实例守卫。必须在所有重量级导入之前运行：QLocalSocket/Server 的阻塞
    用法不依赖事件循环，二次双击 exe 时不必等 torch/WebEngine 加载完就能退出。
    返回主实例的 server；二次启动返回 None（此时已通知主实例恢复窗口）。"""
    from PySide6.QtNetwork import QLocalServer, QLocalSocket

    sock = QLocalSocket()
    sock.connectToServer(_SI_KEY)
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.waitForBytesWritten(300)
        sock.disconnectFromServer()
        return None
    QLocalServer.removeServer(_SI_KEY)   # 清理上次异常退出留下的残留监听
    server = QLocalServer()
    if not server.listen(_SI_KEY):
        return None   # 监听失败不阻止启动，只是失去单实例功能
    return server


_SI_SERVER = None
if __name__ == "__main__":
    _SI_SERVER = _early_single_instance_channel()
    if _SI_SERVER is None:
        sys.exit(0)   # 已有实例在运行：已通知其恢复窗口，本进程直接退出

from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QColor, QFont, QIcon, QFontDatabase, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QMenu, QMessageBox, QPushButton,
                               QStackedWidget, QTableWidget, QTableWidgetItem,
                               QHeaderView, QSizePolicy, QVBoxLayout, QWidget,
                               QGraphicsDropShadowEffect,
                               QGraphicsOpacityEffect,
                               QSystemTrayIcon)
from viewer import fMRIViewWidget
import nibabel as nib
from Read_Write_JSON import R_W_JSON
from datetime import datetime
from Data_Pre import PreprocessingPage
from predict import DiagnosisPage
from individual_report_page import IndividualReportPage
from feature_extraction_page import FeatureExtractionPage
from floating_ball import FloatingBall


def get_resource_path(*parts):
    """兼容开发环境与打包环境的资源路径解析。"""
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, *parts)


def get_preferred_ui_font(point_size=10):
    families = QFontDatabase.families()
    for family in ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        if family in families:
            return QFont(family, point_size)
    font = QFont()
    font.setPointSize(point_size)
    return font


# 主窗口用的是深色主题，但 QMessageBox / QDialog 这类系统弹窗属于顶层窗口，
# 不会继承主窗口样式表，在深色模式下会出现"深底黑字"。这里在应用级统一补一套深色皮肤。
DARK_DIALOG_QSS = """
    QMessageBox, QDialog, QInputDialog, QProgressDialog {
        background-color: #07152B;
        color: #FFFFFF;
    }
    QMessageBox QLabel, QDialog QLabel, QInputDialog QLabel, QProgressDialog QLabel {
        color: #FFFFFF;
        background: transparent;
    }
    QMessageBox QPushButton, QDialog QPushButton, QProgressDialog QPushButton {
        background-color: rgba(10, 48, 98, 0.90);
        color: #FFFFFF;
        border: 1px solid rgba(82, 179, 241, 0.38);
        border-radius: 6px;
        padding: 5px 16px;
        min-width: 76px;
    }
    QMessageBox QPushButton:hover, QDialog QPushButton:hover, QProgressDialog QPushButton:hover {
        background-color: rgba(20, 85, 155, 0.95);
        border-color: rgba(130, 220, 255, 0.65);
    }
    QMessageBox QPushButton:default, QDialog QPushButton:default {
        background-color: rgba(12, 100, 195, 0.95);
        border-color: rgba(140, 225, 255, 0.75);
        font-weight: 700;
    }
    QDialog QLineEdit, QInputDialog QLineEdit {
        background-color: #04101F;
        color: #FFFFFF;
        border: 1px solid rgba(82, 179, 241, 0.30);
        border-radius: 4px;
        padding: 3px 6px;
        selection-background-color: #1258A8;
    }
    QToolTip {
        background-color: #0B1E3A;
        color: #FFFFFF;
        border: 1px solid rgba(120, 200, 255, 0.35);
        padding: 3px 6px;
    }
"""


def apply_dark_dialog_style(app: QApplication):
    """给标准弹窗补深色皮肤，避免深底黑字看不清。"""
    app.setStyleSheet(app.styleSheet() + DARK_DIALOG_QSS)


class CoverImageWidget(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap(image_path)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self._pixmap.isNull():
            painter.fillRect(self.rect(), QColor(4, 13, 28))
            return

        target = self.rect()
        scaled = self._pixmap.scaled(
            target.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        crop_x = max(0, (scaled.width() - target.width()) // 2)
        crop_y = max(0, (scaled.height() - target.height()) // 2)
        source = scaled.rect().adjusted(crop_x, crop_y, -crop_x, -crop_y)
        painter.drawPixmap(target, scaled, source)
        painter.fillRect(target, QColor(2, 10, 24, 72))


class HomePageWidget(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setObjectName("homePage")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._background = CoverImageWidget(image_path, self)
        self._background.lower()

    def resizeEvent(self, event):
        self._background.setGeometry(self.rect())
        super().resizeEvent(event)


class AngledNavButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(52)
        self.setMinimumWidth(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #E0EDFF;
                border: none;
                padding: 0 20px;
                font-size: 15px;
                font-weight: 700;
            }
        """)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        rect = self.rect().adjusted(1, 1, -1, -1)
        cut = 12
        path = QPainterPath()
        path.moveTo(rect.left() + cut, rect.top())
        path.lineTo(rect.right() - cut, rect.top())
        path.lineTo(rect.right(), rect.center().y())
        path.lineTo(rect.right() - cut, rect.bottom())
        path.lineTo(rect.left() + cut, rect.bottom())
        path.lineTo(rect.left(), rect.center().y())
        path.closeSubpath()

        checked = self.isChecked()
        hovered = self.underMouse()

        if checked:
            top_color = QColor(42, 150, 255, 215)
            bottom_color = QColor(8, 52, 128, 215)
            border_color = QColor(116, 219, 255, 255)
            glow_color = QColor(105, 215, 255, 150)
        elif hovered:
            top_color = QColor(20, 78, 150, 165)
            bottom_color = QColor(8, 33, 78, 160)
            border_color = QColor(104, 196, 255, 150)
            glow_color = QColor(105, 215, 255, 80)
        else:
            top_color = QColor(10, 26, 54, 70)
            bottom_color = QColor(6, 18, 38, 40)
            border_color = QColor(80, 146, 208, 70)
            glow_color = QColor(105, 215, 255, 25)

        painter.fillPath(path, top_color)
        painter.fillPath(path, bottom_color)
        painter.setPen(border_color)
        painter.drawPath(path)

        if glow_color.alpha() > 0:
            painter.setPen(glow_color)
            painter.drawLine(rect.left() + cut + 5, rect.top() + 1, rect.right() - cut - 5, rect.top() + 1)

        if checked:
            painter.setPen(QColor(188, 244, 255, 255))
            painter.drawLine(rect.left() + cut + 10, rect.top() + 1, rect.right() - cut - 10, rect.top() + 1)

        painter.setPen(QColor(255, 255, 255) if checked else QColor(224, 237, 255))
        painter.drawText(rect.adjusted(0, 0, 0, -1), Qt.AlignCenter, self.text())


class NavBeamSeparator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(18)
        self.setFixedHeight(42)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        center_x = self.width() / 2
        top = 6
        bottom = self.height() - 6

        painter.setPen(QColor(51, 128, 255, 42))
        painter.drawLine(int(center_x), top, int(center_x), bottom)
        painter.setPen(QColor(112, 220, 255, 190))
        painter.drawLine(int(center_x), top + 2, int(center_x), bottom - 2)
        painter.setPen(QColor(255, 255, 255, 110))
        painter.drawLine(int(center_x) - 1, top + 6, int(center_x) - 1, bottom - 6)


class HomeImageCard(QFrame):
    def __init__(self, title, image_path, desc1, desc2):
        super().__init__()
        self.setObjectName("homeFeatureCard")
        self.setMinimumHeight(198)
        self.setStyleSheet("""
            QFrame#homeFeatureCard {
                background-color: rgba(6, 18, 40, 0.50);
                border: 1px solid rgba(70, 170, 235, 0.20);
                border-radius: 16px;
            }
            QLabel#featureTitle {
                color: #5FE2FF;
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#featureText {
                color: #D6E8FF;
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#featureDesc {
                color: #A9BEDA;
                font-size: 14px;
                line-height: 1.55;
            }
        """)

        glow = QGraphicsDropShadowEffect(self)
        glow.setColor(QColor(36, 157, 255, 110))
        glow.setBlurRadius(30)
        glow.setOffset(0, 0)
        self.setGraphicsEffect(glow)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        image_label = QLabel()
        image_label.setObjectName("featureImage")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setFixedSize(300, 180)
        image_label.setStyleSheet("""
            QLabel {
                background-color: rgba(4, 14, 30, 0.68);
                border-radius: 10px;
                border: 1px solid rgba(82, 179, 241, 0.16);
                color: #79B6D8;
                font-size: 14px;
            }
        """)

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                280,
                168,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            image_label.setPixmap(pixmap)
        else:
            image_label.setText("图片未找到")

        text_layout = QVBoxLayout()
        text_layout.setSpacing(20)
        text_layout.setContentsMargins(0, 8, 8, 8)

        title_label = QLabel(f"{title}")
        title_label.setObjectName("featureTitle")
        title_label.setContentsMargins(0, 0, 0, 4)

        text_line_1 = QLabel(desc1)
        text_line_1.setObjectName("featureText")
        text_line_1.setWordWrap(True)
        text_line_1.setContentsMargins(0, 4, 0, 0)

        desc_label = QLabel(desc2)
        desc_label.setObjectName("featureDesc")
        desc_label.setWordWrap(True)

        text_layout.addWidget(title_label)
        text_layout.addWidget(text_line_1)
        text_layout.addWidget(desc_label)
        text_layout.addStretch()

        layout.addWidget(image_label)
        layout.addLayout(text_layout, 1)


class ADDiagnosisSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("阿尔兹海默症  智能预测系统 v2.0")
        self.resize(1520, 900)
        self.setMinimumSize(1320, 780)
        self.shared_case_data = None
        self.setFont(get_preferred_ui_font(10))

        self.setStyleSheet("""
            QMainWindow {
                background-color: #030B18;
                color: #E7F2FF;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            }
            QFrame#topNavBar {
                background-color: rgba(3, 11, 24, 0.86);
                border-bottom: 1px solid rgba(96, 190, 255, 0.22);
            }
            QLabel#appTitle {
                font-size: 24px;
                font-weight: 700;
                color: #E9F4FF;
                letter-spacing: 1px;
            }
            QPushButton#navButton {
                background-color: transparent;
                color: #E0EDFF;
                border: none;
                padding: 0 20px;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton#navButton:hover {
                background-color: transparent;
            }
            QPushButton#navButton:checked {
                color: #FFFFFF;
            }
            QStackedWidget#contentStack {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #020913,
                    stop:0.5 #061C3B,
                    stop:1 #02101F
                );
            }
            QWidget#homePage {
                background: transparent;
            }
            QFrame#heroPanel {
                background-color: rgba(4, 18, 40, 0.40);
                border: 1px solid rgba(91, 183, 255, 0.18);
                border-radius: 22px;
            }
            QLabel#heroTag {
                color: #9ECCE8;
                font-size: 16px;
                font-weight: 500;
            }
            QLabel#heroTitle {
                color: #E7F5FF;
                font-size: 50px;
                font-weight: 800;
                line-height: 1.12;
            }
            QLabel#heroTitleAccent {
                color: #44CDFF;
            }
            QLabel#heroDescription {
                color: #B7D1ED;
                font-size: 16px;
                line-height: 1.6;
            }
            QFrame#heroVisualPanel {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(8, 24, 52, 150),
                    stop:1 rgba(7, 35, 76, 120)
                );
                border: 1px solid rgba(96, 188, 255, 0.24);
                border-radius: 16px;
            }
            QLabel#heroBrainBg {
                border: none;
                color: #7CB8E4;
                font-size: 14px;
            }
            QPushButton#startButton {
                background-color: rgba(8, 80, 165, 0.72);
                color: #FFFFFF;
                border: 1px solid rgba(91, 204, 255, 0.52);
                border-radius: 12px;
                padding: 10px 18px;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton#startButton:hover {
                background-color: rgba(11, 101, 201, 0.86);
            }
            QLabel#versionText {
                color: #89A7C4;
                font-size: 13px;
            }
        """)

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setCentralWidget(self.main_widget)

        self.build_top_navigation()

        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("contentStack")
        self.main_layout.addWidget(self.content_stack)

        self.btn_home = AngledNavButton("🏠 首页概览")
        self.btn_data = AngledNavButton("📂 数据加载")
        self.btn_preproc = AngledNavButton("⚙ 预处理")
        self.btn_feature = AngledNavButton("🔬 特征提取")
        self.btn_predict = AngledNavButton("🧠 疾病预测诊断")
        self.btn_report = AngledNavButton("📋 个体报告分析")

        nav_buttons = [self.btn_home, self.btn_data, self.btn_preproc, self.btn_feature,
                       self.btn_predict, self.btn_report]
        for index, btn in enumerate(nav_buttons):
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            self.nav_layout.addWidget(btn, 1)
            if index != len(nav_buttons) - 1:
                self.nav_layout.addWidget(NavBeamSeparator(self.nav_buttons_host))


        # 账号区：默认隐藏，调用 attach_account_action 后显示
        self.account_label = QLabel("")
        self.account_label.setStyleSheet("color:#9EC4F5; font-size:13px; padding:0 12px;")
        self.account_label.setVisible(False)
        self.nav_layout.addWidget(self.account_label)

        self.btn_account = QPushButton("切换账号")
        self.btn_account.setObjectName("navButton")
        self.btn_account.setVisible(False)
        self.btn_account.setCursor(Qt.PointingHandCursor)
        self.btn_account.clicked.connect(self._on_switch_account)
        self.nav_layout.addWidget(self.btn_account)

    def _on_switch_account(self):
        if getattr(self, "_on_logout_cb", None):
            self._on_logout_cb()

    def attach_account_action(self, username: str, display_name: str, on_logout):
        """由外部（_launch_app）调用，挂上当前账号显示与切换回调。"""
        self._current_username = username
        self._on_logout_cb = on_logout
        self.account_label.setText(f"医生：{display_name}")
        self.account_label.setVisible(True)
        self.btn_account.setVisible(True)
        self.init_pages()
        self.btn_home.clicked.connect(lambda: self.content_stack.setCurrentIndex(0))
        self.btn_data.clicked.connect(lambda: self.content_stack.setCurrentIndex(1))
        self.btn_preproc.clicked.connect(lambda: self.content_stack.setCurrentIndex(2))
        self.btn_feature.clicked.connect(lambda: self.content_stack.setCurrentIndex(3))
        self.btn_predict.clicked.connect(lambda: self.content_stack.setCurrentIndex(4))
        self.btn_report.clicked.connect(lambda: self.content_stack.setCurrentIndex(5))
        self.btn_home.setChecked(True)

    def build_top_navigation(self):
        self.nav_bar = QFrame()
        self.nav_bar.setObjectName("topNavBar")
        self.nav_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.nav_bar.setFixedHeight(112)
        self.main_layout.addWidget(self.nav_bar)

        nav_container = QVBoxLayout(self.nav_bar)
        nav_container.setContentsMargins(14, 8, 14, 8)
        nav_container.setSpacing(6)

        title_label = QLabel("阿尔茨海默症  智能预测系统 v2.0")
        title_label.setObjectName("appTitle")
        nav_container.addWidget(title_label)
        nav_container.setAlignment(title_label, Qt.AlignLeft)

        self.nav_buttons_host = QWidget()
        self.nav_buttons_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.nav_layout = QHBoxLayout(self.nav_buttons_host)
        self.nav_layout.setContentsMargins(0, 0, 0, 0)
        self.nav_layout.setSpacing(8)
        nav_container.addWidget(self.nav_buttons_host)
        nav_container.setAlignment(self.nav_buttons_host, Qt.AlignVCenter)

    def build_home_page(self):
        page_home = HomePageWidget(get_resource_path("assets", "blackground.png"))

        root = QVBoxLayout(page_home)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)
        root.addLayout(content_layout)

        hero_panel = QFrame()
        hero_panel.setObjectName("heroPanel")
        content_layout.addWidget(hero_panel, 12)

        hero_glow = QGraphicsDropShadowEffect(hero_panel)
        hero_glow.setColor(QColor(24, 132, 243, 110))
        hero_glow.setBlurRadius(42)
        hero_glow.setOffset(0, 0)
        hero_panel.setGraphicsEffect(hero_glow)

        hero_layout = QVBoxLayout(hero_panel)
        hero_layout.setContentsMargins(26, 24, 26, 20)
        hero_layout.setSpacing(50)

        hero_tag = QLabel("基于深度学习的辅助诊断系统")
        hero_tag.setObjectName("heroTag")

        hero_title = QLabel(
            "阿尔茨海默症 <span id='heroTitleAccent'></span><br/>智能预测系统"
        )
        hero_title.setObjectName("heroTitle")
        hero_title.setTextFormat(Qt.RichText)

        # hero_desc = QLabel(
        #     "本系统采用 SA-STGCN 深度学习模型，\n"
        #     "通过分析静息态 fMRI 数据，辅助诊断阿尔茨海默症。"
        # )
        # hero_desc.setObjectName("heroDescription")
        # hero_desc.setWordWrap(True)

        start_button = QPushButton("请选择功能模块开始分析")
        start_button.setObjectName("startButton")
        start_button.setCursor(Qt.PointingHandCursor)
        start_button.clicked.connect(lambda: self.btn_data.click())

        hero_layout.addWidget(hero_tag)
        hero_layout.addWidget(hero_title)
        # hero_layout.addWidget(hero_desc)
        hero_layout.addWidget(start_button, alignment=Qt.AlignLeft)
        hero_layout.addStretch(1)

        version = QLabel("版本 v2.0 | 仅供科研参考")
        version.setObjectName("versionText")
        hero_layout.addWidget(version)

        card_column = QVBoxLayout()
        card_column.setSpacing(12)
        content_layout.addLayout(card_column, 9)

        card_column.addWidget(HomeImageCard(
            title="fMRI 脑影像",
            image_path=get_resource_path("assets", "fmri_preview.png"),
            desc1="静息态 fMRI 脑影像预览",
            desc2="展示脑区功能活动分布特征",
        ))
        card_column.addWidget(HomeImageCard(
            title="脑功能连接网络",
            image_path=get_resource_path("assets", "connectome_preview.png"),
            desc1="脑区功能连接网络图",
            desc2="揭示脑区间连接模式与拓扑特征",
        ))
        card_column.addWidget(HomeImageCard(
            title="智能辅助分析",
            image_path=get_resource_path("assets", "ai_analysis_preview.png"),
            desc1="AI 模型预测结果与关键脑区分析",
            desc2="提供可解释的辅助诊断信息", 
        ))

        return page_home

    def init_pages(self):
        self.content_stack.addWidget(self.build_home_page())

        # 数据加载页
        self.page_data = DataManagerPage()
        self.content_stack.addWidget(self.page_data)

        # 预处理
        self.page_preproc = PreprocessingPage()
        self.content_stack.addWidget(self.page_preproc)

        # 特征提取
        self.feature_page = FeatureExtractionPage()
        self.content_stack.addWidget(self.feature_page)

        # 智能预测
        self.intelligent_prediction = DiagnosisPage(main_window=self)
        self.content_stack.addWidget(self.intelligent_prediction)

        # 个体报告分析
        self.individual_report_page = IndividualReportPage(main_window=self)
        self.content_stack.addWidget(self.individual_report_page)

        # AI 诊断助手 —— 悬浮球（全局可呼出）
        self.floating_ball = FloatingBall(self.main_widget, main_window=self)
        self.floating_ball.show()

    def resizeEvent(self, event):
        """窗口缩放时更新悬浮球位置"""
        super().resizeEvent(event)
        if hasattr(self, 'floating_ball'):
            self.floating_ball.update_position()

    # 接收诊断结果并跳转
    def open_individual_report(self, case_data):
        self.shared_case_data = case_data
        self.individual_report_page.load_case(case_data)
        # 同时把诊断数据推给 AI 助手，方便追问
        self.floating_ball.push_case_data(case_data)
        self.content_stack.setCurrentWidget(self.individual_report_page)
        self.btn_report.setChecked(True)


class DataManagerPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化后端读写工具
        self.reJSON = R_W_JSON()

        # 1. 必须先调用 setup_ui 创建所有控件实例 (解决 AttributeError)
        self.setup_ui()

        # 2. 绑定信号 (确保 self.history_table 等已存在)
        self.history_table.itemClicked.connect(self.on_table_click)
        self.btn_import.clicked.connect(self.import_new_data)

        # 3. 最后从 JSON 加载历史数据并填充表格
        self.refresh_history_from_file()

    def setup_ui(self):
        """实现上下结构的布局设计"""
        self.main_v_layout = QVBoxLayout(self)
        self.main_v_layout.setSpacing(10)

        # ================= 上半部分：历史记录表格 =================
        self.history_group = QGroupBox("历史扫描记录")
        history_v_layout = QVBoxLayout(self.history_group)

        # 定义表格：5列 (增加一列删除按钮)
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(["录入日期", "患者ID", "文件名", "状态", "操作"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)  # 整行选中
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 不允许直接编辑内容

        history_v_layout.addWidget(self.history_table)
        self.main_v_layout.addWidget(self.history_group, 2)  # 占据上半部分，权重设为 2

        # ================= 下半部分：左影像预览 + 右参数详情 =================
        self.bottom_h_layout = QHBoxLayout()

        # 左下：交互式 Viewer
        self.viewer_group = QGroupBox("影像交互预览")
        viewer_v_layout = QVBoxLayout(self.viewer_group)
        self.viewer = fMRIViewWidget()
        viewer_v_layout.addWidget(self.viewer)
        self.bottom_h_layout.addWidget(self.viewer_group, 3)  # Viewer 稍宽

        self.viewer_group.setStyleSheet("QGroupBox { color: white; }")

        # 右下：参数面板
        self.detail_group = QGroupBox("数据参数详情")
        self.detail_f_layout = QFormLayout(self.detail_group)

        self.param_labels = {
            "文件名": QLabel("未加载"),
            "维度": QLabel("-"),
            "体素大小": QLabel("-"),
            "TR": QLabel("-"),
            "时间点": QLabel("-")
        }
        

        self.detail_group.setStyleSheet("""
            QGroupBox {
                color: white;
            }
            QLabel {
                color: white;
            }
        """)


        # 设置标签样式并添加到布局
        for key, label in self.param_labels.items():
            label.setStyleSheet("color: #2980B9; font-weight: bold;")
            self.detail_f_layout.addRow(f"{key}:", label)

        # 导入按钮
        self.btn_import = QPushButton("➕ 导入新 fMRI 数据")
        self.btn_import.setFixedHeight(45)
        self.btn_import.setStyleSheet("""
            QPushButton {
                background-color: #3498DB; 
                color: white; 
                border-radius: 5px; 
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2980B9; }
        """)
        self.detail_f_layout.addRow(QLabel(""))  # 留白
        self.detail_f_layout.addRow(self.btn_import)

        self.bottom_h_layout.addWidget(self.detail_group, 1)  # 参数面板窄一些
        self.main_v_layout.addLayout(self.bottom_h_layout, 3)  # 下半部分权重

    def refresh_history_from_file(self):
        """从 JSON 加载并填充表格 (处理 JSON 为空的情况)"""
        try:
            self.history_table.setRowCount(0)
            history_data = self.reJSON.load_all_history()

            if not history_data:
                print("暂无历史记录。")
                return

            for item in history_data:
                path = item.get('file_path') or ""
                params = item.get('params', {})
                fname = params.get('FileName', os.path.basename(path) if path else "未知文件")

                self.add_table_row(
                    item.get('date', '-'),
                    item.get('patient_id', '-'),
                    fname,
                    path
                )
        except Exception as e:
            print(f"初始化历史记录失败: {e}")

    def add_table_row(self, date, pid, fname, full_path):
        """在表格中添加一行记录"""
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)

        date_item = QTableWidgetItem(date)
        date_item.setToolTip(date)
        pid_item = QTableWidgetItem(pid)
        pid_item.setToolTip(pid)

        self.history_table.setItem(row, 0, date_item)
        self.history_table.setItem(row, 1, pid_item)

        # 关键：将全路径隐藏存入 UserRole，以便点击时通过 data() 获取
        name_item = QTableWidgetItem(fname)
        name_item.setData(Qt.UserRole, full_path)
        name_item.setToolTip(fname)
        self.history_table.setItem(row, 2, name_item)

        self.history_table.setItem(row, 3, QTableWidgetItem("已就绪"))

        # 增加一列删除按钮
        del_btn = QPushButton("移除")
        del_btn.setStyleSheet("background-color: #409EFF; color: white; padding: 5px; border-radius: 3px;")
        # 通过 lambda 传递 path 确保删除正确
        del_btn.clicked.connect(lambda: self.handle_delete_row(full_path))
        self.history_table.setCellWidget(row, 4, del_btn)

    def import_new_data(self):
        """导入数据核心流程：选择文件 -> 提取 -> 存JSON -> 更新UI"""
        path, _ = QFileDialog.getOpenFileName(self, "导入 NIfTI 文件", "", "NIfTI (*.nii *.nii.gz)")
        if not path:
            return

        try:
            # 1. 提取参数 (使用 nibabel)
            img = nib.load(path)
            header = img.header
            dims = header.get_data_shape()
            zooms = header.get_zooms()

            info = {
                "FileName": os.path.basename(path),
                "Dimensions": f"{dims[0]}x{dims[1]}x{dims[2]}",
                "Voxel Size": f"{zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f} mm",
                "TR": f"{zooms[3]:.2f} s" if len(zooms) > 3 else "N/A",
                "Volumes": str(dims[3]) if len(dims) > 3 else "1"
            }

            # 2. 整理数据结构
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            patient_id = f"PAT_{datetime.now().strftime('%m%d%S')}"

            new_entry = {
                "date": date_str,
                "patient_id": patient_id,
                "file_path": path,
                "params": info
            }

            # 3. 写入 JSON 持久化 (确保 Read_Write_JSON 中有 save_to_history)
            self.reJSON.save_to_history(new_entry)

            # 4. 实时更新界面
            self.add_table_row(date_str, patient_id, info["FileName"], path)
            self.viewer.display_nii(path)
            self.update_param_panel(info)

            QMessageBox.information(self, "成功", f"文件 {info['FileName']} 已导入并记录。")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法解析或保存影像文件:\n{str(e)}")

    def on_table_click(self, item):
        """点击表格行：触发重复加载逻辑"""
        row = item.row()
        path_item = self.history_table.item(row, 2)
        full_path = path_item.data(Qt.UserRole)  # 读取之前存的隐藏路径
        self.current_viewing_path = full_path
        self.viewer.display_nii(full_path)

        if full_path and os.path.exists(full_path):
            # 重新渲染 Viewer
            self.viewer.display_nii(full_path)
            # 重新提取并更新参数面板
            try:
                img = nib.load(full_path)
                header = img.header
                dims = header.get_data_shape()
                zooms = header.get_zooms()
                info = {
                    "FileName": os.path.basename(full_path),
                    "Dimensions": f"{dims[0]}x{dims[1]}x{dims[2]}",
                    "Voxel Size": f"{zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f} mm",
                    "TR": f"{zooms[3]:.2f} s" if len(zooms) > 3 else "N/A",
                    "Volumes": str(dims[3]) if len(dims) > 3 else "1"
                }
                self.update_param_panel(info)
            except:
                pass
        else:
            QMessageBox.warning(self, "数据缺失", "找不到原始 NIfTI 文件，请确认文件路径未被移动。")

    def update_param_panel(self, info):
        """刷新右下角参数详情内容"""
        fname = info["FileName"]

        # 1. 设置 ToolTip，这样鼠标指上去能看到完整长文件名
        self.param_labels["文件名"].setToolTip(fname)

        # 2. 如果名字太长，进行截断处理（显示前15个和后10个字符）
        if len(fname) > 25:
            display_name = fname[:15] + "..." + fname[-10:]
        else:
            display_name = fname

        self.param_labels["文件名"].setText(display_name)
        self.param_labels["维度"].setText(info["Dimensions"])
        self.param_labels["体素大小"].setText(info["Voxel Size"])
        self.param_labels["TR"].setText(info["TR"])
        self.param_labels["时间点"].setText(info["Volumes"])

    # 项目.py (DataManagerPage 类内)

    def handle_delete_row(self, path):
        reply = QMessageBox.question(self, '确认删除', '确定要移除此项扫描记录吗？',
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 1. 从 JSON 删除
            if self.reJSON.delete_entry_by_path(path):
                # 2. 从表格 UI 删除
                for i in range(self.history_table.rowCount()):
                    item_path = self.history_table.item(i, 2).data(Qt.UserRole)
                    if os.path.normpath(item_path) == os.path.normpath(path):
                        self.history_table.removeRow(i)
                        break

                # 3. 如果删除的是当前正在显示的图像，则清空预览
                # 假设我们记录了当前正在预览的路径 self.current_viewing_path
                if hasattr(self, 'current_viewing_path') and self.current_viewing_path == path:
                    self.viewer.clear_view()  # 调用 viewer 的清空
                    self.reset_param_panel()  # 调用参数面板的重置
                    self.current_viewing_path = None
            else:
                QMessageBox.warning(self, "错误", "数据库同步失败")

    def reset_param_panel(self):
        """重置右侧参数面板的文字"""
        for label in self.param_labels.values():
            label.setText("-")
        self.param_labels["文件名"].setText("未载入数据")



class _HideToTrayFilter(QObject):
    """拦截窗口关闭事件：点 X 只隐藏到托盘，托盘菜单「退出」才真正结束程序。"""

    def __init__(self, state: dict, tray, parent=None):
        super().__init__(parent)
        self._state = state
        self._tray = tray
        self._hinted = False

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Close and not self._state.get("allow_close"):
            obj.hide()
            event.accept()
            if not self._hinted:
                self._tray.showMessage(
                    "智影识微",
                    "程序已最小化到托盘：双击托盘图标再次打开，右键托盘图标可完全退出。",
                    QSystemTrayIcon.Information, 4000)
                self._hinted = True
            return True
        return False


def _register_hide_to_tray(state: dict, tray, win):
    """给窗口挂「关闭即隐藏」过滤器；过滤器作为窗口子对象，随窗口一起销毁。"""
    win.installEventFilter(_HideToTrayFilter(state, tray, win))


def _app_tray_icon() -> QIcon:
    for name in ("new_icon.png", "login.png", "blackground.png"):
        path = get_resource_path("assets", name)
        if os.path.exists(path):
            return QIcon(path)
    return QIcon()


def _launch_app():
    # QtWebEngineWidgets 必须在 QApplication 创建之前导入，否则患者端登录后会卡死
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    from patient_window import PatientWindow  # noqa: F401 提前导入
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setFont(get_preferred_ui_font(10))
    apply_dark_dialog_style(app)
    # 窗口标题栏 / 任务栏图标与 exe 图标保持一致
    _app_icon_path = get_resource_path("assets", "new_icon.png")
    if os.path.exists(_app_icon_path):
        app.setWindowIcon(QIcon(_app_icon_path))

    state = {"windows": [], "login": None, "allow_close": False}

    # ---------- 系统托盘：点 X 留后台，托盘图标可恢复/完全退出 ----------
    tray = QSystemTrayIcon(_app_tray_icon(), app)
    tray.setToolTip("智影识微")

    def _restore_windows():
        """把最近使用的窗口恢复到前台（登录窗优先于主窗口）。"""
        wins = ([state["login"]] if state["login"] is not None else []) + list(state["windows"])
        if not wins:
            return
        w = wins[-1]
        w.setWindowState(w.windowState() & ~Qt.WindowMinimized)
        w.show()
        w.raise_()
        w.activateWindow()

    tray_menu = QMenu()
    act_open = tray_menu.addAction("打开 智影识微")
    tray_menu.addSeparator()
    act_quit = tray_menu.addAction("退出")
    tray.setContextMenu(tray_menu)
    act_open.triggered.connect(_restore_windows)

    def _quit_app():
        state["allow_close"] = True   # 托盘「退出」走真关闭
        for w in [state["login"], *state["windows"]]:
            try:
                if w is not None:
                    w.close()
            except Exception:
                pass
        app.quit()

    act_quit.triggered.connect(_quit_app)

    def _on_tray_activated(reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            _restore_windows()

    tray.activated.connect(_on_tray_activated)
    tray.show()

    # ---------- 单实例：已有实例通过本地通道发来 "show" 时恢复窗口 ----------
    if _SI_SERVER is not None:
        def _on_si_conn():
            conn = _SI_SERVER.nextPendingConnection()
            if conn is None:
                return
            conn.readyRead.connect(lambda: (conn.readAll(), _restore_windows()))
            conn.disconnected.connect(conn.deleteLater)
        _SI_SERVER.newConnection.connect(_on_si_conn)

    def _do_open(role, username, display_name):
        try:
            if role == "doctor":
                print("[launch] building ADDiagnosisSystem ...", flush=True)
                w = ADDiagnosisSystem()
            else:
                print("[launch] building PatientWindow ...", flush=True)
                w = PatientWindow(username=username, display_name=display_name, on_logout=_back_to_login)
            print(f"[launch] built {w.__class__.__name__}", flush=True)
            # 先把登录窗隐藏（避免窗口数瞬时为 0），再 show 新窗
            if state["login"] is not None:
                state["login"].hide()
            if hasattr(w, "attach_account_action"):
                w.attach_account_action(username=username, display_name=display_name, on_logout=_back_to_login)
            w.show()
            print(f"[launch] shown, visible={w.isVisible()}", flush=True)
            # 安全地销毁登录窗
            if state["login"] is not None:
                state["login"].deleteLater()
                state["login"] = None
            state["windows"].append(w)
            _register_hide_to_tray(state, tray, w)
        except Exception:
            import traceback
            print("[launch] EXCEPTION:", flush=True)
            traceback.print_exc()

    def on_login_success(role, username, display_name):
        print(f"[launch] on_login_success role={role} user={username}", flush=True)
        # 直接同步构造、show，最后才销毁登录窗。
        # 这样窗口数从 1（登录）→ 1（登录+新窗）→ 1（新窗），永远不为 0。
        _do_open(role, username, display_name)

    def _back_to_login():
        state["allow_close"] = True   # 退出登录是真关闭旧窗口，不要走「隐藏到托盘」
        for w in list(state["windows"]):
            try:
                w.close()
            except Exception:
                pass
        state["allow_close"] = False
        state["windows"].clear()
        from login_window import LoginWindow
        state["login"] = LoginWindow(on_login_success=on_login_success)
        state["login"].show()
        _register_hide_to_tray(state, tray, state["login"])

    from login_window import LoginWindow
    state["login"] = LoginWindow(on_login_success=on_login_success)
    state["login"].show()
    _register_hide_to_tray(state, tray, state["login"])
    return app


if __name__ == "__main__":
    app = _launch_app()
    if app is not None:
        sys.exit(app.exec())
    sys.exit(0)   # 已有实例在运行（已通知它恢复窗口）
