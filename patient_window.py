"""
患者端主窗口：
- 我的报告（PDF 查看 / 导入）
- 认知训练（嵌入 F:\\game\\ 下 5 个 HTML 游戏）
- 训练记录（本地 SQLite 汇总）
"""
import os
import sys
import csv
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QSize, QStandardPaths, QTimer, QObject, Slot, Signal
from PySide6.QtGui import QFont, QFontDatabase, QDesktopServices, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGraphicsDropShadowEffect,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSplitter,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QAbstractItemView,
)

from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWebEngineWidgets import QWebEngineView
try:
    from PySide6.QtWebChannel import QWebChannel
except ImportError:
    QWebChannel = None

import auth
from floating_ball import FloatingBall


# ---------- 路径与样式工具 ----------

# 开发环境优先用项目内的 game\（与打包版同源，避免两份文件漂移），没有再回退 F:\game
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOCAL_GAME_DIR = Path(_PROJECT_ROOT) / "game"
GAME_DIR = _LOCAL_GAME_DIR if _LOCAL_GAME_DIR.is_dir() else Path(r"F:\game")


def _game_path(name: str) -> str:
    """患者端游戏根目录；打包后走 _MEIPASS。"""
    base = getattr(sys, "_MEIPASS", None)
    root = Path(base) / "game" if base else GAME_DIR
    return str(root / name)


def _reports_dir_for(username: str) -> Path:
    """每位患者一个子目录，方便管理与导出。"""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        appdata = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        root = Path(appdata) / "ZhyingShiWei" / "reports" / username
    else:
        root = Path(r"F:\Innovation\report_cache") / username
    root.mkdir(parents=True, exist_ok=True)
    return root


def _font(pt=10):
    fams = QFontDatabase.families()
    for f in ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Segoe UI"):
        if f in fams:
            return QFont(f, pt)
    return QFont("Microsoft YaHei UI", pt)


PATIENT_QSS = """
QMainWindow { background:#07152B; }
QWidget#sidebar {
    background-color:#04101F;
    border-right:1px solid rgba(82,179,241,0.15);
}
QLabel#appName {
    color:#FFFFFF; font-size:18px; font-weight:700; letter-spacing:2px;
}
QLabel#userTag {
    color:#9EC4F5; font-size:12px;
}
QPushButton#navBtn {
    background:transparent;
    color:#E0EDFF;
    border:none;
    border-left:3px solid transparent;
    padding:12px 18px;
    text-align:left;
    font-size:14px;
    min-height:44px;
}
QPushButton#navBtn:hover {
    background:rgba(20,80,150,0.25);
    color:#FFFFFF;
}
QPushButton#navBtn:checked {
    background:rgba(12,100,195,0.35);
    border-left:3px solid #5DB8FF;
    color:#FFFFFF;
    font-weight:700;
}
QPushButton#actionBtn {
    background-color:rgba(12,100,195,0.95);
    color:#FFFFFF;
    border:1px solid rgba(140,225,255,0.55);
    border-radius:6px;
    padding:8px 14px;
    font-size:13px;
    min-height:36px;
}
QPushButton#actionBtn:hover {
    background-color:rgba(20,120,225,1.0);
}
QPushButton#ghostBtn {
    background:transparent;
    color:#82B1E8;
    border:1px solid rgba(130,200,255,0.45);
    border-radius:6px;
    padding:8px 14px;
    font-size:13px;
    min-height:36px;
}
QPushButton#ghostBtn:hover { background:rgba(20,80,150,0.25); color:#FFFFFF; }
QLabel#pageTitle {
    color:#FFFFFF; font-size:22px; font-weight:700;
}
QLabel#pageSub {
    color:#82B1E8; font-size:12px;
}
QFrame#card {
    background:#0B1E3A;
    border:1px solid rgba(82,179,241,0.18);
    border-radius:10px;
}
QListWidget {
    background:#04101F;
    color:#E0EDFF;
    border:1px solid rgba(82,179,241,0.20);
    border-radius:8px;
    padding:4px;
    font-size:13px;
}
QListWidget::item { padding:10px 8px; border-bottom:1px solid rgba(82,179,241,0.10); }
QListWidget::item:selected {
    background:rgba(12,100,195,0.45);
    color:#FFFFFF;
}
QTableWidget {
    background:#04101F;
    color:#E0EDFF;
    gridline-color:rgba(82,179,241,0.18);
    font-size:12px;
    border:1px solid rgba(82,179,241,0.20);
    border-radius:8px;
}
QHeaderView::section {
    background:#0B1E3A;
    color:#9EC4F5;
    padding:8px;
    border:none;
    border-right:1px solid rgba(82,179,241,0.15);
}
QLabel#empty {
    color:#6F86A8; font-size:13px;
}
"""


# ---------- 数据访问（reports / training_logs） ----------

def _db_connect():
    conn = sqlite3.connect(auth.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def list_reports(username: str):
    with _db_connect() as conn:
        return [
            dict(r) for r in conn.execute(
                "SELECT id,title,path,uploaded_at FROM reports WHERE username=? ORDER BY id DESC",
                (username,),
            ).fetchall()
        ]


def add_report(username: str, src_path: str) -> int:
    src = Path(src_path)
    dst_dir = _reports_dir_for(username)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = dst_dir / f"{ts}_{src.name}"
    shutil.copy2(src, dst)
    with _db_connect() as conn:
        cur = conn.execute(
            "INSERT INTO reports(username,title,path,uploaded_at) VALUES(?,?,?,?)",
            (username, src.stem, str(dst), datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def delete_report(report_id: int, username: str):
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT path FROM reports WHERE id=? AND username=?", (report_id, username)
        ).fetchone()
        if row:
            try:
                Path(row["path"]).unlink(missing_ok=True)
            except Exception:
                pass
            conn.execute("DELETE FROM reports WHERE id=? AND username=?", (report_id, username))


def list_training(username: str, limit: int = 200):
    with _db_connect() as conn:
        return [
            dict(r) for r in conn.execute(
                "SELECT game,difficulty,total,correct,accuracy,avg_rt_ms,duration_s,played_at "
                "FROM training_logs WHERE username=? ORDER BY id DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        ]


def insert_training(username: str, rec: dict):
    with _db_connect() as conn:
        conn.execute(
            "INSERT INTO training_logs(username,game,difficulty,total,correct,accuracy,avg_rt_ms,duration_s,played_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                username,
                rec.get("game", ""),
                rec.get("difficulty", ""),
                int(rec.get("total", 0) or 0),
                int(rec.get("correct", 0) or 0),
                float(rec.get("accuracy", 0) or 0),
                int(rec.get("avg_rt_ms", 0) or 0),
                int(rec.get("duration_s", 0) or 0),
                rec.get("played_at", datetime.now().isoformat(timespec="seconds")),
            ),
        )


# ---------- 子页面 ----------

class ReportPage(QWidget):
    def __init__(self, username: str, parent=None):
        super().__init__(parent)
        self.username = username
        self._build()
        self.refresh()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(16)

        # 左：列表 + 按钮
        left = QFrame(); left.setObjectName("card")
        left_l = QVBoxLayout(left); left_l.setContentsMargins(14,14,14,14); left_l.setSpacing(10)

        title = QLabel("我的报告"); title.setObjectName("pageTitle")
        sub = QLabel("导入 PDF 后将在此查看"); sub.setObjectName("pageSub")
        left_l.addWidget(title); left_l.addWidget(sub)

        btn_row = QHBoxLayout()
        self.btn_import = QPushButton("导入 PDF"); self.btn_import.setObjectName("actionBtn")
        self.btn_open_dir = QPushButton("打开文件夹"); self.btn_open_dir.setObjectName("ghostBtn")
        self.btn_del = QPushButton("删除"); self.btn_del.setObjectName("ghostBtn")
        btn_row.addWidget(self.btn_import)
        btn_row.addWidget(self.btn_open_dir)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_del)
        left_l.addLayout(btn_row)

        self.list = QListWidget()
        left_l.addWidget(self.list, 1)

        # 右：QPdfView
        right = QFrame(); right.setObjectName("card")
        right_l = QVBoxLayout(right); right_l.setContentsMargins(4,4,4,4)
        self.pdf_view = QPdfView()
        self.pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self.pdf_doc = QPdfDocument(self)
        self.pdf_view.setDocument(self.pdf_doc)
        right_l.addWidget(self.pdf_view)

        lay.addWidget(left, 3)
        lay.addWidget(right, 7)

        # 信号
        self.btn_import.clicked.connect(self._on_import)
        self.btn_open_dir.clicked.connect(self._on_open_dir)
        self.btn_del.clicked.connect(self._on_delete)
        self.list.currentItemChanged.connect(self._on_select)

    def refresh(self):
        self.list.clear()
        self.reports = list_reports(self.username)
        for r in self.reports:
            item = QListWidgetItem(f"{r['uploaded_at']}  ·  {r['title']}")
            item.setData(Qt.UserRole, r)
            self.list.addItem(item)
        if self.reports:
            self.list.setCurrentRow(0)
        else:
            self.pdf_doc.close()

    def _on_select(self, cur, _prev):
        if not cur:
            return
        rec = cur.data(Qt.UserRole)
        self._load_pdf(rec["path"])

    def _load_pdf(self, path):
        self.pdf_doc.close()
        if self.pdf_doc.load(path) != QPdfDocument.Error.None_:
            self.pdf_doc.close()
            QMessageBox.warning(self, "提示", f"无法打开该 PDF:\n{path}")

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 PDF 报告", "", "PDF (*.pdf)")
        if not path:
            return
        try:
            add_report(self.username, path)
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
            return
        self.refresh()

    def _on_open_dir(self):
        d = _reports_dir_for(self.username)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _on_delete(self):
        cur = self.list.currentItem()
        if not cur:
            return
        rec = cur.data(Qt.UserRole)
        if QMessageBox.question(self, "确认", f"删除报告：{rec['title']}？") != QMessageBox.Yes:
            return
        delete_report(rec["id"], self.username)
        self.refresh()


class TrainingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._load_index()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(10)

        top = QHBoxLayout()
        title = QLabel("认知训练"); title.setObjectName("pageTitle")
        sub = QLabel("选择下方入口开始训练"); sub.setObjectName("pageSub")
        self.lbl_crum = QLabel("训练 / 首页"); self.lbl_crum.setStyleSheet("color:#9EC4F5; font-size:12px;")
        self.btn_home = QPushButton("← 返回训练列表"); self.btn_home.setObjectName("ghostBtn")
        self.btn_home.clicked.connect(self._load_index)
        top.addWidget(title); top.addSpacing(12); top.addWidget(sub)
        top.addStretch(1); top.addWidget(self.lbl_crum); top.addWidget(self.btn_home)
        lay.addLayout(top)

        self.web = QWebEngineView()
        # 通道：让 HTML 把训练结果回调到 Python
        self.channel = QWebChannel()
        self.bridge = _TrainingBridge()
        self.channel.registerObject("trainingBridge", self.bridge)
        self.web.page().setWebChannel(self.channel)
        self.bridge.on_record = self._on_record
        lay.addWidget(self.web, 1)

    def _load_index(self):
        self.lbl_crum.setText("训练 / 首页")
        self.web.setUrl(QUrl.fromLocalFile(_game_path("index.html")))

    def open_game(self, idx: int, label: str):
        self.lbl_crum.setText(f"训练 / 游戏 {idx}：{label}")
        self.web.setUrl(QUrl.fromLocalFile(_game_path(f"game{idx}.html")))

    def _on_record(self, rec: dict):
        # rec 形如 {game,difficulty,total,correct,accuracy,avg_rt_ms,duration_s}
        insert_training(current_username(), rec)


class _TrainingBridge(QObject):
    """供 HTML 端通过 QWebChannel 调用。"""

    on_record = None

    @Slot("QVariant")
    def saveRecord(self, rec):
        if self.on_record:
            self.on_record(dict(rec))


def current_username():
    """由 PatientWindow 在登录后注入；训练桥回调时用到。"""
    return _CURRENT_USERNAME[0]


_CURRENT_USERNAME = [""]


class HistoryPage(QWidget):
    def __init__(self, username: str, parent=None):
        super().__init__(parent)
        self.username = username
        self._build()
        self.refresh()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(10)

        top = QHBoxLayout()
        title = QLabel("训练记录"); title.setObjectName("pageTitle")
        sub = QLabel("本机保存，不上传网络"); sub.setObjectName("pageSub")
        self.btn_export = QPushButton("导出 CSV"); self.btn_export.setObjectName("actionBtn")
        self.btn_refresh = QPushButton("刷新"); self.btn_refresh.setObjectName("ghostBtn")
        top.addWidget(title); top.addSpacing(12); top.addWidget(sub)
        top.addStretch(1); top.addWidget(self.btn_export); top.addWidget(self.btn_refresh)
        lay.addLayout(top)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "时间", "游戏", "难度", "完成题数", "正确题数",
            "正确率", "平均反应(ms)", "训练时长(s)",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        lay.addWidget(self.table, 1)

        self.lbl_empty = QLabel("暂无训练记录。请到「认知训练」开始第一次训练。")
        self.lbl_empty.setObjectName("empty")
        self.lbl_empty.setAlignment(Qt.AlignCenter)
        self.lbl_empty.setVisible(False)
        lay.addWidget(self.lbl_empty)

        self.btn_export.clicked.connect(self._export_csv)
        self.btn_refresh.clicked.connect(self.refresh)

    def refresh(self):
        rows = list_training(self.username)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            acc = r["accuracy"]
            acc_text = f"{acc*100:.0f}%" if isinstance(acc, (int, float)) and acc <= 1 else f"{acc:.0f}%"
            values = [
                r["played_at"], r["game"], r["difficulty"] or "-",
                str(r["total"]), str(r["correct"]),
                acc_text, str(r["avg_rt_ms"]), str(r["duration_s"]),
            ]
            for j, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setForeground(QColor("#E0EDFF"))
                self.table.setItem(i, j, item)
        empty = len(rows) == 0
        self.lbl_empty.setVisible(empty)
        self.table.setVisible(not empty)

    def _export_csv(self):
        rows = list_training(self.username, limit=10000)
        if not rows:
            QMessageBox.information(self, "提示", "暂无训练记录")
            return
        default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV",
            os.path.join(default_dir or "", f"训练记录_{self.username}.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["时间", "游戏", "难度", "完成题数", "正确题数", "正确率", "平均反应(ms)", "训练时长(s)"])
            for r in rows:
                w.writerow([
                    r["played_at"], r["game"], r["difficulty"] or "",
                    r["total"], r["correct"], r["accuracy"],
                    r["avg_rt_ms"], r["duration_s"],
                ])
        QMessageBox.information(self, "完成", f"已导出到：\n{path}")


# ---------- 主窗口 ----------

GAMES = [
    (1, "记忆翻牌"),
    (2, "亮灯序列"),
    (3, "购物清单记忆"),
    (4, "日常生活步骤排序"),
    (5, "找目标"),
]


class PatientWindow(QMainWindow):
    """患者端主入口。on_logout 由外部注入（阶段 4 接登录窗）。"""

    def __init__(self, username: str, display_name: str, on_logout=None):
        super().__init__()
        self.username = username
        self.display_name = display_name or username
        self.on_logout = on_logout
        _CURRENT_USERNAME[0] = username

        self.setWindowTitle("智影识微 · 患者训练端")
        self.resize(1280, 800)
        self.setFont(_font(10))

        root = QWidget(); self.setCentralWidget(root)
        outer = QHBoxLayout(root); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        # 侧边栏
        side = QWidget(); side.setObjectName("sidebar"); side.setFixedWidth(220)
        side_l = QVBoxLayout(side); side_l.setContentsMargins(0,0,0,0); side_l.setSpacing(0)

        # 顶部：品牌 + 用户
        brand_box = QWidget(); brand_l = QVBoxLayout(brand_box)
        brand_l.setContentsMargins(18, 22, 18, 12); brand_l.setSpacing(4)
        app = QLabel("智影识微"); app.setObjectName("appName")
        tag = QLabel(f"患者：{self.display_name}"); tag.setObjectName("userTag")
        brand_l.addWidget(app); brand_l.addWidget(tag)
        side_l.addWidget(brand_box)

        # 导航按钮
        self.btn_nav_report = QPushButton("📄  我的报告")
        self.btn_nav_train  = QPushButton("🎯  认知训练")
        self.btn_nav_hist   = QPushButton("📊  训练记录")
        for b in (self.btn_nav_report, self.btn_nav_train, self.btn_nav_hist):
            b.setObjectName("navBtn")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            side_l.addWidget(b)

        side_l.addStretch(1)

        # 退出按钮
        self.btn_logout = QPushButton("⏏  退出登录")
        self.btn_logout.setObjectName("ghostBtn")
        self.btn_logout.setFixedHeight(38)
        side_l.addWidget(self.btn_logout)

        # 游戏子导航（在训练页下方显示）
        self.game_nav = QWidget()
        gn_l = QVBoxLayout(self.game_nav)
        gn_l.setContentsMargins(18, 6, 18, 18); gn_l.setSpacing(6)
        gn_title = QLabel("训练入口"); gn_title.setStyleSheet("color:#9EC4F5; font-size:12px;")
        gn_l.addWidget(gn_title)
        self._game_buttons = []
        for idx, label in GAMES:
            b = QPushButton(f"游戏 {idx}：{label}")
            b.setObjectName("ghostBtn")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=idx, l=label: self._open_game(i, l))
            gn_l.addWidget(b)
            self._game_buttons.append(b)
        self.game_nav.setVisible(False)
        side_l.addWidget(self.game_nav)

        outer.addWidget(side)

        # 内容区
        self.stack = QStackedWidget()
        self.page_report = ReportPage(self.username)
        self.page_train  = TrainingPage()
        self.page_hist   = HistoryPage(self.username)
        self.stack.addWidget(self.page_report)
        self.stack.addWidget(self.page_train)
        self.stack.addWidget(self.page_hist)
        outer.addWidget(self.stack, 1)

        # 信号
        self.btn_nav_report.clicked.connect(lambda: self._switch(0))
        self.btn_nav_train.clicked.connect(lambda: self._switch(1))
        self.btn_nav_hist.clicked.connect(lambda: self._switch(2))
        self.btn_logout.clicked.connect(self._do_logout)

        self.setStyleSheet(PATIENT_QSS)
        self._switch(0)

        # 患者端 AI 悬浮球：仅知识库问答，无工具调用
        self.floating_ball = FloatingBall(self, mode="patient")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "floating_ball"):
            self.floating_ball.update_position()

    def _switch(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self.btn_nav_report.setChecked(idx == 0)
        self.btn_nav_train.setChecked(idx == 1)
        self.btn_nav_hist.setChecked(idx == 2)
        self.game_nav.setVisible(idx == 1)
        if idx == 2:
            self.page_hist.refresh()

    def _open_game(self, idx: int, label: str):
        self._switch(1)
        self.page_train.open_game(idx, label)

    def _do_logout(self):
        if QMessageBox.question(self, "退出登录", "确定退出当前账号？") != QMessageBox.Yes:
            return
        if self.on_logout:
            self.on_logout()
        else:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(_font(10))
    # 自检：直接打开患者窗口（绕过登录）
    w = PatientWindow(username="patient", display_name="示例患者")
    w.show()
    sys.exit(app.exec())
