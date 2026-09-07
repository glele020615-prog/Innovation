import os
import sys
import traceback

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal

from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QProgressBar,
    QGroupBox, QLabel, QLineEdit, QComboBox,
    QFormLayout, QTextEdit, QSpinBox
)

# 项目根目录（用于定位本地 AAL116 图谱资源）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 特征提取的真实实现在 agent/skills/feature_extract.py（无 Qt 依赖），
# 本页的 QThread 只负责把日志/进度转成信号，保证 Agent 侧能复用同一份逻辑。
from agent.skills.feature_extract import (
    run_feature_extraction, locate_atlas, local_atlas_candidates,
)


# ==================== 特征提取后台线程 ====================
class FeatureExtractThread(QThread):
    """
    后台执行特征提取：
      1. 定位 AAL116 图谱（优先 nilearn 内置，回退本地重采样文件）
      2. 用 NiftiLabelsMasker 提取 116 个脑区的 BOLD 时间序列
      3. 用皮尔逊相关系数构建 116x116 功能连接 (FC) 矩阵
      4. 输出 {subject}_bold.npy 与 {subject}_fc.npy
    """
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(bool, str, str, str)  # (success, msg, bold_path, fc_path)

    def __init__(self, bold_path, output_dir, subject_id, atlas_source="auto",
                 standardize=True, parent=None):
        super().__init__(parent)
        self.bold_path = bold_path
        self.output_dir = output_dir
        self.subject_id = subject_id
        self.atlas_source = atlas_source  # "auto" / "nilearn" / "local"
        self.standardize = standardize

    def _local_atlas_candidates(self):
        """委托给 agent.skills.feature_extract（保持旧接口）"""
        return local_atlas_candidates()

    def _locate_atlas(self):
        """委托给 agent.skills.feature_extract（保持旧接口）"""
        return locate_atlas("AAL116", self.atlas_source, log=self.log_signal.emit)

    def run(self):
        """只负责信号转发：真实逻辑在 agent/skills/feature_extract.py，
        这样 Agent 在非 Qt 线程里也能复用同一份实现。"""
        result = run_feature_extraction(
            bold_path=self.bold_path,
            output_dir=self.output_dir,
            subject_id=self.subject_id,
            atlas="AAL116",
            standardize=self.standardize,
            atlas_source=self.atlas_source,
            log=self.log_signal.emit,
            progress=self.progress_signal.emit,
        )
        if result.get("success"):
            self.finished_signal.emit(
                True, f"{self.subject_id} 特征提取完成",
                result["bold_npy"], result["fc_npy"],
            )
        else:
            self.log_signal.emit(f"❌ 提取失败: {result.get('error', '未知错误')}")
            self.finished_signal.emit(False, result.get("error", "提取失败"), "", "")


class FeatureExtractionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None            # 当前运行的 FeatureExtractThread
        self._batch_queue = []         # 批量提取的 (row, subject_id, bold_input) 队列
        self._batch_total = 0          # 批量任务总数
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()

        # ================= 主功能页 =================
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)

        # ---------- 参数配置区 ----------
        config_group = QGroupBox("特征提取参数配置")
        config_layout = QFormLayout(config_group)

        self.fmriprep_dir = QLineEdit()
        self.fmriprep_dir.setPlaceholderText("选择 fMRIPrep 输出目录")
        btn_fmriprep = QPushButton("浏览")
        btn_fmriprep.clicked.connect(lambda: self.select_directory(self.fmriprep_dir))
        fmriprep_layout = QHBoxLayout()
        fmriprep_layout.addWidget(self.fmriprep_dir)
        fmriprep_layout.addWidget(btn_fmriprep)
        config_layout.addRow("fMRIPrep输出目录:", fmriprep_layout)

        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("选择特征输出目录")
        btn_output = QPushButton("浏览")
        btn_output.clicked.connect(lambda: self.select_directory(self.output_dir))
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(btn_output)
        config_layout.addRow("特征输出目录:", output_layout)

        self.subject_id = QLineEdit()
        self.subject_id.setPlaceholderText("例如: sub-01；留空则批量扫描")
        config_layout.addRow("被试ID:", self.subject_id)

        self.space_combo = QComboBox()
        self.space_combo.addItems([
            "MNI152NLin2009cAsym",
            "MNI152NLin6Asym"
        ])
        config_layout.addRow("标准空间:", self.space_combo)

        self.atlas_combo = QComboBox()
        self.atlas_combo.addItems(["AAL116"])
        config_layout.addRow("脑图谱:", self.atlas_combo)

        self.tr_spin = QSpinBox()
        self.tr_spin.setRange(1, 10)
        self.tr_spin.setValue(2)
        config_layout.addRow("TR (秒):", self.tr_spin)

        main_layout.addWidget(config_group)

        # ---------- 顶部按钮区 ----------
        btn_layout = QHBoxLayout()

        self.btn_scan_subjects = QPushButton("从fMRIPrep扫描被试")
        self.btn_add_subject = QPushButton("手动添加被试")
        self.btn_extract_one = QPushButton("提取当前被试")
        self.btn_extract_all = QPushButton("批量提取")
        self.btn_open_output = QPushButton("打开输出目录")

        btn_layout.addWidget(self.btn_scan_subjects)
        btn_layout.addWidget(self.btn_add_subject)
        btn_layout.addWidget(self.btn_extract_one)
        btn_layout.addWidget(self.btn_extract_all)
        btn_layout.addWidget(self.btn_open_output)

        main_layout.addLayout(btn_layout)

        # ---------- 任务表格 ----------
        self.task_table = QTableWidget(0, 5)
        self.task_table.setHorizontalHeaderLabels(["被试ID", "BOLD文件", "FC文件", "状态", "操作"])
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.task_table.setSelectionMode(QTableWidget.SingleSelection)
        self.task_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.task_table.setAlternatingRowColors(True)
        self.task_table.setShowGrid(False)

        header = self.task_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Fixed)

        self.task_table.setColumnWidth(3, 100)
        self.task_table.setColumnWidth(4, 120)

        self.task_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #DCDCDC;
                border-radius: 8px;
                background: white;
                alternate-background-color: #F7F9FC;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #F0F2F5;
                border: none;
                border-bottom: 1px solid #DCDCDC;
                padding: 6px;
                font-weight: 600;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QPushButton {
                padding: 4px 10px;
                border-radius: 5px;
                background-color: #409EFF;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #66B1FF;
            }
        """)

        main_layout.addWidget(self.task_table)

        # ---------- 进度条 ----------
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.tab_widget.addTab(main_tab, "🔬 特征提取")

        # ================= 日志页 =================
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        self.log_text.setPlaceholderText("特征提取日志将在这里显示...")
        log_layout.addWidget(self.log_text)

        btn_clear_log = QPushButton("清空日志")
        btn_clear_log.clicked.connect(self.log_text.clear)
        log_layout.addWidget(btn_clear_log)

        self.tab_widget.addTab(log_tab, "📄 提取日志")

        layout.addWidget(self.tab_widget)

        # ---------- 页面总样式 ----------
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
                font-size: 14px;
            }
            QGroupBox {
                background: white;
                border: 1px solid #E4E7ED;
                border-radius: 10px;
                margin-top: 12px;
                font-weight: bold;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #409EFF;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #66B1FF;
            }
            QComboBox, QLineEdit, QSpinBox {
                padding: 6px 10px;
                border: 1px solid #DCDFE6;
                border-radius: 6px;
                background: white;
            }
            QLabel {
                color: #303133;
            }
            QTabWidget::pane {
                border: 1px solid #E4E7ED;
                border-radius: 10px;
                background: #F5F7FA;
            }
            QTabBar::tab {
                background: #F5F7FA;
                padding: 8px 16px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                font-weight: bold;
            }
        """)

        self.btn_scan_subjects.clicked.connect(self.scan_subjects_from_fmriprep)
        self.btn_add_subject.clicked.connect(self.add_subject_manually)
        self.btn_extract_one.clicked.connect(self.extract_current_subject)
        self.btn_extract_all.clicked.connect(self.extract_all_subjects)
        self.btn_open_output.clicked.connect(self.open_output_dir)


    def select_directory(self, line_edit):
        directory = QFileDialog.getExistingDirectory(self, "选择目录")
        if directory:
            line_edit.setText(directory)

    def append_log(self, message):
        self.log_text.append(message)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def add_task_row(self, subject_id, bold_file="待生成", fc_file="待生成", status="就绪"):
        row = self.find_row_by_subject(subject_id)
        if row != -1:
            self.update_task_row(row, bold_file, fc_file, status)
            return

        row = self.task_table.rowCount()
        self.task_table.insertRow(row)

        self.task_table.setItem(row, 0, QTableWidgetItem(subject_id))
        self.task_table.setItem(row, 1, QTableWidgetItem(bold_file))
        self.task_table.setItem(row, 2, QTableWidgetItem(fc_file))

        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignCenter)
        self.task_table.setItem(row, 3, status_item)
        self.set_status_style(row, status)

        remove_btn = QPushButton("移除")
        remove_btn.clicked.connect(lambda _, b=remove_btn: self.remove_task_row(b))
        self.task_table.setCellWidget(row, 4, remove_btn)

        self.task_table.setRowHeight(row, 42)

    def update_task_row(self, row, bold_file, fc_file, status):
        self.task_table.item(row, 1).setText(bold_file)
        self.task_table.item(row, 2).setText(fc_file)
        self.task_table.item(row, 3).setText(status)
        self.set_status_style(row, status)

    def remove_task_row(self, button):
        for row in range(self.task_table.rowCount()):
            if self.task_table.cellWidget(row, 4) == button:
                self.task_table.removeRow(row)
                break

    def find_row_by_subject(self, subject_id):
        for row in range(self.task_table.rowCount()):
            item = self.task_table.item(row, 0)
            if item and item.text() == subject_id:
                return row
        return -1

    def set_status_style(self, row, status):
        item = self.task_table.item(row, 3)
        if not item:
            return
        if status == "就绪":
            item.setForeground(Qt.darkBlue)
        elif status == "处理中":
            item.setForeground(Qt.darkYellow)
        elif status == "已完成":
            item.setForeground(Qt.darkGreen)
        elif status == "失败":
            item.setForeground(Qt.red)
        else:
            item.setForeground(Qt.black)

    def scan_subjects_from_fmriprep(self):
        root_dir = self.fmriprep_dir.text().strip()
        if not root_dir:
            QMessageBox.warning(self, "提示", "请先选择 fMRIPrep 输出目录")
            return

        if not os.path.exists(root_dir):
            QMessageBox.warning(self, "提示", "目录不存在")
            return

        added = 0
        missing = 0
        for name in sorted(os.listdir(root_dir)):
            sub_dir = os.path.join(root_dir, name)
            if not (name.startswith("sub-") and os.path.isdir(sub_dir)):
                continue
            bold_input = self._find_bold_input(sub_dir)
            if bold_input:
                self.add_task_row(name, bold_input, "待生成", "就绪")
                added += 1
            else:
                self.add_task_row(name, "未找到 BOLD", "待生成", "就绪")
                missing += 1

        self.append_log(f"已扫描目录: {root_dir}")
        self.append_log(f"共识别到 {added + missing} 个被试（{added} 个已定位 BOLD，{missing} 个未定位）")
        QMessageBox.information(self, "完成", f"已加入 {added + missing} 个被试")

    def _find_bold_input(self, subject_dir):
        """
        在被试目录下定位可用 BOLD 文件，优先顺序：
          1. fMRIPrep 产物 desc-preproc_bold.nii.gz（MNI 标准空间）
          2. 自处理的 bold.nii / bold.nii.gz
        返回文件绝对路径，找不到返回 None。
        """
        preferred_keywords = [
            ("desc-preproc_bold.nii.gz", 0),
            ("desc-preproc_bold.nii", 1),
            ("bold.nii.gz", 2),
            ("bold.nii", 3),
        ]
        hits = []
        for dirpath, _dirnames, filenames in os.walk(subject_dir):
            for fn in filenames:
                for kw, priority in preferred_keywords:
                    if fn == kw or fn.endswith(kw):
                        hits.append((priority, os.path.join(dirpath, fn)))
                        break
        if not hits:
            return None
        hits.sort(key=lambda x: x[0])
        return hits[0][1]

    def add_subject_manually(self):
        subject_id = self.subject_id.text().strip()
        if not subject_id:
            QMessageBox.warning(self, "提示", "请先输入被试ID")
            return

        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{subject_id}"

        self.add_task_row(subject_id, "待生成", "待生成", "就绪")
        self.append_log(f"已手动添加被试: {subject_id}")

    def extract_current_subject(self):
        row = self.task_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先在表格中选择一个被试")
            return
        if self._worker is not None:
            QMessageBox.warning(self, "提示", "已有提取任务正在运行，请稍候")
            return

        output_dir = self.output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先设置特征输出目录")
            return

        subject_id = self.task_table.item(row, 0).text()
        bold_input = self.task_table.item(row, 1).text()
        if not bold_input or bold_input in ("待生成", "未找到 BOLD"):
            QMessageBox.warning(self, "提示", f"被试 {subject_id} 未定位到 BOLD 文件，请先扫描或手动添加")
            return

        self._run_extract(row, subject_id, bold_input, output_dir)

    def extract_all_subjects(self):
        if self.task_table.rowCount() == 0:
            QMessageBox.warning(self, "提示", "请先添加被试")
            return
        if self._worker is not None:
            QMessageBox.warning(self, "提示", "已有提取任务正在运行，请稍候")
            return

        output_dir = self.output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先设置特征输出目录")
            return

        # 收集所有可提取的被试
        self._batch_queue = []
        for row in range(self.task_table.rowCount()):
            subject_id = self.task_table.item(row, 0).text()
            bold_input = self.task_table.item(row, 1).text()
            if not bold_input or bold_input in ("待生成", "未找到 BOLD"):
                self.update_task_row(row, bold_input or "未找到 BOLD", "待生成", "失败")
                self.append_log(f"跳过 {subject_id}: 未定位到 BOLD 文件")
                continue
            self._batch_queue.append((row, subject_id, bold_input))

        if not self._batch_queue:
            QMessageBox.warning(self, "提示", "没有可提取的被试（均未定位到 BOLD 文件）")
            return

        self._batch_total = len(self._batch_queue)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self._batch_total)
        self.progress_bar.setValue(0)
        self._process_next_batch(output_dir)

    def _process_next_batch(self, output_dir):
        """从队列中取出下一个被试执行提取"""
        if not self._batch_queue:
            # 全部完成
            self.append_log("全部被试特征提取完成")
            self.progress_bar.setVisible(False)
            QMessageBox.information(self, "完成", "全部被试特征提取完成")
            return

        row, subject_id, bold_input = self._batch_queue.pop(0)
        self._run_extract(row, subject_id, bold_input, output_dir, is_batch=True)

    def _run_extract(self, row, subject_id, bold_input, output_dir, is_batch=False):
        """启动单个被试的特征提取线程"""
        self.update_task_row(row, bold_input, "待生成", "处理中")
        self.append_log(f"开始提取: {subject_id}  <-  {bold_input}")

        self._worker = FeatureExtractThread(
            bold_path=bold_input,
            output_dir=output_dir,
            subject_id=subject_id,
            standardize=True,
        )
        self._worker.log_signal.connect(self.append_log)
        self._worker.finished_signal.connect(
            lambda success, msg, bold_out, fc_out, r=row, sid=subject_id, b=is_batch:
            self._on_extract_finished(success, msg, bold_out, fc_out, r, sid, b, output_dir)
        )
        self._worker.start()

    def _on_extract_finished(self, success, msg, bold_out, fc_out,
                             row, subject_id, is_batch, output_dir):
        if success:
            self.update_task_row(row, bold_out, fc_out, "已完成")
            self.append_log(f"✅ {subject_id} 提取完成")
        else:
            self.update_task_row(row, self.task_table.item(row, 1).text(),
                                 "待生成", "失败")
            self.append_log(f"❌ {subject_id} 提取失败: {msg}")

        self._worker = None

        if is_batch:
            # 更新批量进度
            done = self._batch_total - len(self._batch_queue)
            self.progress_bar.setValue(done)
            self._process_next_batch(output_dir)

    def open_output_dir(self):
        output_dir = self.output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "提示", "输出目录不存在")
            return

        os.startfile(output_dir)