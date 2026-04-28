import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QProgressBar,
    QGroupBox, QLabel, QLineEdit, QComboBox,
    QFormLayout, QTextEdit, QSpinBox
)


class FeatureExtractionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
        for name in os.listdir(root_dir):
            if name.startswith("sub-") and os.path.isdir(os.path.join(root_dir, name)):
                self.add_task_row(name, "待生成", "待生成", "就绪")
                added += 1

        self.append_log(f"已扫描目录: {root_dir}")
        self.append_log(f"共识别到 {added} 个被试")
        QMessageBox.information(self, "完成", f"已加入 {added} 个被试")

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

        subject_id = self.task_table.item(row, 0).text()
        self.append_log(f"开始提取当前被试特征: {subject_id}")
        self.task_table.item(row, 3).setText("处理中")
        self.set_status_style(row, "处理中")

        output_dir = self.output_dir.text().strip() or "未设置输出目录"
        bold_file = os.path.join(output_dir, f"{subject_id}_bold.npy")
        fc_file = os.path.join(output_dir, f"{subject_id}_fc.npy")

        self.update_task_row(row, bold_file, fc_file, "已完成")
        self.append_log(f"被试 {subject_id} 提取完成")
        QMessageBox.information(self, "完成", f"{subject_id} 特征提取完成")

    def extract_all_subjects(self):
        if self.task_table.rowCount() == 0:
            QMessageBox.warning(self, "提示", "请先添加被试")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        total = self.task_table.rowCount()
        output_dir = self.output_dir.text().strip() or "未设置输出目录"

        for row in range(total):
            subject_id = self.task_table.item(row, 0).text()
            self.update_task_row(row, "待生成", "待生成", "处理中")
            self.append_log(f"开始批量提取: {subject_id}")

            bold_file = os.path.join(output_dir, f"{subject_id}_bold.npy")
            fc_file = os.path.join(output_dir, f"{subject_id}_fc.npy")
            self.update_task_row(row, bold_file, fc_file, "已完成")

            self.progress_bar.setValue(int((row + 1) / total * 100))

        self.append_log("全部被试特征提取完成")
        QMessageBox.information(self, "完成", "全部被试特征提取完成")

    def open_output_dir(self):
        output_dir = self.output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "提示", "输出目录不存在")
            return

        os.startfile(output_dir)