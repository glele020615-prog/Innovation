import subprocess
import os

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QProgressBar
)

from Read_Write_JSON import R_W_JSON


class PreprocessingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reJSON = R_W_JSON()
        self.thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 顶部按钮区
        btn_layout = QHBoxLayout()
        self.btn_import_history = QPushButton("从数据加载页导入")
        self.btn_import_new = QPushButton("新数据导入")
        self.btn_dcm_convert = QPushButton("DICOM 转 NIfTI")

        btn_layout.addWidget(self.btn_import_history)
        btn_layout.addWidget(self.btn_import_new)
        btn_layout.addWidget(self.btn_dcm_convert)
        layout.addLayout(btn_layout)

        # 任务表格
        self.task_table = QTableWidget(0, 4)
        self.task_table.setHorizontalHeaderLabels(["文件名", "文件路径", "状态", "操作"])
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.task_table.setSelectionMode(QTableWidget.SingleSelection)
        self.task_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.task_table.setAlternatingRowColors(True)
        self.task_table.setShowGrid(False)

        header = self.task_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Fixed)

        self.task_table.setColumnWidth(2, 90)
        self.task_table.setColumnWidth(3, 150)

        # 简单美化
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

        layout.addWidget(self.task_table)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 信号绑定
        self.btn_import_history.clicked.connect(self.load_from_json)
        self.btn_import_new.clicked.connect(self.open_file)
        self.btn_dcm_convert.clicked.connect(self.start_conversion)

    def open_file(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择 NIfTI 文件",
            "",
            "Images (*.nii *.nii.gz)"
        )
        for file_path in files:
            self.add_task_row(file_path, "就绪")

    def add_task_row(self, file_path, status="就绪"):
        """统一添加任务；若已存在则不重复添加"""
        if not file_path:
            return

        # 去重：按文件路径判断
        exist_row = self.find_row_by_path(file_path)
        if exist_row != -1:
            self.update_task_status(exist_row, status)
            return

        row = self.task_table.rowCount()
        self.task_table.insertRow(row)

        file_name = os.path.basename(file_path)

        # 文件名
        name_item = QTableWidgetItem(file_name)
        name_item.setToolTip(file_name)
        name_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.task_table.setItem(row, 0, name_item)

        # 文件路径
        path_item = QTableWidgetItem(file_path)
        path_item.setToolTip(file_path)
        path_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.task_table.setItem(row, 1, path_item)

        # 状态
        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignCenter)
        self.task_table.setItem(row, 2, status_item)
        self.set_status_style(row, status)

        # 移除按钮
        remove_btn = QPushButton("移除")
        remove_btn.clicked.connect(lambda _, b=remove_btn: self.remove_task_row(b))
        self.task_table.setCellWidget(row, 3, remove_btn)

        self.task_table.setRowHeight(row, 42)

    def remove_task_row(self, button):
        """删除按钮所在行"""
        for row in range(self.task_table.rowCount()):
            if self.task_table.cellWidget(row, 3) == button:
                self.task_table.removeRow(row)
                break

    def find_row_by_path(self, file_path):
        """根据文件路径查找所在行，不存在返回 -1"""
        for row in range(self.task_table.rowCount()):
            item = self.task_table.item(row, 1)
            if item and item.text() == file_path:
                return row
        return -1

    def update_task_status(self, row, status):
        """更新指定行状态"""
        if row < 0 or row >= self.task_table.rowCount():
            return

        item = self.task_table.item(row, 2)
        if item is None:
            item = QTableWidgetItem(status)
            item.setTextAlignment(Qt.AlignCenter)
            self.task_table.setItem(row, 2, item)
        else:
            item.setText(status)

        self.set_status_style(row, status)

    def set_status_style(self, row, status):
        """按状态设置显示颜色"""
        item = self.task_table.item(row, 2)
        if item is None:
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

    def load_from_json(self):
        """从数据加载页历史记录中追加导入，不清空现有表格"""
        history_data = self.reJSON.load_all_history()

        if not history_data:
            QMessageBox.information(self, "提示", "没有找到历史数据。")
            return

        count_before = self.task_table.rowCount()

        for item in history_data:
            file_path = item.get("file_path", "")
            if file_path:
                self.add_task_row(file_path, "就绪")

        count_after = self.task_table.rowCount()
        added_count = count_after - count_before

        QMessageBox.information(self, "导入完成", f"已从历史记录中追加导入 {added_count} 条数据。")

    def start_conversion(self):
        """开始 DICOM 转 NIfTI"""
        dcm_dir = QFileDialog.getExistingDirectory(self, "选择 DICOM 文件夹")
        if not dcm_dir:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "选择 NIfTI 输出存放位置")
        if not out_dir:
            return

        # 先在表格里挂一个“处理中”的任务
        pseudo_output_path = os.path.join(out_dir, os.path.basename(dcm_dir) + ".nii.gz")
        self.add_task_row(pseudo_output_path, "处理中")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 忙碌状态
        self.btn_dcm_convert.setEnabled(False)
        self.btn_import_history.setEnabled(False)
        self.btn_import_new.setEnabled(False)

        self.thread = ConvertThread(dcm_dir, out_dir)
        self.thread.finished_status.connect(self.on_convert_finished)
        self.thread.start()

    def on_convert_finished(self, success, message, output_files):
        self.progress_bar.setVisible(False)
        self.btn_dcm_convert.setEnabled(True)
        self.btn_import_history.setEnabled(True)
        self.btn_import_new.setEnabled(True)

        # 更新转换结果
        if success:
            # 转换成功：把所有输出文件加入队列，并设为“已完成”
            added_count = 0
            for file_path in output_files:
                old_row = self.find_row_by_path(file_path)
                self.add_task_row(file_path, "已完成")
                if old_row == -1:
                    added_count += 1

            QMessageBox.information(
                self,
                "成功",
                f"{message}\n共生成 {len(output_files)} 个 NIfTI 文件，已加入任务队列。"
            )
        else:
            # 失败时，把表格里可能的处理中项目改成失败
            for row in range(self.task_table.rowCount()):
                status_item = self.task_table.item(row, 2)
                if status_item and status_item.text() == "处理中":
                    self.update_task_status(row, "失败")

            QMessageBox.critical(self, "错误", message)


class ConvertThread(QThread):
    finished_status = Signal(bool, str, list)

    def __init__(self, dicom_dir, output_dir):
        super().__init__()
        self.dicom_dir = dicom_dir
        self.output_dir = output_dir

    def run(self):
        try:
            cmd = [
                "dcm2niix",
                "-z", "y",
                "-f", "%f",
                "-o", self.output_dir,
                self.dicom_dir
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8"
            )

            if process.returncode == 0:
                # 扫描输出目录中新增的 nii / nii.gz 文件
                output_files = []
                for f in os.listdir(self.output_dir):
                    if f.endswith(".nii") or f.endswith(".nii.gz"):
                        output_files.append(os.path.join(self.output_dir, f))

                self.finished_status.emit(True, "转换完成！文件已存至输出目录。", output_files)
            else:
                self.finished_status.emit(False, f"转换失败：{process.stderr}", [])

        except Exception as e:
            self.finished_status.emit(False, f"发生异常：{str(e)}", [])
