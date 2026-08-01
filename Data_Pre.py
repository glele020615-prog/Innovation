import subprocess
import os
import shutil
import json
import csv
import nibabel as nib
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QProgressBar,
    QGroupBox, QLabel, QLineEdit, QCheckBox, QSpinBox,
    QComboBox, QFormLayout, QTextEdit, QSplitter, QInputDialog, QDialog
)
from PySide6.QtGui import QFont, QTextCursor

from Read_Write_JSON import R_W_JSON


def write_nifti_gz(source_path, destination_path):
    """Read a NIfTI file and write it back as a real .nii.gz file."""
    image = nib.load(source_path)
    nib.save(image, destination_path)


# ==================== BIDS转换线程 ====================
class BIDSConvertThread(QThread):
    """BIDS转换线程"""
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None

    def run(self):
        try:
            self.log_signal.emit("=" * 60)
            self.log_signal.emit("启动 BIDS 转换...")
            self.log_signal.emit(f"命令: {' '.join(self.command)}")
            self.log_signal.emit("=" * 60)

            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.log_signal.emit(line.rstrip())

            self.process.wait()

            if self.process.returncode == 0:
                self.finished_signal.emit(True, "BIDS转换完成！")
            else:
                self.finished_signal.emit(False, f"转换失败，错误代码: {self.process.returncode}")

        except Exception as e:
            self.finished_signal.emit(False, f"执行错误: {str(e)}")

    def stop(self):
        if self.process:
            self.process.terminate()


# ==================== fMRIPrep 工作线程 ====================
class FMRIPrepThread(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(bool, str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process = None
        self._stop_requested = False

    def check_docker_available(self):
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=20
            )
            return result.returncode == 0, result.stderr.strip() or result.stdout.strip()
        except Exception as e:
            return False, str(e)

    def run(self):
        try:
            self.log_signal.emit("=" * 60)
            self.log_signal.emit("启动 fMRIPrep 预处理流程")
            self.log_signal.emit("=" * 60)

            if self.config.get('use_docker', False):
                available, detail = self.check_docker_available()
                if not available:
                    self.finished_signal.emit(
                        False,
                        f"Docker 不可用，请先启动 Docker Desktop 并确认引擎正常运行。详情: {detail}"
                    )
                    return

            cmd = self.build_command()
            self.log_signal.emit(f"执行命令: {' '.join(cmd)}\n")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            output_buffer = ""
            while True:
                char = self.process.stdout.read(1)

                if char == "" and self.process.poll() is not None:
                    if output_buffer.strip():
                        self.log_signal.emit(output_buffer.rstrip())
                    break

                if not char:
                    continue

                output_buffer += char

                if "Continue anyway? [y/N]" in output_buffer or "Continue anyway?" in output_buffer:
                    if self.process.stdin:
                        self.process.stdin.write("y\n")
                        self.process.stdin.flush()
                    output_buffer = output_buffer.replace("Continue anyway? [y/N]", "")
                    output_buffer = output_buffer.replace("Continue anyway?", "")

                if char == "\n":
                    cleaned_line = output_buffer.rstrip()
                    if cleaned_line:
                        self.log_signal.emit(cleaned_line)
                        if "Running subject" in cleaned_line:
                            self.progress_signal.emit(50)
                        elif "Finished" in cleaned_line:
                            self.progress_signal.emit(90)
                    output_buffer = ""

            self.process.wait()

            if self._stop_requested:
                self.finished_signal.emit(False, "预处理已被用户终止")
                return

            if self.process.returncode == 0:
                self.progress_signal.emit(100)
                self.finished_signal.emit(True, "fMRIPrep 预处理完成！")
            else:
                if self._stop_requested:
                    self.finished_signal.emit(False, "预处理已被用户终止")
                    return
                self.finished_signal.emit(False, f"预处理失败，错误代码: {self.process.returncode}")

        except Exception as e:
            self.finished_signal.emit(False, f"执行错误: {str(e)}")

    def build_command(self):
        cmd = []

        if self.config.get('use_docker', False):
            cmd.extend(['docker', 'run', '--rm'])

            cmd.extend([
                '-v', f"{self.config['bids_dir']}:/data:ro",
                '-v', f"{self.config['output_dir']}:/out"
            ])

            work_dir = self.config.get('work_dir')
            if work_dir:
                cmd.extend(['-v', f"{work_dir}:/work"])

            fs_license = self.config.get('fs_license')
            if fs_license:
                cmd.extend(['-v', f"{fs_license}:/opt/freesurfer/license.txt"])

            cmd.append('nipreps/fmriprep:25.2.5')
            cmd.extend(['/data', '/out', 'participant'])

            if work_dir:
                cmd.extend(['-w', '/work'])

            if fs_license:
                cmd.extend(['--fs-license-file', '/opt/freesurfer/license.txt'])
        else:
            cmd.extend(['apptainer', 'run', '--cleanenv'])
            cmd.extend(['-B', f"{self.config['bids_dir']}:/data:ro"])
            cmd.extend(['-B', f"{self.config['output_dir']}:/out"])
            cmd.extend(['-B', f"{self.config['work_dir']}:/work"])

            if self.config.get('fs_license'):
                cmd.extend(['-B', f"{self.config['fs_license']}:/opt/freesurfer/license.txt"])

            cmd.extend([self.config.get('container_path', '/containers/fmriprep.sif')])
            cmd.extend(['/data', '/out', 'participant'])
            cmd.extend(['-w', '/work'])

            if self.config.get('fs_license'):
                cmd.extend(['--fs-license-file', '/opt/freesurfer/license.txt'])

        if self.config.get('participant_label'):
            cmd.extend(['--participant-label', self.config['participant_label']])

        cmd.extend(['--nprocs', str(self.config.get('n_cpus', 8))])
        cmd.extend(['--mem', str(self.config.get('mem_mb', 32000))])

        if self.config.get('output_spaces'):
            cmd.extend(['--output-spaces'] + self.config['output_spaces'])

        return cmd

    def stop(self):
        if self.process:
            self._stop_requested = True
            try:
                if self.config.get('use_docker', False):
                    result = subprocess.run(
                        [
                            "docker", "ps",
                            "--filter", "ancestor=nipreps/fmriprep:25.2.5",
                            "--format", "{{.ID}}"
                        ],
                        capture_output=True,
                        text=True
                    )
                    container_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                    for container_id in container_ids:
                        subprocess.run(["docker", "stop", container_id], capture_output=True, text=True)

                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                    capture_output=True,
                    text=True
                )
            except Exception:
                self.process.terminate()
            self.log_signal.emit("预处理已被用户终止")


# ==================== fMRIPrep 配置界面 ====================
class FMRIPrepWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Vertical)

        # 配置区域
        config_widget = self.create_config_widget()
        splitter.addWidget(config_widget)

        # 日志区域
        log_widget = self.create_log_widget()
        splitter.addWidget(log_widget)

        splitter.setSizes([500, 300])
        layout.addWidget(splitter)

    def create_config_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("fMRIPrep 预处理配置")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)

        # BIDS 目录
        self.bids_dir = QLineEdit()
        self.bids_dir.setPlaceholderText("选择 BIDS 格式的数据目录")
        btn_bids = QPushButton("浏览")
        btn_bids.clicked.connect(lambda: self.select_directory(self.bids_dir))
        bids_layout = QHBoxLayout()
        bids_layout.addWidget(self.bids_dir)
        bids_layout.addWidget(btn_bids)
        form_layout.addRow("BIDS 数据集路径:", bids_layout)

        # 输出目录
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("预处理结果输出目录")
        btn_output = QPushButton("浏览")
        btn_output.clicked.connect(lambda: self.select_directory(self.output_dir))
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(btn_output)
        form_layout.addRow("输出目录:", output_layout)

        # 工作目录
        self.work_dir = QLineEdit()
        self.work_dir.setPlaceholderText("临时工作目录")
        btn_work = QPushButton("浏览")
        btn_work.clicked.connect(lambda: self.select_directory(self.work_dir))
        work_layout = QHBoxLayout()
        work_layout.addWidget(self.work_dir)
        work_layout.addWidget(btn_work)
        form_layout.addRow("临时工作目录:", work_layout)

        # FreeSurfer 许可证
        self.fs_license = QLineEdit()
        self.fs_license.setPlaceholderText("FreeSurfer license.txt 路径")
        btn_license = QPushButton("浏览")
        btn_license.clicked.connect(lambda: self.select_file(self.fs_license, "License文件 (*.txt)"))
        license_layout = QHBoxLayout()
        license_layout.addWidget(self.fs_license)
        license_layout.addWidget(btn_license)
        form_layout.addRow("FreeSurfer许可证:", license_layout)

        # 容器路径
        self.container_path = QLineEdit()
        self.container_path.setPlaceholderText("fmriprep.sif 文件路径")
        btn_container = QPushButton("浏览")
        btn_container.clicked.connect(lambda: self.select_file(self.container_path, "Singularity文件 (*.sif)"))
        container_layout = QHBoxLayout()
        container_layout.addWidget(self.container_path)
        container_layout.addWidget(btn_container)
        form_layout.addRow("容器路径:", container_layout)

        layout.addWidget(form_widget)

        # 处理参数
        param_group = QGroupBox("处理参数")
        param_layout = QFormLayout(param_group)

        self.participant_label = QLineEdit()
        self.participant_label.setPlaceholderText("例如: sub-01 sub-02 (留空处理全部)")
        param_layout.addRow("被试标签:", self.participant_label)

        self.n_cpus = QSpinBox()
        self.n_cpus.setRange(1, 64)
        self.n_cpus.setValue(8)
        param_layout.addRow("CPU 核心数:", self.n_cpus)

        self.mem_mb = QSpinBox()
        self.mem_mb.setRange(4096, 128000)
        self.mem_mb.setValue(32000)
        param_layout.addRow("内存限制 (MB):", self.mem_mb)

        self.output_spaces = QComboBox()
        self.output_spaces.addItems(["MNI152NLin2009cAsym", "MNI152NLin6Asym", "fsaverage5", "fsaverage"])
        param_layout.addRow("输出空间:", self.output_spaces)

        self.use_docker = QCheckBox("使用 Docker (否则使用 Apptainer)")
        self.use_docker.setChecked(False)
        param_layout.addRow("", self.use_docker)

        layout.addWidget(param_group)

        # 按钮
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("启动预处理")
        self.btn_start.setStyleSheet(
            "background-color: #409EFF; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_preprocessing)

        self.btn_stop = QPushButton("停止处理")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "background-color: #F56C6C; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_stop.clicked.connect(self.stop_preprocessing)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        layout.addWidget(self.progress_bar)
        layout.addLayout(btn_layout)

        return widget

    def create_log_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        label = QLabel("处理日志")
        label.setFont(QFont("", 10, QFont.Bold))
        layout.addWidget(label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        layout.addWidget(self.log_text)

        btn_clear = QPushButton("清空日志")
        btn_clear.clicked.connect(self.log_text.clear)
        layout.addWidget(btn_clear)

        return widget

    def select_directory(self, line_edit):
        directory = QFileDialog.getExistingDirectory(self, "选择目录")
        if directory:
            line_edit.setText(directory)

    def select_file(self, line_edit, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            line_edit.setText(file_path)

    def start_preprocessing(self):
        if not self.bids_dir.text():
            QMessageBox.warning(self, "警告", "请选择 BIDS 数据集目录")
            return
        if not self.output_dir.text():
            QMessageBox.warning(self, "警告", "请选择输出目录")
            return

        config = {
            'bids_dir': self.bids_dir.text(),
            'output_dir': self.output_dir.text(),
            'work_dir': self.work_dir.text(),
            'fs_license': self.fs_license.text(),
            'container_path': self.container_path.text(),
            'participant_label': self.participant_label.text(),
            'n_cpus': self.n_cpus.value(),
            'mem_mb': self.mem_mb.value(),
            'output_spaces': [self.output_spaces.currentText()],
            'use_docker': self.use_docker.isChecked()
        }

        self.worker = FMRIPrepThread(config)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker.start()

    def stop_preprocessing(self):
        if self.worker:
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    def append_log(self, message):
        self.log_text.append(message)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_finished(self, success, message):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if success:
            QMessageBox.information(self, "完成", message)
        else:
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
                output_files = []
                for f in os.listdir(self.output_dir):
                    if f.endswith(".nii") or f.endswith(".nii.gz"):
                        output_files.append(os.path.join(self.output_dir, f))
                self.finished_status.emit(True, "转换完成！文件已存至输出目录。", output_files)
            else:
                self.finished_status.emit(False, f"转换失败：{process.stderr}", [])
        except Exception as e:
            self.finished_status.emit(False, f"发生异常：{str(e)}", [])


# ==================== 预处理主页面（带选项卡） ====================
class PreprocessingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reJSON = R_W_JSON()
        self.thread = None
        self.bids_thread = None

        self.bids_log_text = QTextEdit()
        self.bids_log_text.setReadOnly(True)
        self.bids_log_text.setFont(QFont("Courier New", 9))
        self.bids_log_text.setPlaceholderText("BIDS整理日志将在这里显示...")

        self.init_ui()



    def init_ui(self):
        layout = QVBoxLayout(self)

        # 创建选项卡
        self.tab_widget = QTabWidget()

        original_tab = QWidget()
        original_layout = QVBoxLayout(original_tab)

        # 顶部按钮区
        btn_layout = QHBoxLayout()
        self.btn_import_history = QPushButton("从数据加载页导入")
        self.btn_import_new = QPushButton("新数据导入")
        self.btn_dcm_convert = QPushButton("DICOM 转 NIfTI")
        self.btn_to_bids = QPushButton("📁 整理为BIDS格式")

        btn_layout.addWidget(self.btn_import_history)
        btn_layout.addWidget(self.btn_import_new)
        btn_layout.addWidget(self.btn_dcm_convert)
        btn_layout.addWidget(self.btn_to_bids)
        original_layout.addLayout(btn_layout)

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

        original_layout.addWidget(self.task_table)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        original_layout.addWidget(self.progress_bar)

        # 信号绑定
        self.btn_import_history.clicked.connect(self.load_from_json)
        self.btn_import_new.clicked.connect(self.open_file)
        self.btn_dcm_convert.clicked.connect(self.start_conversion)
        self.btn_to_bids.clicked.connect(self.organize_to_bids)  # 绑定新按钮

        self.tab_widget.addTab(original_tab, "📁 数据导入与转换")

        # ================= BIDS日志 Tab =================
        bids_log_tab = QWidget()
        bids_log_layout = QVBoxLayout(bids_log_tab)
        bids_log_layout.addWidget(self.bids_log_text)

        # 清空日志按钮
        btn_clear_bids_log = QPushButton("清空BIDS日志")
        btn_clear_bids_log.clicked.connect(self.bids_log_text.clear)
        bids_log_layout.addWidget(btn_clear_bids_log)

        self.tab_widget.addTab(bids_log_tab, "📄 BIDS整理日志")

        self.fmriprep_tab = FMRIPrepWidget()
        self.tab_widget.addTab(self.fmriprep_tab, "🧠 fMRIPrep 预处理")

        layout.addWidget(self.tab_widget)

    def append_bids_log(self, message):
        """在BIDS整理日志中添加一条记录"""

        self.bids_log_text.append(message)
        cursor = self.bids_log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.bids_log_text.setTextCursor(cursor)

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
        if not file_path:
            return

        exist_row = self.find_row_by_path(file_path)
        if exist_row != -1:
            self.update_task_status(exist_row, status)
            return

        row = self.task_table.rowCount()
        self.task_table.insertRow(row)

        file_name = os.path.basename(file_path)

        name_item = QTableWidgetItem(file_name)
        name_item.setToolTip(file_name)
        name_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.task_table.setItem(row, 0, name_item)

        path_item = QTableWidgetItem(file_path)
        path_item.setToolTip(file_path)
        path_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.task_table.setItem(row, 1, path_item)

        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignCenter)
        self.task_table.setItem(row, 2, status_item)
        self.set_status_style(row, status)

        remove_btn = QPushButton("移除")
        remove_btn.clicked.connect(lambda _, b=remove_btn: self.remove_task_row(b))
        self.task_table.setCellWidget(row, 3, remove_btn)

        self.task_table.setRowHeight(row, 42)

    def remove_task_row(self, button):
        for row in range(self.task_table.rowCount()):
            if self.task_table.cellWidget(row, 3) == button:
                self.task_table.removeRow(row)
                break

    def find_row_by_path(self, file_path):
        for row in range(self.task_table.rowCount()):
            item = self.task_table.item(row, 1)
            if item and item.text() == file_path:
                return row
        return -1

    def update_task_status(self, row, status):
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
        dcm_dir = QFileDialog.getExistingDirectory(self, "选择 DICOM 文件夹")
        if not dcm_dir:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "选择 NIfTI 输出存放位置")
        if not out_dir:
            return

        pseudo_output_path = os.path.join(out_dir, os.path.basename(dcm_dir) + ".nii.gz")
        self.add_task_row(pseudo_output_path, "处理中")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.btn_dcm_convert.setEnabled(False)
        self.btn_import_history.setEnabled(False)
        self.btn_import_new.setEnabled(False)
        self.btn_to_bids.setEnabled(False)

        self.thread = ConvertThread(dcm_dir, out_dir)
        self.thread.finished_status.connect(self.on_convert_finished)
        self.thread.start()

    def on_convert_finished(self, success, message, output_files):
        self.progress_bar.setVisible(False)
        self.btn_dcm_convert.setEnabled(True)
        self.btn_import_history.setEnabled(True)
        self.btn_import_new.setEnabled(True)
        self.btn_to_bids.setEnabled(True)

        if success:
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
            for row in range(self.task_table.rowCount()):
                status_item = self.task_table.item(row, 2)
                if status_item and status_item.text() == "处理中":
                    self.update_task_status(row, "失败")

            QMessageBox.critical(self, "错误", message)

    # ==================== BIDS格式整理功能 ====================

    def organize_to_bids(self):
        """将表格中的NIfTI文件整理成BIDS格式"""

        # 获取任务表格中的所有文件
        files_to_organize = []
        for row in range(self.task_table.rowCount()):
            path_item = self.task_table.item(row, 1)
            if path_item:
                files_to_organize.append(path_item.text())

        if not files_to_organize:
            QMessageBox.warning(self, "提示", "没有待整理的文件，请先导入数据")
            return

        # 选择输出目录（BIDS根目录）
        output_dir = QFileDialog.getExistingDirectory(self, "选择BIDS输出目录")
        if not output_dir:
            return

        # 询问被试ID
        subject_id, ok = QInputDialog.getText(self, "被试信息", "请输入被试ID (例如: sub-01):")
        if not ok or not subject_id:
            return

        # 确保被试ID以sub-开头
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{subject_id}"

        # 选择转换方式
        reply = QMessageBox.question(self, "选择转换方式",
                                     "请选择转换方式:\n\n"
                                     "是 - 自动识别 (根据文件名判断T1/BOLD)\n"
                                     "否 - 手动指定 (逐个文件选择类型)\n"
                                     "取消 - 退出",
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

        if reply == QMessageBox.Cancel:
            return
        elif reply == QMessageBox.Yes:
            self.auto_convert_to_bids(files_to_organize, output_dir, subject_id)
        else:
            self.manual_convert_to_bids(files_to_organize, output_dir, subject_id)

    def auto_convert_to_bids(self, files, output_dir, subject_id):
        """自动识别文件类型并转换为BIDS"""

        # 创建BIDS目录结构
        anat_dir = os.path.join(output_dir, subject_id, "anat")
        func_dir = os.path.join(output_dir, subject_id, "func")
        os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(func_dir, exist_ok=True)

        t1_files = []
        bold_files = []
        unknown_files = []

        # 根据文件名判断文件类型
        for file_path in files:
            filename = os.path.basename(file_path).lower()

            # 判断是否为结构像
            if any(keyword in filename for keyword in ['t1', 'mprage', '3d', 'structural', 'anat']):
                t1_files.append(file_path)
            # 判断是否为功能像
            elif any(keyword in filename for keyword in ['bold', 'rest', 'fmri', 'func', 'task']):
                bold_files.append(file_path)
            else:
                unknown_files.append(file_path)

        # 处理结构像
        for i, file_path in enumerate(t1_files):
            if i == 0:
                dest = os.path.join(anat_dir, f"{subject_id}_T1w.nii.gz")
            else:
                dest = os.path.join(anat_dir, f"{subject_id}_run-{i + 1}_T1w.nii.gz")
            write_nifti_gz(file_path, dest)
            self.generate_json_metadata(dest, "T1w")
            self.append_bids_log(f"已添加结构像: {os.path.basename(dest)}")

        # 处理功能像
        for i, file_path in enumerate(bold_files):
            if len(bold_files) == 1:
                dest = os.path.join(func_dir, f"{subject_id}_task-rest_bold.nii.gz")
            else:
                dest = os.path.join(func_dir, f"{subject_id}_task-rest_run-{i + 1}_bold.nii.gz")
            write_nifti_gz(file_path, dest)
            self.generate_json_metadata(dest, "bold")
            self.append_bids_log(f"已添加功能像: {os.path.basename(dest)}")

        # 处理未知文件
        for file_path in unknown_files:
            reply = QMessageBox.question(self, "未知文件类型",
                                         f"文件: {os.path.basename(file_path)}\n\n"
                                         "无法自动识别类型，请选择:\n"
                                         "是 - 作为结构像 (T1)\n"
                                         "否 - 作为功能像 (BOLD)\n"
                                         "取消 - 跳过此文件",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

            if reply == QMessageBox.Yes:
                dest = os.path.join(anat_dir, f"{subject_id}_T1w.nii.gz")
                write_nifti_gz(file_path, dest)
                self.generate_json_metadata(dest, "T1w")
                self.append_bids_log(f"已添加结构像(手动): {os.path.basename(dest)}")
            elif reply == QMessageBox.No:
                dest = os.path.join(func_dir, f"{subject_id}_task-rest_bold.nii.gz")
                write_nifti_gz(file_path, dest)
                self.generate_json_metadata(dest, "bold")
                self.append_bids_log(f"已添加功能像(手动): {os.path.basename(dest)}")

        # 生成BIDS必需文件
        self.generate_dataset_description(output_dir)
        self.generate_participants_tsv(output_dir, subject_id)

        QMessageBox.information(self, "完成",
                                f"BIDS数据集已创建在:\n{output_dir}\n\n"
                                f"目录结构:\n"
                                f"├── {subject_id}/\n"
                                f"│   ├── anat/\n"
                                f"│   │   └── {subject_id}_T1w.nii.gz\n"
                                f"│   └── func/\n"
                                f"│       └── {subject_id}_task-rest_bold.nii.gz\n"
                                f"├── dataset_description.json\n"
                                f"└── participants.tsv\n\n"
                                f"现在可以在 fMRIPrep 预处理界面选择该目录进行处理。")

    def manual_convert_to_bids(self, files, output_dir, subject_id):
        """手动指定每个文件的类型"""

        # 创建BIDS目录结构
        anat_dir = os.path.join(output_dir, subject_id, "anat")
        func_dir = os.path.join(output_dir, subject_id, "func")
        os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(func_dir, exist_ok=True)

        t1_count = 0
        bold_count = 0

        for file_path in files:
            filename = os.path.basename(file_path)

            reply = QMessageBox.question(self, "文件类型",
                                         f"文件: {filename}\n\n"
                                         "这是结构像(T1)还是功能像(fMRI)?\n\n"
                                         "是 - 结构像 (anat/T1w)\n"
                                         "否 - 功能像 (func/task-rest_bold)",
                                         QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                # 结构像
                t1_count += 1
                if t1_count == 1:
                    dest = os.path.join(anat_dir, f"{subject_id}_T1w.nii.gz")
                else:
                    dest = os.path.join(anat_dir, f"{subject_id}_run-{t1_count}_T1w.nii.gz")
                write_nifti_gz(file_path, dest)
                self.generate_json_metadata(dest, "T1w")
                self.append_bids_log(f"已添加结构像: {os.path.basename(dest)}")
            else:
                # 功能像
                bold_count += 1
                if bold_count == 1:
                    dest = os.path.join(func_dir, f"{subject_id}_task-rest_bold.nii.gz")
                else:
                    dest = os.path.join(func_dir, f"{subject_id}_task-rest_run-{bold_count}_bold.nii.gz")
                write_nifti_gz(file_path, dest)
                self.generate_json_metadata(dest, "bold")
                self.append_bids_log(f"已添加功能像: {os.path.basename(dest)}")

        # 生成BIDS必需文件
        self.generate_dataset_description(output_dir)
        self.generate_participants_tsv(output_dir, subject_id)

        QMessageBox.information(self, "完成",
                                f"BIDS数据集已创建在:\n{output_dir}\n\n"
                                f"现在可以在 fMRIPrep 预处理界面选择该目录进行处理。")

    def generate_json_metadata(self, nifti_path, modality):
        """为NIfTI文件生成BIDS兼容的JSON元数据"""

        json_path = nifti_path.replace('.nii.gz', '.json').replace('.nii', '.json')

        # 尝试从原文件中读取信息
        try:
            img = nib.load(nifti_path)
            header = img.header
            zooms = header.get_zooms()

            # 获取TR（重复时间）
            if len(zooms) > 3:
                tr = float(zooms[3])
            else:
                tr = 2.0

            # 获取体素大小
            voxel_sizes = zooms[:3] if len(zooms) >= 3 else [1.0, 1.0, 1.0]

        except Exception as e:
            print(f"读取NIfTI信息失败: {e}")
            tr = 2.0
            voxel_sizes = [1.0, 1.0, 1.0]

        # 构建元数据
        metadata = {
            "RepetitionTime": tr,
            "EchoTime": 0.03,
            "FlipAngle": 90,
            "Manufacturer": "Unknown",
            "ManufacturersModelName": "Unknown",
            "PixelSpacing": [voxel_sizes[0], voxel_sizes[1]],
            "SliceThickness": voxel_sizes[2]
        }

        if modality == "bold":
            metadata["TaskName"] = "rest"

        # 写入JSON文件，处理 numpy 类型
        def np_encoder(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                raise TypeError(f"Unserializable object {obj} of type {type(obj)}")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=np_encoder)



    def append_bids_log(self, message):
        """在BIDS整理日志中添加一条记录"""
        self.bids_log_text.append(message)
        cursor = self.bids_log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.bids_log_text.setTextCursor(cursor)

    def generate_dataset_description(self, output_dir):
        """生成 dataset_description.json"""

        description = {"Name": "BIDS Dataset", "BIDSVersion": "1.8.0", "DatasetType": "raw", "License": "CC0"}
        path = os.path.join(output_dir, "dataset_description.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(description, f, indent=2, ensure_ascii=False)
        self.append_bids_log("已生成 dataset_description.json")

    def generate_participants_tsv(self, output_dir, subject_id):
        """生成 participants.tsv"""

        tsv_path = os.path.join(output_dir, "participants.tsv")
        header = ["participant_id"]
        data = [subject_id]
        write_header = not os.path.exists(tsv_path)
        with (open(tsv_path, "a", encoding="utf-8") as f):
            if write_header:
                f.write("\t".join(header) + "\n")
            f.write("\t".join(data) + "\n")
        self.append_bids_log("已生成 participants.tsv")