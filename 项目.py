import sys
import os
from PySide6.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QStackedWidget, QFrame)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGroupBox, QLabel, QPushButton, QFileDialog, QFormLayout, QMessageBox,QAbstractItemView)
from viewer import fMRIViewWidget
import nibabel as nib
from Read_Write_JSON import R_W_JSON
from datetime import datetime
from Data_Pre import PreprocessingPage
from predict import DiagnosisPage
from individual_report_page import IndividualReportPage



class ADDiagnosisSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("阿尔兹海默症 fMRI 智能预测系统 v2.0")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #F5F7FA;")  # 浅灰色背景，更具医疗感

        # 1. 主布局
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setCentralWidget(self.main_widget)

        self.shared_case_data = None

        # 2. 左侧导航栏
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #2C3E50; 
                border-right: 1px solid #DCDFE6;
            }
            QPushButton {
                color: black;
                border: none;
                padding: 15px;
                text-align: left;
                font-size: 14px;
                font-family: 'Microsoft YaHei';
            }
            QPushButton:hover {
                background-color: #34495E;
            }
            QPushButton:checked {
                background-color: #3498DB;
                font-weight: bold;
            }
        """)

        self.sidebar_layout = QVBoxLayout(self.sidebar)

        # 导航标题
        nav_title = QLabel("功能导航")
        nav_title.setStyleSheet("color: #BDC3C7; padding: 10px; font-weight: bold;")
        self.sidebar_layout.addWidget(nav_title)

        # 导航按钮组
        self.btn_home = QPushButton(" 🏠 首页概览")
        self.btn_data = QPushButton(" 📂 数据加载")
        self.btn_preproc = QPushButton(" ⚙️ 预处理")
        self.btn_predict = QPushButton(" 🧠 疾病预测诊断")
        self.btn_report = QPushButton(" 📋 个体报告分析")

        # 设置按钮可选中状态
        for btn in [self.btn_home, self.btn_data, self.btn_preproc, self.btn_predict, self.btn_report]:
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            self.sidebar_layout.addWidget(btn)

        self.sidebar_layout.addStretch()  # 推到顶部
        self.main_layout.addWidget(self.sidebar)

        # 3. 右侧内容区 (Stacked Widget)
        self.content_stack = QStackedWidget()
        self.main_layout.addWidget(self.content_stack)

        # 初始化各页面
        self.init_pages()

        # 4. 信号连接
        self.btn_home.clicked.connect(lambda: self.content_stack.setCurrentIndex(0))
        self.btn_data.clicked.connect(lambda: self.content_stack.setCurrentIndex(1))
        self.btn_preproc.clicked.connect(lambda: self.content_stack.setCurrentIndex(2))
        self.btn_predict.clicked.connect(lambda: self.content_stack.setCurrentIndex(3))
        self.btn_report.clicked.connect(lambda: self.content_stack.setCurrentIndex(4))

        # 默认选中首页
        self.btn_home.setChecked(True)

    def init_pages(self):
        # 首页 (Dashboard)
        page_home = QWidget()
        layout = QVBoxLayout(page_home)
        layout.addWidget(QLabel("<h1>欢迎使用个性化 AD 智能预测系统</h1>"), alignment=Qt.AlignCenter)
        layout.addWidget(QLabel("请从左侧选择操作流程进行分析。"), alignment=Qt.AlignCenter)
        self.content_stack.addWidget(page_home)

        # 数据加载页
        self.page_data = QWidget()
        self.content_stack.addWidget(DataManagerPage())

        # 预处理
        self.page_preproc = PreprocessingPage()
        self.content_stack.addWidget(self.page_preproc)

        #智能预测
        self.intelligent_prediction = DiagnosisPage(main_window=self)
        self.content_stack.addWidget(self.intelligent_prediction)

        # 个体报告分析
        self.individual_report_page = IndividualReportPage(main_window=self)
        self.content_stack.addWidget(self.individual_report_page)

#接收诊断结果并跳转
    def open_individual_report(self, case_data):
        self.shared_case_data = case_data
        self.individual_report_page.load_case(case_data)
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ADDiagnosisSystem()
    window.show()
    sys.exit(app.exec())