import sys
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from nilearn import plotting, image


class fMRIViewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 1. 图像显示区域 (Matplotlib 画布)
        self.figure = Figure(figsize=(5, 4), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # 2. 控制区域布局 (滑条 + 时间标签)
        self.control_layout = QHBoxLayout()
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setEnabled(False)  # 初始禁用

        self.time_label = QLabel("Time Point: 0 / 0")
        self.time_label.setStyleSheet("color: #3498DB; font-weight: bold; font-size: 14px;")

        self.control_layout.addWidget(self.time_slider)
        self.control_layout.addWidget(self.time_label)
        self.layout.addLayout(self.control_layout)

        # 状态变量
        self.full_img = None
        self.current_time_idx = 0
        self.max_time_points = 1
        self.current_coords = (0, 0, 0)  # 记录当前的十字坐标

        # 3. 信号绑定：拖动滑条实时更新
        self.time_slider.valueChanged.connect(self.on_slider_change)

    def display_nii(self, nii_path):
        """加载影像并初始化"""
        try:
            # 强制转换为 float32 消除警告
            raw_img = image.load_img(nii_path)
            data = raw_img.get_fdata().astype(np.float32)
            self.full_img = image.new_img_like(raw_img, data)

            # 获取维度信息
            shape = self.full_img.shape
            if len(shape) == 4:
                self.max_time_points = shape[3]
                self.time_slider.setEnabled(True)
                self.time_slider.setMaximum(self.max_time_points - 1)
            else:
                self.max_time_points = 1
                self.time_slider.setEnabled(False)

            self.current_time_idx = 0
            self.time_slider.setValue(0)
            self.update_plot()

        except Exception as e:
            print(f"加载影像失败: {e}")

    def on_slider_change(self, value):
        """滑条变动处理"""
        self.current_time_idx = value
        self.update_plot()

    def update_plot(self):
        """核心渲染逻辑"""
        if self.full_img is None:
            return

        self.figure.clear()

        # 提取当前时间点的 3D 图像
        if len(self.full_img.shape) == 4:
            display_img = image.index_img(self.full_img, self.current_time_idx)
            self.time_label.setText(f"Time Point: {self.current_time_idx + 1} / {self.max_time_points}")
        else:
            display_img = self.full_img
            self.time_label.setText("Static 3D Volume")

        # 使用 plot_epi 绘图，关键在于使用固定的切片坐标实现“同步”感
        # 即使切换了时间点，视野中心（十字线）也会保持在你之前选定的位置
        self.display = plotting.plot_epi(
            display_img,
            figure=self.figure,
            display_mode='ortho',
            cut_coords=None,  # cut_coords=None 让 nilearn 自动找中心，或传入记录的坐标
            annotate=True,
            draw_cross=True,
            black_bg=True
        )

        self.canvas.draw()

    def clear_view(self):
        """清空界面"""
        self.full_img = None
        self.time_slider.setValue(0)
        self.time_slider.setEnabled(False)
        self.time_label.setText("Time Point: 0 / 0")
        self.figure.clear()
        self.figure.patch.set_facecolor('black')
        self.canvas.draw()