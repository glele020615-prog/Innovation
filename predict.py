import os
import sys
import torch
import numpy as np
from scipy import io
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QPushButton, QFileDialog, QLabel, QComboBox, QMessageBox)
from PySide6.QtCore import Qt
from SelfAttentionBlock_ST_GCN import Model
import numpy as np
import torch.nn.functional as F


def get_resource_path(*parts):
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, *parts)

class DiagnosisPage(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window

        self.init_ui()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_diagnostic_model()
        self.current_case = {
            "fc_path": "",
            "fc_var": "",
            "bold_path": "",
            "bold_var": "",
            "fc_shape": None,
            "bold_shape": None,
            "prediction": None,
            "prediction_name": "",
            "prediction_prob": None,
            "probs": None,
            "explain_result": None
        }

        # 存储加载的数据
        self.fc_data = None
        self.bold_data = None

        self.setStyleSheet("""
        QWidget {
            background-color: #F5F7FA;
            font-size: 14px;
        }
        QGroupBox {
            background: white;
            border: 1px solid #E4E7ED;
            border-radius: 10px;
            margin-top: 10px;
            font-weight: bold;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
            color: white;
            background-color: #2c3e50;
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
        QComboBox {
            padding: 6px 10px;
            border: 1px solid #DCDFE6;
            border-radius: 6px;
            background: white;
        }
        QLabel {
            color: #303133;
        }
        """)

        self.btn_predict.setStyleSheet("""
        QPushButton {
            background-color: #2ECC71;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 12px;
        }
        QPushButton:hover {
            background-color: #58D68D;
        }
        """)



    def init_ui(self):
        layout = QVBoxLayout(self)

        self.label_fc_info = QLabel("FC：未加载")
        self.label_bold_info = QLabel("BOLD：未加载")
        self.label_prob_info = QLabel("概率：--")

        self.label_fc_info.setWordWrap(True)
        self.label_bold_info.setWordWrap(True)
        self.label_prob_info.setWordWrap(True)

        # --- 1. 数据输入区 ---
        input_group = QGroupBox("1. 输入诊断数据 (.mat / .npy)")
        input_layout = QVBoxLayout(input_group)

        # FC 矩阵加载
        fc_layout = QHBoxLayout()
        self.btn_load_fc = QPushButton("加载功能连接 (FC)")
        self.combo_fc_var = QComboBox()  # 用于选择 .mat 内部变量
        self.combo_fc_var.setPlaceholderText("选择变量...")
        fc_layout.addWidget(self.btn_load_fc)
        fc_layout.addWidget(self.combo_fc_var)
        input_layout.addLayout(fc_layout)

        # BOLD 信号加载
        bold_layout = QHBoxLayout()
        self.btn_load_bold = QPushButton("加载血氧水平依赖 (BOLD)")
        self.combo_bold_var = QComboBox()
        self.combo_bold_var.setPlaceholderText("选择变量...")
        bold_layout.addWidget(self.btn_load_bold)
        bold_layout.addWidget(self.combo_bold_var)
        input_layout.addLayout(bold_layout)

        layout.addWidget(input_group)

        info_group = QGroupBox("2. 当前已加载数据")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(self.label_fc_info)
        info_layout.addWidget(self.label_bold_info)
        info_layout.addWidget(self.label_prob_info)
        layout.addWidget(info_group)

        # --- 3. 诊断控制与结果 ---
        diag_group = QGroupBox("3. 智能诊断预测")
        diag_layout = QVBoxLayout(diag_group)

        self.btn_predict = QPushButton("智能预测")
        self.btn_predict.setFixedHeight(50)
        self.btn_predict.setStyleSheet("background-color: #2ECC71; color: white; font-weight: bold;")

        self.btn_to_report = QPushButton("发送到个体报告分析")
        self.btn_to_report.setEnabled(False)

        self.label_result = QLabel("诊断结果: 等待输入")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("font-size: 18px; color: #E67E22; margin: 20px;")

        diag_layout.addWidget(self.btn_predict)
        diag_layout.addWidget(self.btn_to_report)
        diag_layout.addWidget(self.label_result)
        layout.addWidget(diag_group)


        # 信号绑定
        self.btn_load_fc.clicked.connect(lambda: self.load_data_file("FC"))
        self.btn_load_bold.clicked.connect(lambda: self.load_data_file("BOLD"))
        self.btn_predict.clicked.connect(self.run_inference)

        self.combo_fc_var.currentTextChanged.connect(self.on_fc_var_changed)
        self.combo_bold_var.currentTextChanged.connect(self.on_bold_var_changed)

        self.btn_to_report.clicked.connect(self.send_to_report_page)





    def load_data_file(self, data_type):
        """支持 .mat 和 .npy 的通用加载函数"""
        path, _ = QFileDialog.getOpenFileName(self, f"选择 {data_type} 文件", "", "Data (*.mat *.npy)")
        if not path: return

        if path.endswith('.mat'):
            data_dict = io.loadmat(path)
            variables = [k for k in data_dict.keys() if not k.startswith('__')]
            combo = self.combo_fc_var if data_type == "FC" else self.combo_bold_var
            combo.clear()
            combo.addItems(variables)
            setattr(self, f"{data_type.lower()}_dict", data_dict)
        else:
            # .npy 直接加载
            data = np.load(path)
            if data_type == "FC":
                self.fc_data = data
            else:
                self.bold_data = data
            QMessageBox.information(self, "成功", f"{data_type} .npy 数据加载成功")

        if data_type == "FC":
            self.current_case["fc_path"] = path
            if path.endswith(".npy"):
                self.current_case["fc_var"] = "npy_array"
                self.current_case["fc_shape"] = data.shape
                self.label_fc_info.setText(
                    f"FC文件：{os.path.basename(path)}\n"
                    f"路径：{path}\n"
                    f"变量：npy_array\n"
                )
        else:
            self.current_case["bold_path"] = path
            if path.endswith(".npy"):
                self.current_case["bold_var"] = "npy_array"
                self.current_case["bold_shape"] = data.shape
                self.label_bold_info.setText(
                    f"BOLD文件：{os.path.basename(path)}\n"
                    f"路径：{path}\n"
                    f"变量：npy_array\n"
                )
        if data_type == "FC" and path.endswith(".mat") and self.combo_fc_var.count() > 0:
            self.on_fc_var_changed(self.combo_fc_var.currentText())

        if data_type == "BOLD" and path.endswith(".mat") and self.combo_bold_var.count() > 0:
            self.on_bold_var_changed(self.combo_bold_var.currentText())

    def on_fc_var_changed(self, var_name):
        if hasattr(self, "fc_dict") and var_name in self.fc_dict:
            self.fc_data = self.fc_dict[var_name]
            self.current_case["fc_var"] = var_name
            self.current_case["fc_shape"] = self.fc_data.shape
            self.label_fc_info.setText(
                f"FC文件：{os.path.basename(self.current_case['fc_path'])}\n"
                f"路径：{self.current_case['fc_path']}\n"
                f"变量：{var_name}\n"
            )

    def on_bold_var_changed(self, var_name):
        if hasattr(self, "bold_dict") and var_name in self.bold_dict:
            self.bold_data = self.bold_dict[var_name]
            self.current_case["bold_var"] = var_name
            self.current_case["bold_shape"] = self.bold_data.shape
            self.label_bold_info.setText(
                f"BOLD文件：{os.path.basename(self.current_case['bold_path'])}\n"
                f"路径：{self.current_case['bold_path']}\n"
                f"变量：{var_name}\n"
            )

    def binarize_matrix(self, matrix, threshold_percent=30):
        flat_matrix = matrix.flatten()
        num_elements = len(flat_matrix)
        num_top_elements = int(num_elements * threshold_percent / 100)
        threshold = np.partition(flat_matrix, -num_top_elements)[-num_top_elements]
        return np.where(matrix >= threshold, 1, 0)

    def resize_bold_to_130(self, bold_data):
        # bold_data: (V, T)
        x = torch.tensor(bold_data, dtype=torch.float32)  # (V, T)
        x = x.unsqueeze(0)  # (1, V, T)
        x = F.interpolate(x, size=130, mode='linear', align_corners=False)
        x = x.squeeze(0).cpu().numpy()  # (V, 130)
        return x

    def load_diagnostic_model(self, num_nodes=116, num_time_points=130):
        """
        根据输入维度动态加载模型。
        优先复用 agent.skills.predict 的全局单例（诊断页与 AI 助手共用一份权重，
        避免打包后同时驻留两个 Model 实例、显存占用翻倍）。
        """
        try:
            from agent.skills.predict import shared_model, install_model_cleanup
            install_model_cleanup()
            model = shared_model(num_nodes=num_nodes, num_time_points=num_time_points)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"[提示] 未能复用共享模型，回退到独立加载: {e}")

        try:
            # 动态传入参数
            model = Model(
                in_channels=1,
                out_channels=32,
                num_class=2,
                edge_importance_weighting=True,
                temporal_kernel_size=7,
                kernel_gcn=3,
                num_nodes= num_nodes,
                num_time_points= num_time_points,
                d_model=32
            )

            # 加载权重时使用 weights_only=True 以消除警告
            model_path = get_resource_path("best_acc_model_fold_1.pt")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

            # 如果加载的权重维度与新模型不完全匹配（例如全连接层），可能需要处理
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None

    def send_to_report_page(self):
        if not self.current_case:
            QMessageBox.warning(self, "提示", "请先完成诊断")
            return

        if self.main_window is None:
            QMessageBox.warning(self, "提示", "主窗口未连接")
            return

        self.main_window.open_individual_report(self.current_case)

    def run_inference(self):
        try:
            if hasattr(self, "bold_dict"):
                self.bold_data = self.bold_dict[self.combo_bold_var.currentText()]
            if hasattr(self, "fc_dict"):
                self.fc_data = self.fc_dict[self.combo_fc_var.currentText()]

            if self.fc_data is None or self.bold_data is None:
                raise ValueError("请先确保 FC 和 BOLD 数据已加载并选择变量")

            # 1. 处理 BOLD：与训练阶段对齐到130时间点
            bold = np.nan_to_num(self.bold_data, nan=0.0).astype(np.float32)
            if bold.shape[1] != 130:
                bold = self.resize_bold_to_130(bold)

            actual_nodes = bold.shape[0]

            # 2. 处理 FC：与训练一致（二值化 + 对称化）
            fc = np.nan_to_num(self.fc_data, nan=0.0).astype(np.float32)
            fc = self.binarize_matrix(fc, threshold_percent=30)
            fc = (fc + fc.T) / 2.0

            # 3. 固定用训练时维度加载模型
            self.model = self.load_diagnostic_model(
                num_nodes=actual_nodes,
                num_time_points=130
            )
            if self.model is None:
                raise RuntimeError("模型加载失败，请确认 best_acc_model_fold_1.pt 已随程序打包并可正常读取")

            # 4. 构造输入
            x_tensor = torch.from_numpy(bold).float().transpose(0, 1).unsqueeze(0).unsqueeze(0).to(self.device)
            adj_tensor = torch.from_numpy(fc).float().unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(x_tensor, adj_tensor)
                probs = torch.exp(output)
                prediction = torch.argmax(output, dim=1).item()

            self.current_case["fc_data"] = fc.copy()
            self.current_case["bold_data"] = bold.copy()
            self.current_case["prediction"] = prediction
            self.current_case["pred_prob"] = probs[0, prediction].item()
            self.current_case["probs"] = probs.squeeze(0).detach().cpu().numpy()

            self.btn_to_report.setEnabled(True)

            #print(f"BOLD shape: {bold.shape}")
            #print(f"FC shape: {fc.shape}")
            #print(f"Raw model output: {output}")
            #print(f"Predicted probabilities: {probs}")

            pred_prob = probs[0, prediction].item() * 100

            if prediction == 0:
                self.label_result.setText(f"{pred_prob:.2f}%的概率四年内不发生认知进展")
            else:
                self.label_result.setText(f"{pred_prob:.2f}%的概率四年内会发生认知进展")

            prob_0 = probs[0, 0].item() * 100
            prob_1 = probs[0, 1].item() * 100
            self.label_prob_info.setText(
                f"认知稳定的概率：{prob_0:.2f}%\n"
                f"认知进展的概率：{prob_1:.2f}%"
            )

        except Exception as e:
            QMessageBox.critical(self, "推理失败", f"错误详情: {str(e)}")



