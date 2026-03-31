import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QHeaderView, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QGroupBox, QScrollArea
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from report_explain_utils import (
    load_region_order,
    load_node_coordinates,
    top10_nodes_from_edge_importance,
    top10_edges_from_edge_importance,
    plot_top10_nodes_bar,
    plot_top10_edges_chord,
    plot_top10_nodes_brain,
    export_report_pdf
)

from explainability_utils import (
    compute_gradient_based_edge_importance,
    generate_per_subject_analysis_with_stats
)
import torch


class IndividualReportPage(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.case_data = None
        self.analysis_result = None

        self.label_file = r"label_order_jian.node"
        self.coord_file = r"Node_AAL116.node"

        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)

        self.label_title = QLabel("个体分析报告")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.label_info = QLabel("当前样本：未加载")
        self.label_info.setWordWrap(True)

        # Top10 脑区表
        group_nodes = QGroupBox("Top10 脑区")
        group_nodes_layout = QVBoxLayout(group_nodes)
        self.node_table = QTableWidget(0, 3)
        self.node_table.setHorizontalHeaderLabels(["排名", "脑区", "重要性"])
        self.node_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        group_nodes_layout.addWidget(self.node_table)

        # Top10 连接表
        group_edges = QGroupBox("Top10 脑区连接")
        group_edges_layout = QVBoxLayout(group_edges)
        self.edge_table = QTableWidget(0, 3)
        self.edge_table.setHorizontalHeaderLabels(["排名", "连接", "重要性"])
        self.edge_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        group_edges_layout.addWidget(self.edge_table)

        # 图像显示
        group_figs = QGroupBox("可视化结果")
        figs_layout = QVBoxLayout(group_figs)

        self.label_bar_title = QLabel("Top10 脑区条形图")
        self.label_bar = QLabel()
        self.label_bar.setAlignment(Qt.AlignCenter)

        self.label_circle_title = QLabel("Top10 脑区连接图")
        self.label_circle = QLabel()
        self.label_circle.setAlignment(Qt.AlignCenter)

        self.label_brain_title = QLabel("脑图点图")
        self.label_brain = QLabel()
        self.label_brain.setAlignment(Qt.AlignCenter)

        figs_layout.addWidget(self.label_bar_title)
        figs_layout.addWidget(self.label_bar)
        figs_layout.addWidget(self.label_circle_title)
        figs_layout.addWidget(self.label_circle)
        figs_layout.addWidget(self.label_brain_title)
        figs_layout.addWidget(self.label_brain)

        # 底部按钮
        btn_layout = QHBoxLayout()
        self.btn_export_pdf = QPushButton("导出PDF")
        self.btn_export_pdf.clicked.connect(self.export_pdf)

        self.btn_back = QPushButton("返回诊断页")
        self.btn_back.clicked.connect(self.go_back)

        btn_layout.addWidget(self.btn_export_pdf)
        btn_layout.addWidget(self.btn_back)

        layout.addWidget(self.label_title)
        layout.addWidget(self.label_info)
        layout.addWidget(group_nodes)
        layout.addWidget(group_edges)
        layout.addWidget(group_figs)
        layout.addLayout(btn_layout)

    def load_case(self, case_data):
        self.case_data = case_data

        pred = case_data.get("prediction", None)
        pred_prob = case_data.get("pred_prob", 0)

        pred_name = "认知稳定" if pred == 0 else "认知进展"
        case_data["prediction_name"] = pred_name

        self.label_info.setText(
            f"FC文件：{case_data.get('fc_path', '')}\n"
            f"BOLD文件：{case_data.get('bold_path', '')}\n"
            f"预测类别：{pred_name}\n"
            f"预测概率：{pred_prob * 100:.2f}%"
        )

        self.run_individual_analysis()

    def run_individual_analysis(self):
        try:
            if self.case_data is None:
                return

            # 1. 从诊断页复用模型和设备
            diagnosis_page = self.main_window.intelligent_prediction
            model = diagnosis_page.model
            device = diagnosis_page.device

            if model is None:
                raise ValueError("诊断模型未加载，请先在诊断页完成预测")

            # 2. 取已经在诊断页按训练方式处理好的输入
            bold = self.case_data.get("bold_data")
            fc = self.case_data.get("fc_data")
            pred = self.case_data.get("prediction")
            pred_prob = self.case_data.get("pred_prob", 0)

            if bold is None or fc is None:
                raise ValueError("未接收到 BOLD 或 FC 数据")

            # 3. 构造与预测一致的张量
            # bold: (V, 130) -> (1, 1, 130, V)
            x_tensor = torch.from_numpy(bold).float().transpose(0, 1).unsqueeze(0).unsqueeze(0).to(device)

            # fc: (V, V) -> (1, 1, V, V)
            adj_tensor = torch.from_numpy(fc).float().unsqueeze(0).unsqueeze(0).to(device)

            # 4. 用预测类别作为解释目标
            target_label = torch.tensor([pred], dtype=torch.long).to(device)

            # 5. 真实梯度边重要性
            edge_imp = compute_gradient_based_edge_importance(
                model=model,
                data=x_tensor,
                adj=adj_tensor,
                label=target_label,
                device=device
            )

            # 6. 读取脑区名
            region_names = load_region_order(self.label_file)
            coord_map = load_node_coordinates(self.coord_file)

            # 7. 生成单被试 Top10 结果
            subject_stats = generate_per_subject_analysis_with_stats(
                subject_id="current_case",
                edge_imp=edge_imp,
                brain_region_names=region_names,
                true_label=pred,
                pred_label=pred,
                prob_class1=pred_prob if pred == 1 else (1 - pred_prob),
                top_k=10
            )

            top_nodes = [
                (item["node_name"], float(item["importance"]), int(item["node_idx"]))
                for item in subject_stats["top_nodes"]
            ]

            top_edges = [
                {
                    "rank": idx + 1,
                    "i": item["i"],
                    "j": item["j"],
                    "node1": region_names[item["i"]],
                    "node2": region_names[item["j"]],
                    "name": item["edge_name"],
                    "score": float(item["importance"])
                }
                for idx, item in enumerate(subject_stats["top_edges"])
            ]

            out_dir = os.path.join(os.getcwd(), "report_cache")
            os.makedirs(out_dir, exist_ok=True)

            bar_png = os.path.join(out_dir, "top10_nodes_bar.png")
            circle_png = os.path.join(out_dir, "top10_edges_circle.png")
            brain_png = os.path.join(out_dir, "top10_nodes_brain.png")

            plot_top10_nodes_bar(top_nodes, bar_png)
            #print("top_edges =", top_edges)
            plot_top10_edges_chord(top_edges, circle_png)
            plot_top10_nodes_brain(top_nodes, coord_map, brain_png)

            self.analysis_result = {
                "edge_imp": edge_imp,
                "subject_stats": subject_stats,
                "top_nodes": top_nodes,
                "top_edges": top_edges,
                "bar_png": bar_png,
                "circle_png": circle_png,
                "brain_png": brain_png
            }

            self.fill_node_table(top_nodes)
            self.fill_edge_table(top_edges)
            self.load_images()

        except Exception as e:
            QMessageBox.critical(self, "分析失败", str(e))

    def fill_node_table(self, top_nodes):
        self.node_table.setRowCount(len(top_nodes))
        for row, (name, score, _) in enumerate(top_nodes):
            self.node_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.node_table.setItem(row, 1, QTableWidgetItem(name))
            self.node_table.setItem(row, 2, QTableWidgetItem(f"{score:.6f}"))

    def fill_edge_table(self, top_edges):
        self.edge_table.setRowCount(len(top_edges))
        for row, item in enumerate(top_edges):
            self.edge_table.setItem(row, 0, QTableWidgetItem(str(item["rank"])))
            self.edge_table.setItem(row, 1, QTableWidgetItem(item["name"]))
            self.edge_table.setItem(row, 2, QTableWidgetItem(f"{item['score']:.6f}"))

    def load_images(self):
        if not self.analysis_result:
            return

        self.label_bar.setPixmap(QPixmap(self.analysis_result["bar_png"]).scaledToWidth(700, Qt.SmoothTransformation))
        self.label_circle.setPixmap(QPixmap(self.analysis_result["circle_png"]).scaledToWidth(700, Qt.SmoothTransformation))
        self.label_brain.setPixmap(QPixmap(self.analysis_result["brain_png"]).scaledToWidth(700, Qt.SmoothTransformation))

    def export_pdf(self):
        if not self.analysis_result or not self.case_data:
            QMessageBox.warning(self, "提示", "请先完成个体分析")
            return

        path, _ = QFileDialog.getSaveFileName(self, "导出PDF", "individual_report.pdf", "PDF Files (*.pdf)")
        if not path:
            return

        try:
            export_report_pdf(path, self.case_data, self.analysis_result)
            QMessageBox.information(self, "成功", f"PDF已导出：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def go_back(self):
        if self.main_window is not None:
            self.main_window.content_stack.setCurrentWidget(self.main_window.intelligent_prediction)
