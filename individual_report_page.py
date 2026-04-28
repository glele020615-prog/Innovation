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

NODE_NAME_CN = {
    # ===== 额叶 / Frontal =====
    "PreCG": "中央前回",
    "PostCG": "中央后回",
    "PoCG": "中央后回",

    "SFG": "额上回",
    "SFGmed": "内侧额上回",
    "SFGdor": "额上回背外侧部",

    "MFG": "额中回",

    "IFG": "额下回",
    "IFGoperc": "额下回盖部",
    "IFGtriang": "额下回三角部",
    "IFGorb": "额下回眶部",

    "ROL": "Rolandic盖区",
    "SMA": "辅助运动区",
    "OLF": "嗅皮层",

    "ORBsup": "眶额上回",
    "ORBsupmed": "眶额上内侧回",
    "ORBmid": "眶额中回",
    "ORBinf": "眶额下回",
    "REC": "直回",

    # ===== 岛叶 / 扣带回 / Insula & Cingulate =====
    "INS": "岛叶",
    "ACG": "前扣带回",
    "DCG": "中扣带回",
    "PCG": "后扣带回",
    "PCC": "后扣带皮层",
    "MCC": "中扣带皮层",

    # ===== 边缘系统 / Limbic =====
    "HIP": "海马",
    "PHG": "海马旁回",
    "AMYG": "杏仁核",

    # ===== 枕叶 / Occipital =====
    "CAL": "距状裂周围皮层",
    "CUN": "楔叶",
    "LING": "舌回",
    "SOG": "枕上回",
    "MOG": "枕中回",
    "IOG": "枕下回",
    "FFG": "梭状回",

    # ===== 顶叶 / Parietal =====
    "SPG": "顶上小叶",
    "IPL": "顶下小叶",
    "SMG": "缘上回",
    "ANG": "角回",
    "PCUN": "楔前叶",
    "PCL": "旁中央小叶",
    "PreCUN": "楔前叶",

    # ===== 颞叶 / Temporal =====
    "HES": "Heschl回（横颞回）",
    "STG": "颞上回",
    "TPOsup": "颞极上部",
    "MTG": "颞中回",
    "TPOmid": "颞极中部",
    "ITG": "颞下回",

    # ===== 基底节 / 丘脑 / Deep nuclei =====
    "CAU": "尾状核",
    "PUT": "壳核",
    "PAL": "苍白球",
    "THA": "丘脑",
    "NAC": "伏隔核",
    "NAc": "伏隔核",

    # ===== 小脑 / Cerebellum =====
    "CRBLCrus1": "小脑脚I区",
    "CRBLCrus2": "小脑脚II区",
    "CRBL1_2": "小脑1-2区",
    "CRBL3": "小脑3区",
    "CRBL4_5": "小脑4/5区",
    "CRBL6": "小脑6区",
    "CRBL7b": "小脑7b区",
    "CRBL8": "小脑8区",
    "CRBL9": "小脑9区",
    "CRBL10": "小脑10区",

    # ===== 常见别名兼容 =====
    "Cerebelum_Crus1": "小脑脚I区",
    "Cerebelum_Crus2": "小脑小脑脚II区",
    "Cerebelum_1_2": "小脑1-2区",
    "Cerebelum_3": "小脑3区",
    "Cerebelum_4_5": "小脑4/5区",
    "Cerebelum_6": "小脑6区",
    "Cerebelum_7b": "小脑7b区",
    "Cerebelum_8": "小脑8区",
    "Cerebelum_9": "小脑9区",
    "Cerebelum_10": "小脑10区",

    # ===== 另一套常见英文别名兼容（有些AAL导出会这样写） =====
    "Precentral": "中央前回",
    "Postcentral": "中央后回",
    "Frontal_Sup": "额上回",
    "Frontal_Sup_Medial": "内侧额上回",
    "Frontal_Mid": "额中回",
    "Frontal_Inf_Oper": "额下回盖部",
    "Frontal_Inf_Tri": "额下回三角部",
    "Frontal_Inf_Orb": "额下回眶部",
    "Rolandic_Oper": "Rolandic盖区",
    "Supp_Motor_Area": "辅助运动区",
    "Olfactory": "嗅皮层",
    "Frontal_Sup_Orb": "眶额上回",
    "Frontal_Med_Orb": "眶额内侧回",
    "Frontal_Mid_Orb": "眶额中回",
    "Frontal_Inf_Orb_2": "眶额下回",
    "Rectus": "直回",
    "Insula": "岛叶",
    "Cingulum_Ant": "前扣带回",
    "Cingulum_Mid": "中扣带回",
    "Cingulum_Post": "后扣带回",
    "Hippocampus": "海马",
    "ParaHippocampal": "海马旁回",
    "Amygdala": "杏仁核",
    "Calcarine": "距状裂周围皮层",
    "Cuneus": "楔叶",
    "Lingual": "舌回",
    "Occipital_Sup": "枕上回",
    "Occipital_Mid": "枕中回",
    "Occipital_Inf": "枕下回",
    "Fusiform": "梭状回",
    "Parietal_Sup": "顶上小叶",
    "Parietal_Inf": "顶下小叶",
    "SupraMarginal": "缘上回",
    "Angular": "角回",
    "Precuneus": "楔前叶",
    "Paracentral_Lobule": "旁中央小叶",
    "Heschl": "Heschl回",
    "Temporal_Sup": "颞上回",
    "Temporal_Pole_Sup": "颞极上部",
    "Temporal_Mid": "颞中回",
    "Temporal_Pole_Mid": "颞极中部",
    "Temporal_Inf": "颞下回",
    "Caudate": "尾状核",
    "Putamen": "壳核",
    "Pallidum": "苍白球",
    "Thalamus": "丘脑",
    "Accumbens": "伏隔核",
    "Vermis12": "小脑蚓部1-2区",
    "Vermis3": "小脑蚓部3区",
    "Vermis45": "小脑蚓部4/5区",
    "Vermis6": "小脑蚓部6区",
    "Vermis7": "小脑蚓部7区",
    "Vermis8": "小脑蚓部8区",
    "Vermis9": "小脑蚓部9区",
    "Vermis10": "小脑蚓部10区",
}

def to_cn_region(name):
    if not name:
        return name

    side = ""
    base = name

    if "." in name:
        base, suffix = name.rsplit(".", 1)
        if suffix == "L":
            side = "左侧"
        elif suffix == "R":
            side = "右侧"

    cn = NODE_NAME_CN.get(base, base)
    return f"{side}{cn}" if side else cn
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
        self.node_table.verticalHeader().setVisible(False)
        self.edge_table.verticalHeader().setVisible(False)
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

            # 原始英文名：给脑图坐标匹配用
            top_nodes_raw = [
                (item["node_name"], float(item["importance"]), int(item["node_idx"]))
                for item in subject_stats["top_nodes"]
            ]

            top_nodes = [
                (to_cn_region(item["node_name"]), float(item["importance"]), int(item["node_idx"]))
                for item in subject_stats["top_nodes"]
            ]

            # Top10 连接
            def cn_region(name):
                if not name:
                    return name

                base = name
                side = ""

                if "." in name:
                    base, suffix = name.rsplit(".", 1)
                    if suffix == "L":
                        side = "左侧"
                    elif suffix == "R":
                        side = "右侧"

                cn = NODE_NAME_CN.get(base, base)
                return f"{side}{cn}" if side else cn

            top_edges = [
                {
                    "rank": idx + 1,
                    "i": item["i"],
                    "j": item["j"],
                    "node1": cn_region(region_names[item["i"]]),
                    "node2": cn_region(region_names[item["j"]]),
                    "name": f"{cn_region(region_names[item['i']])}-{cn_region(region_names[item['j']])}",
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
            plot_top10_edges_chord(top_edges, circle_png)
            plot_top10_nodes_brain(top_nodes_raw, coord_map, brain_png)

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
