import os
import numpy as np
import networkx as nx
from nilearn import plotting
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
from pycirclize import Circos


def get_preferred_cjk_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            return font_path
    return None


def configure_matplotlib_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_font()



def load_region_order(label_file):
    with open(label_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_node_coordinates(node_file):
    coords = {}
    with open(node_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[:3])
            name = parts[-1]
            coords[name] = (x, y, z)
    return coords


def top10_nodes_from_edge_importance(edge_imp, region_names):
    node_scores = edge_imp.sum(axis=1)
    idx = np.argsort(node_scores)[::-1][:10]
    return [(region_names[i], float(node_scores[i]), int(i)) for i in idx]


def top10_edges_from_edge_importance(edge_imp, region_names):
    n = edge_imp.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(edge_imp[i, j])))
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
    return [
        {
            "rank": k + 1,
            "i": i,
            "j": j,
            "name": f"{region_names[i]} -- {region_names[j]}",
            "score": score
        }
        for k, (i, j, score) in enumerate(pairs)
    ]


def plot_top10_nodes_bar(top_nodes, out_png):
    names = [x[0] for x in top_nodes][::-1]
    scores = [x[1] for x in top_nodes][::-1]

    plt.figure(figsize=(8, 5))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.barh(names, scores)
    plt.xlabel("Importance", fontsize=14)
    plt.xticks(fontsize=24)
    plt.title("Top10 Brain Regions", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_top10_edges_chord(top_edges, out_png):
    import matplotlib.pyplot as plt
    from pycirclize import Circos

    # 收集脑区名
    nodes = []
    for e in top_edges:
        nodes.extend([e["node1"], e["node2"]])
    nodes = list(dict.fromkeys(nodes))

    # 每个脑区扇区长度统一设为 1
    sectors = {node: 1 for node in nodes}
    circos = Circos(sectors, space=4)

    # 画外圈和标签
    for sector in circos.sectors:
        track = sector.add_track((90, 100))
        track.axis()
        track.text(sector.name, r=105, size=14, orientation="vertical")

    # 线宽归一化
    weights = [e["score"] for e in top_edges]
    max_w = max(weights) if weights else 1.0

    # 方式 A：用 link_line，传 2 元组 (name, pos)
    for e in top_edges:
        node1 = e["node1"]
        node2 = e["node2"]
        score = e["score"]

        lw = 1.5 + 5.0 * (score / max_w)

        circos.link_line(
            (node1, 0.5),
            (node2, 0.5),
            color="black",
            height_ratio=0.5,
            lw=lw,
            alpha=0.7
        )

    fig = circos.plotfig()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)







def plot_top10_nodes_brain(top_nodes, coord_map, out_png):
    coords = []
    values = []
    for name, score, _ in top_nodes:
        if name in coord_map:
            coords.append(coord_map[name])
            values.append(score)

    display = plotting.plot_markers(
        node_values=values,
        node_coords=coords,
        node_size=80,
        display_mode="ortho",
        colorbar=True,
        title="Top10 Brain Regions"
    )
    display.savefig(out_png, dpi=200)
    display.close()


def export_report_pdf(out_pdf, case_data, result):
    # 注册中文字体（Windows 常用）
    font_path = get_preferred_cjk_font()
    if font_path is None:
        raise RuntimeError("未找到可用的中文字体，请安装 Microsoft YaHei / SimHei / 宋体 等字体")

    pdfmetrics.registerFont(TTFont("CNFont", font_path))


    c = canvas.Canvas(out_pdf, pagesize=A4)
    w, h = A4

    y = h - 40
    c.setFont("CNFont", 16)
    c.drawString(40, y, "Individual fMRI Analysis Report")

    y -= 30
    c.setFont("CNFont", 10)
    subject_name = extract_subject_name(case_data)
    prediction_name = case_data.get("prediction_name", "")
    probability = case_data.get("pred_prob", 0) * 100

    c.drawString(40, y, f"被试名: {subject_name}")
    y -= 20
    c.drawString(40, y, f"预测结果: {prediction_name}")
    y -= 20
    c.drawString(40, y, f"预测概率: {probability:.2f}%")

    y -= 30
    c.setFont("CNFont", 12)
    c.drawString(40, y, "Top10 Brain Regions")
    c.drawString(300, y, "Top10 Brain Connections")
    y -= 18
    c.setFont("Helvetica", 9)
    for i, (name, score, _) in enumerate(result["top_nodes"], 1):
        c.drawString(50, y, f"{i}. {name}   {score:.4f}")
        y -= 14
    y = h - 158
    c.setFont("Helvetica", 9)
    for item in result["top_edges"]:
        c.drawString(310, y, f"{item['rank']}. {item['name']}   {item['score']:.4f}")
        y -= 14

    c.drawImage(result["bar_png"], 40, 350, width=240, height=200)
    c.drawImage(result["circle_png"], 300, 350, width=240, height=200)

    #c.showPage()

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, 100, "Brain Map")
    c.drawImage(result["brain_png"], 40, 90, width=500, height=250)


    c.save()


def extract_subject_name(case_data):
    import os

    # 优先用 FC 路径
    fc_path = case_data.get("fc_path", "")
    bold_path = case_data.get("bold_path", "")

    path = fc_path if fc_path else bold_path
    if not path:
        return "未知被试"

    base = os.path.basename(path)   # matr_002_S_0729_2011-08-16.mat
    name, _ = os.path.splitext(base)  # matr_002_S_0729_2011-08-16

    return name

