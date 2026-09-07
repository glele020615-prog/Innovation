"""
报告子系统：把 individual_report_page 的渲染逻辑抽成无 UI 的纯函数
====================================================================
render_case_report() 一次做完：可解释性分析 -> 三张图 -> PDF。
individual_report_page.py 与 Agent 的 generate_report 都调用它，
避免"页面里一份、Agent 里再抄一份"的双实现。
"""

import os
import sys
import json
import time
import subprocess
import traceback

from agent.util import PROJECT_ROOT, slugify
from agent.skills.base import BaseTool
from agent.skills.predict import (
    CaseStore, compute_case_explanation, load_brain_names,
)


def resource_path(*parts) -> str:
    base = getattr(sys, "_MEIPASS", PROJECT_ROOT)
    return os.path.join(base, *parts)


def default_report_dir() -> str:
    d = os.path.join(PROJECT_ROOT, "report_cache", "reports")
    os.makedirs(d, exist_ok=True)
    return d


def render_case_report(case_id: str, out_dir: str = "", fmt: str = "pdf",
                       top_k: int = 10) -> dict:
    """完整报告渲染（不依赖任何 Qt 控件）。

    返回 {success, pdf_path, bar_png, circle_png, brain_png, top_nodes, top_edges}
    """
    # 报告渲染通常在工具线程里跑，matplotlib 必须用无 GUI 后端
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)

    from report_explain_utils import (
        load_node_coordinates,
        plot_top10_nodes_bar,
        plot_top10_edges_chord,
        plot_top10_nodes_brain,
        export_report_pdf,
    )

    explanation = compute_case_explanation(case_id, top_k=top_k)
    if not explanation.get("success"):
        return explanation

    store = CaseStore()
    case = store.get(explanation["case_id"]) or {}
    out_dir = out_dir or default_report_dir()
    os.makedirs(out_dir, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = f"{slugify(explanation['case_id'])}_{stamp}"
    bar_png = os.path.join(out_dir, f"{prefix}_top10_nodes_bar.png")
    circle_png = os.path.join(out_dir, f"{prefix}_top10_edges_circle.png")
    brain_png = os.path.join(out_dir, f"{prefix}_top10_nodes_brain.png")

    coord_file = resource_path("Node_AAL116.node")
    coord_map = load_node_coordinates(coord_file) if os.path.exists(coord_file) else {}

    plot_top10_nodes_bar(explanation["top_nodes"], bar_png)
    if coord_map:
        plot_top10_nodes_brain(explanation["top_nodes_raw"], coord_map, brain_png)
    try:
        plot_top10_edges_chord(explanation["top_edges"], circle_png)
    except Exception as e:
        circle_png = ""   # 弦图依赖 pycirclize，失败不阻断报告

    case_data = {
        "fc_path": case.get("fc_path", ""),
        "bold_path": case.get("bold_path", ""),
        "prediction_name": explanation.get("prediction_name", ""),
        "pred_prob": explanation.get("pred_prob", 0.0),
        "prediction": explanation.get("prediction"),
    }
    analysis_result = {
        "top_nodes": explanation["top_nodes"],
        "top_edges": explanation["top_edges"],
        "bar_png": bar_png,
        "circle_png": circle_png,
        "brain_png": brain_png,
    }

    paths = {"png": [p for p in (bar_png, circle_png, brain_png) if p]}

    if fmt.lower() == "pdf":
        pdf_path = os.path.join(out_dir, f"{prefix}_report.pdf")
        export_report_pdf(pdf_path, case_data, analysis_result)
        paths["pdf"] = pdf_path
    elif fmt.lower() == "json":
        pdf_path = os.path.join(out_dir, f"{prefix}_report.json")
        with open(pdf_path, "w", encoding="utf-8") as f:
            json.dump({"case": case_data, "analysis": analysis_result},
                      f, ensure_ascii=False, indent=2)
        paths["pdf"] = pdf_path
    else:
        return {"success": False, "error": f"不支持的报告格式: {fmt}（支持 pdf / json）"}

    out_path = paths["pdf"]

    # 记录到 case，方便 open_report / list_cases
    try:
        data = store.load()
        if explanation["case_id"] in data:
            data[explanation["case_id"]]["report_path"] = out_path
            data[explanation["case_id"]]["report_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            data[explanation["case_id"]]["figures"] = paths["png"]
            store.save(data)
    except Exception:
        pass

    return {
        "success": True,
        "case_id": explanation["case_id"],
        "format": fmt.lower(),
        "report_path": out_path,
        "figures": paths["png"],
        "prediction_name": explanation.get("prediction_name", ""),
        "pred_prob": explanation.get("pred_prob", 0.0),
        "top_nodes": [{"name": n, "importance": round(v, 4)} for n, v, _ in explanation["top_nodes"]],
        "top_edges": [
            {"rank": e["rank"], "name": f"{e['node1']}-{e['node2']}", "score": round(e["score"], 4)}
            for e in explanation["top_edges"]
        ],
        "summary": f"报告已生成: {os.path.basename(out_path)}",
    }


def open_report_file(path: str) -> dict:
    """用系统默认程序打开报告"""
    if not path or not os.path.exists(path):
        return {"success": False, "error": f"报告文件不存在: {path}"}
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return {"success": True, "path": path, "summary": "已用默认程序打开报告"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "path": path}


# ==================== 工具 ====================

class GenerateReportTool(BaseTool):
    name = "generate_report"
    category = "report"
    description = (
        "为已完成预测的病例生成个体化诊断报告（PDF）："
        "包含风险结论、Top10 关键脑区表、Top10 脑区连接表、"
        "条形图 / 弦图 / 脑图三张可视化。case_id 留空则用最近一次预测。"
        "前置条件：该病例已调用过 predict_ad_risk。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "case_id": {"type": "string", "description": "病例ID，留空用最近一次预测", "default": ""},
            "format": {"type": "string", "description": "报告格式：pdf（默认）或 json", "default": "pdf"},
            "out_dir": {"type": "string", "description": "输出目录，默认 report_cache/reports", "default": ""},
        },
        "required": [],
    }
    timeout = 300

    def validate_args(self, args: dict) -> tuple:
        from agent.security import safe_path_or_error
        out_dir = args.get("out_dir") or ""
        if out_dir:
            try:
                args["out_dir"] = safe_path_or_error(out_dir)
            except ValueError as e:
                return args, str(e)
        return args, None

    def run(self, case_id: str = "", format: str = "pdf", out_dir: str = "") -> dict:
        try:
            if not case_id:
                case_id = CaseStore().latest()[0]
            if not case_id:
                return {"success": False, "error": "还没有任何已预测病例，请先调用 predict_ad_risk"}
            return render_case_report(case_id, out_dir=out_dir, fmt=format)
        except Exception as e:
            return {"success": False, "error": f"报告生成失败: {type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-1200:]}

    def summarize(self, result: dict) -> str:
        if not result.get("success"):
            return f"❌ {str(result.get('error', ''))[:60]}"
        return f"报告: {os.path.basename(str(result.get('report_path', '')))}"


class OpenReportTool(BaseTool):
    name = "open_report"
    category = "report"
    description = (
        "用系统默认程序打开已生成的报告文件。case_id 留空则打开最近一次生成的报告；"
        "若该病例尚未生成报告，会先生成再打开。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "case_id": {"type": "string", "description": "病例ID，留空用最近一次", "default": ""},
        },
        "required": [],
    }
    timeout = 120

    def run(self, case_id: str = "") -> dict:
        store = CaseStore()
        case_id = case_id or store.latest()[0]
        if not case_id:
            return {"success": False, "error": "还没有任何病例"}
        case = store.get(case_id)
        path = (case or {}).get("report_path", "")
        if not path or not os.path.exists(path):
            generated = render_case_report(case_id)
            if not generated.get("success"):
                return generated
            path = generated["report_path"]
        return open_report_file(path)


report_tools = [GenerateReportTool(), OpenReportTool()]

__all__ = ["report_tools", "render_case_report", "open_report_file", "default_report_dir"]
