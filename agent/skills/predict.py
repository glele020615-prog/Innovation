"""
预测与可解释性子系统
======================
- run_prediction(): 纯函数，numpy 进 -> 结构化结果出（UI 与 Agent 共用）
- predict_ad_risk(): 路径进 -> 预测 + 注册 case（供 explain / report 复用）
- explain_prediction(case_id): 梯度边重要性 + Top 脑区/连接
"""

import os
import sys
import json
import time
import traceback
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import io

from agent.util import PROJECT_ROOT, extract_subject_id, slugify
from agent.skills.base import BaseTool
from agent.security import safe_path_or_error
from brain_region_names import to_cn_region, find_region

_LABEL_FILE = "label_order_jian.node"

_device: Optional[torch.device] = None
_model: Optional[torch.nn.Module] = None
_model_nodes: Optional[int] = None
_brain_names: Optional[list] = None


def resource_path(*parts) -> str:
    base = getattr(sys, "_MEIPASS", PROJECT_ROOT)
    return os.path.join(base, *parts)


def load_brain_names() -> list:
    global _brain_names
    if _brain_names is not None:
        return _brain_names
    node_path = resource_path(_LABEL_FILE)
    if os.path.exists(node_path):
        with open(node_path, "r", encoding="utf-8") as f:
            _brain_names = [line.strip() for line in f if line.strip()]
    else:
        _brain_names = [f"ROI_{i}" for i in range(116)]
    return _brain_names


def get_model(num_nodes: int = 116, num_time_points: int = 130) -> torch.nn.Module:
    global _device, _model, _model_nodes
    if _model is not None and _model_nodes == num_nodes:
        return _model
    from SelfAttentionBlock_ST_GCN import Model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = resource_path("best_acc_model_fold_1.pt")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"模型权重未找到: {weight_path}")
    model = Model(
        in_channels=1, out_channels=32, num_class=2,
        edge_importance_weighting=True, temporal_kernel_size=7, kernel_gcn=3,
        num_nodes=num_nodes, num_time_points=num_time_points, d_model=32,
    )
    state_dict = torch.load(weight_path, map_location=_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()
    _model = model
    _model_nodes = num_nodes
    return _model


def get_device() -> torch.device:
    if _device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def shared_model(num_nodes: int = 116, num_time_points: int = 130) -> torch.nn.Module:
    """全局唯一的模型实例 —— 诊断页与 Agent 共用同一份权重。

    诊断页（predict.py:DiagnosisPage）和 Agent 各建一个 Model 会让
    PyInstaller 打包后的显存/内存占用翻倍，这里统一走一个入口。
    """
    return get_model(num_nodes=num_nodes, num_time_points=num_time_points)


def release_model(empty_cache: bool = True):
    """释放模型单例（退出时调用，或切换到报告页时可主动降显存）"""
    global _model, _model_nodes
    _model = None
    _model_nodes = None
    if empty_cache and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


_cleanup_installed = False


def install_model_cleanup():
    """挂到 QApplication.aboutToQuit：退出时自动释放模型显存（只需调用一次）"""
    global _cleanup_installed
    if _cleanup_installed:
        return
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return
        app.aboutToQuit.connect(release_model)
        _cleanup_installed = True
    except Exception:
        pass


def load_matrix_file(path: str, var_name: str = "") -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    if path.endswith(".mat"):
        data_dict = io.loadmat(path)
        variables = [k for k in data_dict.keys() if not k.startswith("__")]
        if not variables:
            raise ValueError(f"{path} 中没有可用变量")
        key = var_name if var_name in data_dict else variables[0]
        return np.array(data_dict[key])
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path)


def binarize_matrix(matrix: np.ndarray, threshold_percent: int = 30) -> np.ndarray:
    flat = matrix.flatten()
    k = max(1, int(len(flat) * threshold_percent / 100))
    threshold = np.partition(flat, -k)[-k]
    return np.where(matrix >= threshold, 1.0, 0.0)


def resize_bold_to_130(bold: np.ndarray) -> np.ndarray:
    x = torch.tensor(bold, dtype=torch.float32).unsqueeze(0)   # (1, V, T)
    x = F.interpolate(x, size=130, mode="linear", align_corners=False)
    return x.squeeze(0).cpu().numpy()


def prepare_inputs(fc: np.ndarray, bold: np.ndarray) -> tuple:
    bold = np.nan_to_num(bold, nan=0.0).astype(np.float32)
    if bold.ndim == 1:
        bold = bold.reshape(1, -1)
    if bold.shape[1] != 130:
        bold = resize_bold_to_130(bold)
    fc = np.nan_to_num(fc, nan=0.0).astype(np.float32)
    if fc.shape[0] != bold.shape[0]:
        raise ValueError(
            f"FC 与 BOLD 的脑区数不一致: FC={fc.shape[0]}, BOLD={bold.shape[0]}"
        )
    fc = binarize_matrix(fc, 30)
    fc = (fc + fc.T) / 2.0
    return fc, bold


def run_prediction(fc: np.ndarray, bold: np.ndarray) -> dict:
    """核心推理：numpy 进，结构化结果出。"""
    fc, bold = prepare_inputs(fc, bold)
    num_nodes = bold.shape[0]
    model = get_model(num_nodes=num_nodes, num_time_points=130)
    device = get_device()

    x_tensor = torch.from_numpy(bold).float().transpose(0, 1).unsqueeze(0).unsqueeze(0).to(device)
    adj_tensor = torch.from_numpy(fc).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x_tensor, adj_tensor)
        probs = torch.exp(output)
        prediction = int(torch.argmax(output, dim=1).item())

    prob_stable = round(float(probs[0, 0].item()) * 100, 2)
    prob_progress = round(float(probs[0, 1].item()) * 100, 2)

    return {
        "success": True,
        "prediction": prediction,
        "prediction_label": ("认知稳定（4年内不发生进展）" if prediction == 0
                             else "认知进展（4年内可能发生进展）"),
        "prediction_name": "认知稳定" if prediction == 0 else "认知进展",
        "prob_stable": prob_stable,
        "prob_progress": prob_progress,
        "raw_probs": [prob_stable, prob_progress],
        "pred_prob": (prob_progress if prediction == 1 else prob_stable) / 100.0,
        "num_nodes": num_nodes,
        "num_time_points": int(bold.shape[1]),
        "_tensors": (x_tensor, adj_tensor),
        "_arrays": (fc, bold),
    }


def compute_top_regions(x_tensor, adj_tensor, top_k: int = 5) -> list:
    from explainability_utils import compute_gradient_based_edge_importance

    brain_names = load_brain_names()
    try:
        edge_imp = compute_gradient_based_edge_importance(
            get_model(num_nodes=adj_tensor.shape[-1]), x_tensor, adj_tensor,
            label=torch.tensor([0], device=get_device()), device=get_device(),
        )
    except Exception:
        edge_imp = np.abs(adj_tensor[0, 0].cpu().numpy())

    if edge_imp is None or edge_imp.shape != (len(brain_names), len(brain_names)):
        edge_imp = np.zeros((len(brain_names), len(brain_names)))
    return node_ranking(edge_imp, brain_names, top_k)


def node_ranking(edge_imp: np.ndarray, brain_names: list, top_k: int = 5) -> list:
    node_importance = np.sum(np.abs(edge_imp), axis=1) - np.diag(np.abs(edge_imp))
    if node_importance.max() > 0:
        node_importance = node_importance / node_importance.max()
    top_indices = np.argsort(node_importance)[-top_k:][::-1]
    out = []
    for idx in top_indices:
        raw = brain_names[idx] if idx < len(brain_names) else f"ROI_{idx}"
        abbr, info = find_region(raw)
        out.append({
            "name": raw,
            "cn_name": to_cn_region(raw),
            "importance": round(float(node_importance[idx]), 4),
            "ad_relevance": info.get("ad_relevance", "暂无详细 AD 关联知识"),
        })
    return out


# ==================== CaseStore：预测结果的持久化 ====================

class CaseStore:
    """把一次预测固化成 case，供 explain / report 复用（避免重复推理）"""

    def __init__(self, root: str = ""):
        self.root = root or os.path.join(PROJECT_ROOT, "report_cache", "cases")
        self.index_file = os.path.join(self.root, "cases.json")
        os.makedirs(self.root, exist_ok=True)

    def load(self) -> dict:
        if not os.path.exists(self.index_file):
            return {}
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self, data: dict):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get(self, case_id: str) -> dict:
        data = self.load()
        if case_id in data:
            return data[case_id]
        # 模糊匹配（LLM 常给文件名主干）
        for key, value in data.items():
            if case_id.lower() in key.lower():
                return value
        return {}

    def latest(self) -> tuple:
        data = self.load()
        if not data:
            return "", {}
        key = sorted(data.keys(), key=lambda k: data[k].get("created_at", ""))[-1]
        return key, data[key]

    def register(self, case_id: str, payload: dict) -> dict:
        data = self.load()
        payload["case_id"] = case_id
        payload["created_at"] = payload.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S")
        data[case_id] = payload
        self.save(data)
        return payload

    def arrays_dir(self, case_id: str) -> str:
        d = os.path.join(self.root, slugify(case_id))
        os.makedirs(d, exist_ok=True)
        return d


def predict_ad_risk(fc_path: str, bold_path: str, fc_var_name: str = "",
                    bold_var_name: str = "") -> dict:
    """路径版预测：加载 -> 推理 -> 注册 case。返回不含大数组的结构化结果。"""
    try:
        fc_path, err = resolve_data_file(fc_path, prefer="fc")
        if err:
            return {"success": False, "error": err}
        bold_path, err = resolve_data_file(bold_path, prefer="bold")
        if err:
            return {"success": False, "error": err}

        fc_raw = load_matrix_file(fc_path, fc_var_name)
        bold_raw = load_matrix_file(bold_path, bold_var_name)
        result = run_prediction(fc_raw, bold_raw)
        x_tensor, adj_tensor = result.pop("_tensors")
        fc_aligned, bold_aligned = result.pop("_arrays")

        top_regions = compute_top_regions(x_tensor, adj_tensor, top_k=5)
        result["top_brain_regions"] = top_regions

        # 固化 case（含与训练对齐后的 fc/bold，供 explain & report 复用）
        case_id = extract_subject_id(fc_path) or extract_subject_id(bold_path) or "case"
        store = CaseStore()
        arrays_dir = store.arrays_dir(case_id)
        fc_file = os.path.join(arrays_dir, "fc.npy")
        bold_file = os.path.join(arrays_dir, "bold.npy")
        np.save(fc_file, fc_aligned)
        np.save(bold_file, bold_aligned)

        case = store.register(case_id, {
            "case_id": case_id,
            "fc_path": fc_path,
            "bold_path": bold_path,
            "fc_var": fc_var_name,
            "bold_var": bold_var_name,
            "fc_file": fc_file,
            "bold_file": bold_file,
            "fc_shape": list(fc_aligned.shape),
            "bold_shape": list(bold_aligned.shape),
            "prediction": result["prediction"],
            "prediction_name": result["prediction_name"],
            "pred_prob": result["pred_prob"],
            "probs": result["raw_probs"],
            "num_nodes": result["num_nodes"],
            "num_time_points": result["num_time_points"],
            "top_regions": top_regions,
        })

        result.update({
            "success": True,
            "case_id": case_id,
            "summary": f"{result['prediction_name']} {max(result['raw_probs']):.2f}%",
            "next_step": f"调用 explain_prediction(case_id='{case_id}') 看关键脑区，"
                         f"或 generate_report(case_id='{case_id}') 出 PDF",
        })
        return result

    except Exception as e:
        return {"success": False, "error": f"预测失败: {type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1200:]}


def compute_case_explanation(case_id: str, top_k: int = 10) -> dict:
    """对已注册的 case 做梯度可解释性分析，返回完整结构（含 edge_imp 数组）。"""
    from explainability_utils import (
        compute_gradient_based_edge_importance,
        generate_per_subject_analysis_with_stats,
    )

    store = CaseStore()
    case = store.get(case_id)
    if not case:
        return {"success": False, "error": f"未找到病例: {case_id}",
                "hint": "先调用 predict_ad_risk 完成预测，或用 list_cases 查看已有病例"}

    fc = np.load(case["fc_file"]) if os.path.exists(case.get("fc_file", "")) else None
    bold = np.load(case["bold_file"]) if os.path.exists(case.get("bold_file", "")) else None
    if fc is None or bold is None:
        fc = load_matrix_file(case["fc_path"], case.get("fc_var", ""))
        bold = load_matrix_file(case["bold_path"], case.get("bold_var", ""))
        fc, bold = prepare_inputs(fc, bold)

    num_nodes = bold.shape[0]
    model = get_model(num_nodes=num_nodes, num_time_points=130)
    device = get_device()
    x_tensor = torch.from_numpy(bold).float().transpose(0, 1).unsqueeze(0).unsqueeze(0).to(device)
    adj_tensor = torch.from_numpy(fc).float().unsqueeze(0).unsqueeze(0).to(device)
    pred = int(case.get("prediction", 0))

    edge_imp = compute_gradient_based_edge_importance(
        model=model, data=x_tensor, adj=adj_tensor,
        label=torch.tensor([pred], dtype=torch.long).to(device), device=device,
    )

    region_names = load_brain_names()
    stats = generate_per_subject_analysis_with_stats(
        subject_id=case["case_id"], edge_imp=edge_imp, brain_region_names=region_names,
        true_label=pred, pred_label=pred,
        prob_class1=float(case.get("pred_prob", 0.0)), top_k=top_k,
    )

    top_nodes = [
        (to_cn_region(item["node_name"]), float(item["importance"]), int(item["node_idx"]))
        for item in stats["top_nodes"]
    ]
    top_nodes_raw = [
        (item["node_name"], float(item["importance"]), int(item["node_idx"]))
        for item in stats["top_nodes"]
    ]
    top_edges = [
        {
            "rank": idx + 1,
            "i": int(item["i"]), "j": int(item["j"]),
            "node1": to_cn_region(region_names[item["i"]]) if item["i"] < len(region_names) else str(item["i"]),
            "node2": to_cn_region(region_names[item["j"]]) if item["j"] < len(region_names) else str(item["j"]),
            "name": item["edge_name"],
            "score": float(item["importance"]),
        }
        for idx, item in enumerate(stats["top_edges"])
    ]

    arrays_dir = store.arrays_dir(case["case_id"])
    edge_imp_file = os.path.join(arrays_dir, "edge_importance.npy")
    np.save(edge_imp_file, edge_imp)

    return {
        "success": True,
        "case_id": case["case_id"],
        "edge_imp": edge_imp,
        "edge_imp_file": edge_imp_file,
        "fc": fc,
        "bold": bold,
        "top_nodes": top_nodes,
        "top_nodes_raw": top_nodes_raw,
        "top_edges": top_edges,
        "subject_stats": stats,
        "prediction": pred,
        "prediction_name": case.get("prediction_name", ""),
        "pred_prob": case.get("pred_prob", 0.0),
    }


def explain_prediction(case_id: str = "", top_k: int = 10) -> dict:
    """工具入口：只回传 LLM 需要的摘要（不含大数组）。"""
    try:
        case_id = case_id or CaseStore().latest()[0]
        if not case_id:
            return {"success": False, "error": "当前还没有任何已预测的病例，请先调用 predict_ad_risk"}
        result = compute_case_explanation(case_id, top_k=top_k)
        if not result.get("success"):
            return result
        top_nodes = result["top_nodes"]
        return {
            "success": True,
            "case_id": result["case_id"],
            "prediction": result["prediction"],
            "prediction_name": result["prediction_name"],
            "pred_prob": result["pred_prob"],
            "top_nodes": [{"name": n, "importance": round(v, 4)} for n, v, _ in top_nodes],
            "top_edges": [
                {"rank": e["rank"], "name": f"{e['node1']}-{e['node2']}", "score": round(e["score"], 4)}
                for e in result["top_edges"]
            ],
            "summary": f"Top1 脑区: {top_nodes[0][0] if top_nodes else '-'}",
            "next_step": f"需要完整报告时调用 generate_report(case_id='{result['case_id']}')",
        }
    except Exception as e:
        return {"success": False, "error": f"解释失败: {type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1200:]}


def infer_counterpart(known_path: str, want: str) -> str:
    """从已知的 FC（或 BOLD）路径推断另一个文件。

    数据集常见布局：Data/pMCI/FC/xxx.mat 与 Data/pMCI/BOLD/xxx.mat 同名。
    找不到返回空字符串。
    """
    if not known_path or not os.path.exists(known_path):
        return ""
    known = os.path.abspath(known_path)
    parent = os.path.dirname(known)
    base = os.path.basename(known)
    want_low = want.lower()

    # 1) 兄弟目录（FC ↔ BOLD）
    grand = os.path.dirname(parent)
    if os.path.isdir(grand):
        candidates = []
        for name in sorted(os.listdir(grand)):
            sub = os.path.join(grand, name)
            if not os.path.isdir(sub) or os.path.abspath(sub) == parent:
                continue
            cand = os.path.join(sub, base)
            if os.path.isfile(cand):
                if want_low in name.lower():
                    return cand
                candidates.append(cand)
        if len(candidates) == 1:
            return candidates[0]

    # 2) 同目录里名字带关键字的文件
    for name in sorted(os.listdir(parent)):
        if name == base:
            continue
        if want_low in name.lower() and os.path.isfile(os.path.join(parent, name)):
            return os.path.join(parent, name)

    # 3) 项目 Data 目录下按文件名找（取路径中不含 known 目录的那个）
    data_root = os.path.join(PROJECT_ROOT, "Data")
    if os.path.isdir(data_root):
        hits = []
        for root, _dirs, files in os.walk(data_root):
            if base in files:
                cand = os.path.join(root, base)
                if os.path.abspath(cand) != known:
                    hits.append(cand)
            if len(hits) > 8:
                break
        if len(hits) == 1:
            return hits[0]
        for cand in hits:
            if want_low in cand.lower().replace("\\", "/").split("/")[-2:][-1].lower():
                return cand
        if hits:
            return sorted(hits)[0]
    return ""


def resolve_data_file(path: str, prefer: str = "") -> tuple:
    """解析数据文件路径：越界检查 → 模糊补全 → 目录时自动找内部文件。

    返回 (resolved_path, error_message)
    """
    from agent.security import safe_path

    if not path:
        return "", ""
    ok, value = safe_path(path, prefer=prefer)
    if not ok:
        return "", str(value)
    if not os.path.exists(value):
        return "", f"文件不存在: {path}"
    if os.path.isdir(value):
        hits = []
        for root, _dirs, files in os.walk(value):
            for f in files:
                if not f.lower().endswith((".mat", ".npy")):
                    continue
                full = os.path.join(root, f)
                if prefer and prefer.lower() in full.lower():
                    return full, ""
                hits.append(full)
            if hits:
                break
        if hits:
            return sorted(hits)[0], ""
        return "", f"目录中没有找到 .mat/.npy 数据文件: {value}"
    return value, ""


def check_data_ready(fc_path: str = "", bold_path: str = "") -> dict:
    """校验 FC / BOLD 文件是否可用于预测"""
    result = {"success": True, "fc": {}, "bold": {}, "ready": False}
    try:
        for key, path in (("fc", fc_path), ("bold", bold_path)):
            if not path:
                continue
            resolved, error = resolve_data_file(path, prefer=key)
            if error:
                result[key] = {"exists": False, "error": error}
                continue
            try:
                data = load_matrix_file(resolved)
                result[key] = {
                    "exists": True,
                    "path": resolved,
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "has_nan": bool(np.any(np.isnan(np.asarray(data, dtype=np.float64)))),
                    "filesize_mb": round(os.path.getsize(resolved) / 1024 / 1024, 2),
                }
            except Exception as e:
                result[key] = {"exists": True, "path": resolved, "error": f"无法解析: {e}"}

        fc_ok = bool(result["fc"].get("exists") and result["fc"].get("shape"))
        bold_ok = bool(result["bold"].get("exists") and result["bold"].get("shape"))
        result["ready"] = fc_ok and bold_ok
        if not result["ready"]:
            issues = []
            if fc_path and not fc_ok:
                issues.append("FC 数据未就绪")
            if bold_path and not bold_ok:
                issues.append("BOLD 数据未就绪")
            result["issues"] = issues
        if fc_ok and bold_ok:
            result["node_match"] = result["fc"]["shape"][0] == result["bold"]["shape"][0]
        return result
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


# ==================== 工具 ====================

class PredictAdRiskTool(BaseTool):
    name = "predict_ad_risk"
    category = "predict"
    description = (
        "核心诊断工具：基于 fMRI 功能连接 (FC) 和 BOLD 信号，用 SA-STGCN 模型"
        "预测 MCI/AD 患者 4 年内认知进展风险。输入 .mat/.npy 文件路径，"
        "返回认知稳定/进展概率、预测标签、Top-5 贡献脑区与 case_id。"
        "这是唯一可信的诊断数据源，所有结论必须原样引用其返回值。"
        "若手上只有 BOLD 影像（.nii/.nii.gz），应先调 extract_features 得到这两个文件。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "fc_path": {"type": "string", "description": "功能连接矩阵文件路径 (.mat / .npy)"},
            "bold_path": {"type": "string", "description": "BOLD 时序信号文件路径 (.mat / .npy)"},
            "fc_var_name": {"type": "string", "description": ".mat 中 FC 变量名，.npy 留空", "default": ""},
            "bold_var_name": {"type": "string", "description": ".mat 中 BOLD 变量名，.npy 留空", "default": ""},
        },
        "required": ["fc_path", "bold_path"],
    }
    timeout = 300

    def validate(self, args: dict) -> tuple:
        """用户常常只给出一个文件路径 —— 先尝试推断另一个（FC ↔ BOLD）"""
        args = dict(args or {})
        self._inferred = ""
        if args.get("fc_path") and not args.get("bold_path"):
            guess = infer_counterpart(args["fc_path"], "bold")
            if guess:
                args["bold_path"] = guess
                self._inferred = f"bold_path ← {guess}"
        elif args.get("bold_path") and not args.get("fc_path"):
            guess = infer_counterpart(args["bold_path"], "fc")
            if guess:
                args["fc_path"] = guess
                self._inferred = f"fc_path ← {guess}"
        return super().validate(args)

    def run(self, fc_path: str, bold_path: str, fc_var_name: str = "",
            bold_var_name: str = "") -> dict:
        result = predict_ad_risk(fc_path, bold_path, fc_var_name, bold_var_name)
        if result.get("success") and getattr(self, "_inferred", ""):
            result["inferred_path"] = self._inferred
        return result

    def summarize(self, result: dict) -> str:
        if not result.get("success"):
            return f"❌ {str(result.get('error', ''))[:60]}"
        return f"{result.get('prediction_name', '')} 稳定{result.get('prob_stable')}% / 进展{result.get('prob_progress')}%"


class ExplainPredictionTool(BaseTool):
    name = "explain_prediction"
    category = "predict"
    description = (
        "对已完成的预测做梯度可解释性分析（compute_gradient_based_edge_importance），"
        "返回 Top-N 关键脑区（中文名 + 归一化重要性）与 Top-10 关键脑区连接。"
        "case_id 留空则自动使用最近一次预测。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "case_id": {"type": "string", "description": "病例ID（predict_ad_risk 返回），留空用最近一次", "default": ""},
            "top_k": {"type": "integer", "description": "返回脑区数量，默认 10", "default": 10},
        },
        "required": [],
    }
    timeout = 300

    def run(self, case_id: str = "", top_k: int = 10) -> dict:
        return explain_prediction(case_id=case_id, top_k=top_k)

    def summarize(self, result: dict) -> str:
        if not result.get("success"):
            return f"❌ {str(result.get('error', ''))[:60]}"
        top = result.get("top_nodes") or [{}]
        return f"Top1 脑区: {top[0].get('name', '-')}"


class ListCasesTool(BaseTool):
    name = "list_cases"
    category = "predict"
    description = "列出已完成预测的所有病例（case_id、预测结论、概率），供 explain / report 引用。"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs) -> dict:
        store = CaseStore()
        data = store.load()
        cases = [
            {
                "case_id": k,
                "created_at": v.get("created_at", ""),
                "prediction_name": v.get("prediction_name", ""),
                "pred_prob": v.get("pred_prob", 0),
                "fc_path": v.get("fc_path", ""),
                "bold_path": v.get("bold_path", ""),
                "has_report": bool(v.get("report_path")),
            }
            for k, v in sorted(data.items(), key=lambda kv: kv[1].get("created_at", ""), reverse=True)
        ]
        return {"success": True, "total": len(cases), "cases": cases}

    def summarize(self, result: dict) -> str:
        return f"共 {result.get('total', 0)} 个病例"


predict_tools = [PredictAdRiskTool(), ExplainPredictionTool(), ListCasesTool()]

__all__ = ["predict_tools", "predict_ad_risk", "explain_prediction", "compute_case_explanation",
           "run_prediction", "check_data_ready", "CaseStore", "load_matrix_file",
           "load_brain_names", "get_model", "get_device", "shared_model",
           "release_model", "install_model_cleanup", "infer_counterpart"]
