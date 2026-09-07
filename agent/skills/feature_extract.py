"""
特征提取子系统（纯函数 + Agent 工具）
=====================================
核心是 run_feature_extraction()：不依赖 Qt，Agent 与 UI 共用。
feature_extraction_page.py 的 QThread 只负责把进度/日志转发成信号。
"""

import os
import traceback

import numpy as np

from agent.util import PROJECT_ROOT, extract_subject_id, slugify
from agent.skills.base import BaseTool
from agent.security import safe_path_or_error

SUPPORTED_ATLASES = ("AAL116",)


# ==================== 图谱定位 ====================

def local_atlas_candidates() -> list:
    """按优先级返回本地 AAL116 图谱候选路径（标准 MNI 空间，标签 0-116）"""
    return [
        (os.path.join(PROJECT_ROOT, "AAL", "aal_MNI.nii"),
         "高分辨率 AAL116 (AAL/aal_MNI.nii, 1mm MNI)"),
    ]


def locate_atlas(atlas: str = "AAL116", source: str = "auto", log=None) -> tuple:
    """定位脑图谱，返回 (path, description)"""
    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(f"暂不支持的脑图谱: {atlas}（当前支持 {', '.join(SUPPORTED_ATLASES)}）")

    def emit(msg):
        if log:
            log(msg)

    if source == "nilearn":
        from nilearn.datasets import fetch_atlas_aal
        aal = fetch_atlas_aal(version="SPM12")
        maps = aal.get("maps", "")
        if maps and os.path.exists(maps):
            return maps, "nilearn 内置 AAL116 (SPM12)"
        raise FileNotFoundError("nilearn 内置 AAL116 不可用")

    for path, desc in local_atlas_candidates():
        if os.path.exists(path):
            return path, desc

    if source in ("auto", "nilearn"):
        try:
            from nilearn.datasets import fetch_atlas_aal
            aal = fetch_atlas_aal(version="SPM12")
            maps = aal.get("maps", "")
            if maps and os.path.exists(maps):
                return maps, "nilearn 内置 AAL116 (SPM12)"
        except Exception as e:
            emit(f"[警告] nilearn 内置 AAL 加载失败: {e}")

    raise FileNotFoundError("无法定位 AAL116 图谱（本地与 nilearn 内置均不可用）")


# ==================== 纯函数：一次完整特征提取 ====================

def run_feature_extraction(bold_path: str, output_dir: str, subject_id: str = "",
                           atlas: str = "AAL116", standardize: bool = True,
                           atlas_source: str = "auto",
                           log=None, progress=None, save: bool = True) -> dict:
    """从 BOLD 影像提取 AAL116 时间序列与 FC 矩阵。

    返回 {"success", "bold_npy", "fc_npy", "bold", "fc", "log", "shape"}
    bold: (T, 116)  fc: (116, 116)
    """
    logs: list = []

    def emit(msg):
        logs.append(str(msg))
        if log:
            log(str(msg))

    def tick(value):
        if progress:
            progress(int(value))

    try:
        import nibabel as nib
        from nilearn import image as nimage
        from nilearn.maskers import NiftiLabelsMasker

        emit("=" * 60)
        emit(f"开始特征提取: {subject_id or extract_subject_id(bold_path)}")
        emit(f"输入 BOLD: {bold_path}")
        emit("=" * 60)

        if not os.path.exists(bold_path):
            raise FileNotFoundError(f"BOLD 文件不存在: {bold_path}")

        os.makedirs(output_dir, exist_ok=True)
        tick(5)

        emit("[1/4] 读取 BOLD 影像...")
        bold_img = nib.load(bold_path)
        emit(f"      BOLD 形状: {bold_img.shape}")
        tick(10)

        emit("[2/4] 定位 AAL116 图谱并重采样到 BOLD 空间...")
        atlas_path, atlas_desc = locate_atlas(atlas, atlas_source, log=emit)
        emit(f"      使用图谱: {atlas_desc}")
        tick(20)

        atlas_img = nib.load(atlas_path)
        if (atlas_img.shape[:3] != bold_img.shape[:3]
                or not np.allclose(atlas_img.affine, bold_img.affine)):
            atlas_resampled = nimage.resample_to_img(atlas_img, bold_img, interpolation="nearest")
        else:
            atlas_resampled = atlas_img

        emit("[3/4] 提取 AAL116 脑区 BOLD 时间序列...")
        masker = NiftiLabelsMasker(
            labels_img=atlas_resampled,
            standardize="zscore_sample" if standardize else False,
            memory="nilearn_cache",
            memory_level=1,
            verbose=0,
        )
        time_series = masker.fit_transform(bold_img)   # (T, 116)
        emit(f"      时间序列形状: {time_series.shape} (时间点 x 脑区)")
        tick(55)

        emit("[4/4] 用皮尔逊相关系数构建 FC 矩阵...")
        fc = np.corrcoef(time_series.T)
        np.fill_diagonal(fc, 0.0)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        emit(f"      FC 矩阵形状: {fc.shape} (脑区 x 脑区)")
        tick(80)

        n_regions = int(time_series.shape[1])
        if atlas == "AAL116" and n_regions != 116:
            emit(f"[警告] 提取到 {n_regions} 个脑区，期望 116")

        bold_out = os.path.join(output_dir, f"{subject_id}_bold.npy")
        fc_out = os.path.join(output_dir, f"{subject_id}_fc.npy")
        if save:
            np.save(bold_out, time_series)
            np.save(fc_out, fc)
            emit(f"      已保存 BOLD: {bold_out}")
            emit(f"      已保存 FC:   {fc_out}")
        tick(100)
        emit("=" * 60)
        emit(f"✅ {subject_id} 特征提取完成")

        return {
            "success": True,
            "summary": f"{subject_id} 特征提取完成",
            "bold_npy": bold_out if save else "",
            "fc_npy": fc_out if save else "",
            "bold": time_series,
            "fc": fc,
            "log": logs,
            "shape": {
                "bold": list(time_series.shape),
                "fc": list(fc.shape),
                "atlas": atlas,
                "atlas_desc": atlas_desc,
            },
        }

    except Exception as e:
        emit(f"❌ 提取失败: {e}")
        emit(traceback.format_exc()[-1200:])
        return {"success": False, "error": f"{type(e).__name__}: {e}", "log": logs}


def default_output_dir() -> str:
    return os.path.join(PROJECT_ROOT, "Data", "features")


# ==================== 工具 ====================

class ExtractFeaturesTool(BaseTool):
    name = "extract_features"
    category = "feature"
    description = (
        "从预处理后的 BOLD 影像（.nii/.nii.gz）提取 AAL116 脑区时间序列和功能连接矩阵："
        "输出 {subject_id}_bold.npy (时间点 x 116) 与 {subject_id}_fc.npy (116 x 116)。"
        "这是进入 AI 预测的必经步骤。提取成功后应立刻接 predict_ad_risk。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "bold_path": {"type": "string", "description": "BOLD 影像路径 (.nii / .nii.gz)"},
            "output_dir": {"type": "string",
                           "description": "特征输出目录，留空则用 Data/features", "default": ""},
            "subject_id": {"type": "string",
                           "description": "被试ID，留空则从文件名推断", "default": ""},
            "atlas": {"type": "string", "description": "脑图谱，当前仅支持 AAL116", "default": "AAL116"},
        },
        "required": ["bold_path"],
    }
    timeout = 900   # 影像提取慢，单独放宽

    def validate_args(self, args: dict) -> tuple:
        try:
            args["bold_path"] = safe_path_or_error(args["bold_path"], must_exist=True)
        except ValueError as e:
            return args, str(e)
        out_dir = args.get("output_dir") or default_output_dir()
        try:
            args["output_dir"] = safe_path_or_error(out_dir)
        except ValueError as e:
            return args, str(e)
        if not args.get("subject_id"):
            args["subject_id"] = slugify(extract_subject_id(args["bold_path"]))
        return args, None

    def run(self, bold_path: str, output_dir: str = "", subject_id: str = "",
            atlas: str = "AAL116") -> dict:
        result = run_feature_extraction(
            bold_path=bold_path,
            output_dir=output_dir or default_output_dir(),
            subject_id=subject_id or slugify(extract_subject_id(bold_path)),
            atlas=atlas,
        )
        if not result.get("success"):
            return result
        # 不把大数组塞回上下文，只给路径和形状
        return {
            "success": True,
            "summary": result.get("summary", "特征提取完成"),
            "bold_npy": result["bold_npy"],
            "fc_npy": result["fc_npy"],
            "shape": result["shape"],
            "log_tail": result["log"][-6:],
            "next_step": f"调用 predict_ad_risk(fc_path='{result['fc_npy']}', bold_path='{result['bold_npy']}')",
        }


class CheckFeatureStatusTool(BaseTool):
    name = "check_features"
    category = "feature"
    description = (
        "检查某个被试的特征文件（_bold.npy / _fc.npy）是否已经存在且形状正确，"
        "避免重复跑耗时的特征提取。无需参数时会检查默认输出目录 Data/features。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string", "description": "被试ID，如 sub-01 或 002_S_0729", "default": ""},
            "output_dir": {"type": "string", "description": "特征目录，默认 Data/features", "default": ""},
        },
        "required": [],
    }

    def run(self, subject_id: str = "", output_dir: str = "") -> dict:
        out_dir = output_dir or default_output_dir()
        if not os.path.isdir(out_dir):
            return {"success": True, "exists": False, "output_dir": out_dir,
                    "files": [], "hint": "特征目录尚未创建"}
        hits = []
        for fname in sorted(os.listdir(out_dir)):
            if not fname.endswith(".npy"):
                continue
            if subject_id and subject_id not in fname:
                continue
            path = os.path.join(out_dir, fname)
            try:
                arr = np.load(path, mmap_mode="r")
                hits.append({"file": fname, "path": path, "shape": list(arr.shape)})
            except Exception as e:
                hits.append({"file": fname, "path": path, "error": str(e)})
        return {
            "success": True,
            "exists": len(hits) >= 2,
            "output_dir": out_dir,
            "files": hits,
            "hint": "" if hits else "尚无特征文件，需先跑 extract_features",
        }


feature_tools = [ExtractFeaturesTool(), CheckFeatureStatusTool()]

__all__ = ["run_feature_extraction", "locate_atlas", "local_atlas_candidates",
           "feature_tools", "default_output_dir"]
