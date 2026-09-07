"""
预处理子系统
================
- run_fmriprep_preprocess：生成 fMRIPrep（Docker）命令并记录 todo；
  真正的 fMRIPrep 一次要跑几十分钟，不适合塞进 Agent 的 30s 工具预算，
  所以默认 dry_run，把可执行命令交给用户在预处理页或终端执行。
- run_custom_preprocess：基于 nilearn 的轻量版，支持平滑 / 去线性漂移 /
  带通滤波 / 空间标准化 / 重采样；头动校正与时间层校正需要 SPM/AFNI，标记为待办。
"""

import os
import json
import time
import shutil
import subprocess

import numpy as np

from agent.util import PROJECT_ROOT, slugify, extract_subject_id
from agent.skills.base import BaseTool
from agent.security import safe_path_or_error, CONFIG_DIR

TODO_FILE = os.path.join(CONFIG_DIR, "todo.json")

# 步骤别名 → 内部标准名
STEP_ALIASES = {
    "平滑": "smooth", "空间平滑": "smooth", "smooth": "smooth", "smoothing": "smooth",
    "去线性漂移": "detrend", "detrend": "detrend", "去漂移": "detrend",
    "滤波": "bandpass", "带通滤波": "bandpass", "bandpass": "bandpass", "filter": "bandpass",
    "空间标准化": "standardize", "标准化": "standardize", "standardize": "standardize",
    "mni": "standardize", "normalize": "standardize",
    "重采样": "resample", "resample": "resample",
    "头动校正": "motion_correction", "motion": "motion_correction", "realign": "motion_correction",
    "时间层校正": "slice_timing", "slice_timing": "slice_timing", "stc": "slice_timing",
    "zscore": "zscore", "标准化信号": "zscore",
}

NEEDS_EXTERNAL_TOOL = {
    "motion_correction": "需要 SPM / AFNI / FSL 做刚体配准，nilearn 无内置实现",
    "slice_timing": "需要切片采集顺序元数据，建议交给 fMRIPrep 完成",
}


def _append_todo(kind: str, payload: dict):
    todo = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind}
    todo.update(payload)
    try:
        items = []
        if os.path.exists(TODO_FILE):
            with open(TODO_FILE, "r", encoding="utf-8") as f:
                items = json.load(f)
        items.append(todo)
        with open(TODO_FILE, "w", encoding="utf-8") as f:
            json.dump(items[-200:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return todo


def normalize_steps(steps) -> list:
    if isinstance(steps, str):
        steps = [s.strip() for s in steps.replace(",", " ").split() if s.strip()]
    out = []
    for s in steps or []:
        key = str(s).strip().lower()
        std = STEP_ALIASES.get(key, key)
        if std not in out:
            out.append(std)
    return out


def docker_available() -> bool:
    return shutil.which("docker") is not None


def run_fmriprep_preprocess(bids_dir: str, subject_id: str, output_dir: str,
                            dry_run: bool = True, log=None) -> dict:
    """生成（可选执行）fMRIPrep 命令。返回命令 + todo 记录。"""
    logs = []

    def emit(m):
        logs.append(str(m))
        if log:
            log(str(m))

    bids_dir = safe_path_or_error(bids_dir, must_exist=True)
    output_dir = safe_path_or_error(output_dir)
    if not subject_id.startswith("sub-"):
        subject_id = f"sub-{slugify(subject_id)}"

    sub_dir = os.path.join(bids_dir, subject_id)
    if not os.path.isdir(sub_dir):
        return {"success": False, "error": f"BIDS 目录下找不到被试目录: {sub_dir}",
                "hint": "先完成 DICOM→NIfTI 转换并组织成 BIDS 结构", "log": logs}

    os.makedirs(output_dir, exist_ok=True)
    fs_license = os.path.join(PROJECT_ROOT, "FreeSurfer", "license.txt")
    if not os.path.exists(fs_license):
        for cand in ("license.txt", os.path.join("FreeSurfer", "license.txt")):
            p = os.path.join(PROJECT_ROOT, cand)
            if os.path.exists(p):
                fs_license = p
                break

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{bids_dir}:/data:ro",
        "-v", f"{output_dir}:/out",
    ]
    if os.path.exists(fs_license):
        cmd += ["-v", f"{fs_license}:/opt/freesurfer/license.txt"]
    cmd += [
        "nipreps/fmriprep:latest",
        "/data", "/out", "participant",
        "--participant-label", subject_id.replace("sub-", ""),
        "--output-spaces", "MNI152NLin2009cAsym",
        "--fs-no-reconall" if not os.path.exists(fs_license) else "--fs-license-file",
        "/opt/freesurfer/license.txt" if os.path.exists(fs_license) else "",
    ]
    cmd = [c for c in cmd if c]

    emit(f"fMRIPrep 命令已生成（dry_run={dry_run}）")
    emit(" ".join(cmd))

    if not docker_available():
        emit("[提示] 未检测到 docker，无法在此环境执行 fMRIPrep")

    todo = _append_todo("fmriprep", {
        "bids_dir": bids_dir, "subject_id": subject_id,
        "output_dir": output_dir, "command": " ".join(cmd),
        "docker_available": docker_available(),
    })

    result = {
        "success": True,
        "mode": "dry_run" if dry_run else "executed",
        "command": " ".join(cmd),
        "subject_id": subject_id,
        "output_dir": output_dir,
        "docker_available": docker_available(),
        "log": logs,
        "todo_id": todo["ts"],
    }

    if not dry_run:
        if not docker_available():
            return {"success": False, "error": "未检测到 docker，无法执行 fMRIPrep",
                    "command": result["command"], "log": logs}
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            result["returncode"] = proc.returncode
            result["stdout_tail"] = proc.stdout[-2000:]
            result["success"] = proc.returncode == 0
            result["summary"] = "fMRIPrep 执行完成" if proc.returncode == 0 else "fMRIPrep 失败"
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "fMRIPrep 超时（>60min），已中断", "log": logs}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}", "log": logs}

    result["summary"] = "已生成 fMRIPrep 命令（未执行）"
    result["next_step"] = "在「预处理」页执行，或把命令复制到装了 docker 的机器上跑；完成后回到 extract_features"
    return result


def run_custom_preprocess(nifti_path: str, steps, output_dir: str = "",
                          tr: float = 0.0, fwhm: float = 6.0,
                          high_pass: float = 0.01, low_pass: float = 0.1,
                          log=None) -> dict:
    """nilearn 轻量预处理。返回 applied / skipped / output_path。"""
    logs = []

    def emit(m):
        logs.append(str(m))
        if log:
            log(str(m))

    try:
        import nibabel as nib
        from nilearn import image as nimage
    except ImportError as e:
        return {"success": False, "error": f"缺少影像处理依赖: {e}", "log": logs}

    try:
        nifti_path = safe_path_or_error(nifti_path, must_exist=True)
    except ValueError as e:
        return {"success": False, "error": str(e), "log": logs}

    steps = normalize_steps(steps)
    if not steps:
        return {"success": False, "error": "steps 为空，至少给一个步骤",
                "supported": sorted(set(STEP_ALIASES.values())), "log": logs}

    output_dir = output_dir or os.path.join(PROJECT_ROOT, "Data", "preprocessed")
    try:
        output_dir = safe_path_or_error(output_dir)
    except ValueError as e:
        return {"success": False, "error": str(e), "log": logs}
    os.makedirs(output_dir, exist_ok=True)

    applied, skipped = [], []
    emit(f"[0] 读取影像: {nifti_path}")
    img = nib.load(nifti_path)
    emit(f"    形状: {img.shape}")

    if tr <= 0:
        tr = float(img.header.get_zooms()[3]) if len(img.header.get_zooms()) > 3 else 0.0
        if tr <= 0:
            emit("[警告] 未能获取 TR，滤波步骤将被跳过（可在参数里显式给 tr）")

    for step in steps:
        if step in NEEDS_EXTERNAL_TOOL:
            skipped.append({"step": step, "reason": NEEDS_EXTERNAL_TOOL[step]})
            emit(f"[跳过] {step}: {NEEDS_EXTERNAL_TOOL[step]}")
            continue
        try:
            if step == "smooth":
                emit(f"[处理] 空间平滑 fwhm={fwhm}mm")
                img = nimage.smooth_img(img, fwhm=fwhm)
                applied.append("smooth")
            elif step == "detrend":
                emit("[处理] 去线性漂移")
                img = nimage.clean_img(img, detrend=True, standardize=None)
                applied.append("detrend")
            elif step == "bandpass":
                if tr <= 0:
                    skipped.append({"step": "bandpass", "reason": "缺少 TR"})
                    emit("[跳过] bandpass: 缺少 TR")
                    continue
                emit(f"[处理] 带通滤波 high_pass={high_pass} low_pass={low_pass}")
                img = nimage.clean_img(img, detrend=True, standardize=None,
                                       high_pass=high_pass, low_pass=low_pass, t_r=tr)
                applied.append("bandpass")
            elif step in ("standardize", "resample"):
                emit("[处理] 重采样到 MNI152 模板")
                try:
                    from nilearn.datasets import load_mni152_template
                    target = load_mni152_template(resolution=2)
                    img = nimage.resample_to_img(img, target, interpolation="continuous")
                    applied.append("standardize")
                except Exception as e:
                    skipped.append({"step": step, "reason": f"MNI 模板不可用: {e}"})
                    emit(f"[跳过] {step}: MNI 模板不可用（{e}）")
            elif step == "zscore":
                emit("[处理] 信号 z-score 标准化")
                img = nimage.clean_img(img, detrend=None, standardize="zscore_sample")
                applied.append("zscore")
            else:
                skipped.append({"step": step, "reason": "未知步骤"})
                emit(f"[跳过] 未知步骤: {step}")
        except Exception as e:
            skipped.append({"step": step, "reason": f"{type(e).__name__}: {e}"})
            emit(f"[失败] {step}: {e}")

    stem = slugify(extract_subject_id(nifti_path))
    suffix = "_preproc" if applied else "_copy"
    out_path = os.path.join(output_dir, f"{stem}{suffix}.nii.gz")
    nib.save(img, out_path)
    emit(f"[输出] {out_path}")

    if skipped:
        _append_todo("preprocess_skipped", {
            "nifti_path": nifti_path, "skipped": skipped,
        })

    return {
        "success": bool(applied),
        "applied": applied,
        "skipped": skipped,
        "output_path": out_path,
        "shape": list(img.shape),
        "log": logs,
        "summary": f"已完成 {len(applied)} 步，跳过 {len(skipped)} 步",
        "next_step": f"调用 extract_features(bold_path='{out_path}')",
    }


# ==================== 工具 ====================

class FmriprepPreprocessTool(BaseTool):
    name = "run_fmriprep_preprocess"
    category = "preprocess"
    description = (
        "为 BIDS 数据集中的某个被试生成 fMRIPrep 预处理命令（Docker 版）并记录待办。"
        "默认 dry_run=true 只返回命令不执行——fMRIPrep 单次耗时数十分钟，"
        "应在「预处理」页或终端执行。需要目录已按 BIDS 组织（含 sub-xx）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "bids_dir": {"type": "string", "description": "BIDS 根目录（含 sub-xx）"},
            "subject_id": {"type": "string", "description": "被试ID，如 sub-01 或 01"},
            "output_dir": {"type": "string", "description": "fMRIPrep 输出目录"},
            "dry_run": {"type": "boolean", "description": "true=只生成命令；false=直接执行", "default": True},
        },
        "required": ["bids_dir", "subject_id", "output_dir"],
    }
    timeout = 60

    def validate_args(self, args: dict) -> tuple:
        for key in ("bids_dir", "output_dir"):
            try:
                args[key] = safe_path_or_error(args[key], must_exist=(key == "bids_dir"))
            except ValueError as e:
                return args, str(e)
        return args, None

    def run(self, bids_dir: str, subject_id: str, output_dir: str, dry_run: bool = True) -> dict:
        return run_fmriprep_preprocess(bids_dir, subject_id, output_dir, dry_run=dry_run)


class CustomPreprocessTool(BaseTool):
    name = "run_custom_preprocess"
    category = "preprocess"
    description = (
        "对单个 NIfTI 做轻量预处理（基于 nilearn，秒级完成）。"
        "支持步骤：smooth(平滑)、detrend(去线性漂移)、bandpass(带通滤波)、"
        "standardize(重采样到MNI152)、resample、zscore。"
        "不支持：motion_correction(头动校正)、slice_timing(时间层校正)——需要 SPM/AFNI/fMRIPrep。"
        "返回处理后的 nii.gz 路径，之后接 extract_features。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "nifti_path": {"type": "string", "description": "NIfTI 影像路径 (.nii / .nii.gz)"},
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "处理步骤列表，如 ['detrend','bandpass','smooth']",
            },
            "output_dir": {"type": "string", "description": "输出目录，默认 Data/preprocessed", "default": ""},
            "tr": {"type": "number", "description": "重复时间(秒)，留空则从影像头读取", "default": 0},
        },
        "required": ["nifti_path", "steps"],
    }
    timeout = 600

    def validate_args(self, args: dict) -> tuple:
        try:
            args["nifti_path"] = safe_path_or_error(args["nifti_path"], must_exist=True)
        except ValueError as e:
            return args, str(e)
        if isinstance(args.get("steps"), str):
            args["steps"] = normalize_steps(args["steps"])
        if not args.get("steps"):
            return args, "steps 不能为空"
        return args, None

    def run(self, nifti_path: str, steps, output_dir: str = "", tr: float = 0.0) -> dict:
        return run_custom_preprocess(nifti_path, steps, output_dir=output_dir, tr=tr)


preprocess_tools = [FmriprepPreprocessTool(), CustomPreprocessTool()]

__all__ = ["preprocess_tools", "run_fmriprep_preprocess", "run_custom_preprocess",
           "normalize_steps", "docker_available"]
