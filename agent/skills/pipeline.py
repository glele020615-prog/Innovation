"""
流程引导子系统：让 Agent 知道"系统长什么样、下一步干什么"
==========================================================
describe_pipeline 是 Agent 的地图：每一步都标注了可代替用户执行的工具。
"""

from agent.skills.base import BaseTool
from agent.skills.predict import check_data_ready
from agent.skills.knowledge import _KNOWLEDGE_HINT

PIPELINE = [
    {
        "step": 1, "page": "数据加载", "name": "导入 fMRI / DICOM 数据",
        "description": "导入 NIfTI (.nii/.nii.gz) 影像，读取维度/体素/TR/时间点，写入 scan_history.json。",
        "agent_tools": ["list_scans", "get_scan", "import_nifti", "delete_scan"],
        "what_you_need": "静息态 fMRI NIfTI 文件或 DICOM 序列",
    },
    {
        "step": 2, "page": "预处理", "name": "MRI 预处理",
        "description": (
            "fMRIPrep 标准化流程（需 Docker）或 nilearn 轻量流程"
            "（平滑 / 去线性漂移 / 带通滤波 / MNI 标准化）。"
            "头动校正与时间层校正需 fMRIPrep 完成。"
        ),
        "agent_tools": ["run_fmriprep_preprocess", "run_custom_preprocess"],
        "what_you_need": "已导入的 NIfTI 或 BIDS 目录",
    },
    {
        "step": 3, "page": "特征提取", "name": "提取 BOLD 时序与 FC 矩阵",
        "description": "基于 AAL116 图谱提取 116 个脑区时间序列，用皮尔逊相关构建 116x116 FC 矩阵。",
        "agent_tools": ["extract_features", "check_features"],
        "output_files": "<subject>_bold.npy (Tx116) + <subject>_fc.npy (116x116)",
    },
    {
        "step": 4, "page": "疾病预测诊断", "name": "SA-STGCN 风险预测",
        "description": "预测 4 年内认知进展风险，输出稳定/进展概率与 Top-5 贡献脑区。",
        "agent_tools": ["check_ready", "predict_ad_risk", "explain_prediction", "list_cases"],
        "what_you_need": "特征提取产出的 FC 与 BOLD 文件",
    },
    {
        "step": 5, "page": "个体报告分析", "name": "生成个体化报告",
        "description": "生成含风险结论、Top10 脑区/连接、三张可视化图的 PDF 报告。",
        "agent_tools": ["generate_report", "open_report"],
        "what_you_need": "已完成预测的 case_id",
    },
]


def describe_pipeline() -> dict:
    return {
        "success": True,
        "pipeline": PIPELINE,
        "knowledge_base": _KNOWLEDGE_HINT,
        "full_chain": (
            "list_scans → run_custom_preprocess → extract_features → "
            "check_ready → predict_ad_risk → explain_prediction → generate_report"
        ),
        "note": (
            "第 1/3/4/5 步我可以代你完整执行；第 2 步的 fMRIPrep 需 Docker，"
            "我只能生成命令（轻量步骤可直接跑）。"
            "用户直接给出 FC/BOLD 文件路径时，从 check_ready 开始即可。"
        ),
    }


# ==================== 工具 ====================

class DescribePipelineTool(BaseTool):
    name = "describe_pipeline"
    category = "guide"
    description = (
        "返回系统完整工作流程（5 个阶段）以及每个阶段我（AI 助手）能代执行的工具。"
        "当用户问「流程是什么」「怎么操作」「下一步该干什么」时调用。无需参数。"
    )
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs) -> dict:
        return describe_pipeline()

    def summarize(self, result: dict) -> str:
        return f"共 {len(result.get('pipeline', []))} 个阶段"


class CheckReadyTool(BaseTool):
    name = "check_ready"
    category = "guide"
    description = (
        "检查 FC / BOLD 数据文件是否存在、形状是否匹配、是否含 NaN，"
        "判断是否可以直接送进 predict_ad_risk。在预测前调用可避免无谓报错。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "fc_path": {"type": "string", "description": "FC 矩阵文件路径 (.mat/.npy)", "default": ""},
            "bold_path": {"type": "string", "description": "BOLD 时序文件路径 (.mat/.npy)", "default": ""},
        },
        "required": [],
    }
    timeout = 60

    def run(self, fc_path: str = "", bold_path: str = "") -> dict:
        return check_data_ready(fc_path, bold_path)

    def summarize(self, result: dict) -> str:
        if not result.get("success"):
            return f"❌ {str(result.get('error', ''))[:60]}"
        return "✅ 数据就绪" if result.get("ready") else f"未就绪: {', '.join(result.get('issues', []))}"


guide_tools = [DescribePipelineTool(), CheckReadyTool()]

__all__ = ["guide_tools", "describe_pipeline", "PIPELINE"]
