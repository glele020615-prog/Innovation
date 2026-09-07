"""
数据加载子系统：scan_history.json 的唯一 IO 入口
================================================
ScanStore 集中所有读写（UI 页面与 Agent 工具共用），
避免 scan_history.json 的路径 / 格式逻辑散落各处。
"""

import os
import re
import json
from datetime import datetime

from agent.util import PROJECT_ROOT, extract_subject_id
from agent.skills.base import BaseTool
from agent.security import safe_path_or_error

_HISTORY_NAME = "scan_history.json"


def resolve_history_path() -> str:
    """优先项目根，其次当前工作目录（兼容旧版 R_W_JSON 的相对路径写法）"""
    primary = os.path.join(PROJECT_ROOT, _HISTORY_NAME)
    if os.path.exists(primary):
        return primary
    legacy = os.path.join(os.getcwd(), _HISTORY_NAME)
    if os.path.exists(legacy):
        return legacy
    return primary


class ScanStore:
    """scan_history.json 的读写封装（线程不安全，调用方自行串行化）"""

    def __init__(self, path: str = ""):
        self.path = path or resolve_history_path()

    # ---------------- IO ----------------

    def load(self) -> list:
        if not os.path.exists(self.path):
            return []
        try:
            if os.path.getsize(self.path) == 0:
                return []
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []

    def save(self, items: list):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)

    # ---------------- 查询 ----------------

    def list(self, keyword: str = "", limit: int = 50) -> list:
        items = self.load()
        if keyword:
            kw = keyword.lower()
            items = [
                it for it in items
                if kw in str(it.get("patient_id", "")).lower()
                or kw in str(it.get("file_path", "")).lower()
                or kw in str(it.get("params", {}).get("FileName", "")).lower()
            ]
        return items[:limit] if limit > 0 else items

    def get(self, scan_id: str) -> dict:
        items = self.load()
        return self._match(items, scan_id)

    @staticmethod
    def _match(items: list, scan_id: str) -> dict:
        if not items:
            return {}
        sid = str(scan_id).strip()
        if not sid:
            return {}
        # 1) 序号：0 / #0 / scan-0
        m = re.fullmatch(r"#?(\d+)|scan-(\d+)", sid, flags=re.I)
        if m:
            idx = int(m.group(1) or m.group(2))
            if 0 <= idx < len(items):
                return items[idx]
        # 2) patient_id 精确
        for it in items:
            if str(it.get("patient_id", "")).lower() == sid.lower():
                return it
        # 3) 文件名 / 路径（含子串）
        for it in items:
            path = str(it.get("file_path", ""))
            fname = str(it.get("params", {}).get("FileName", ""))
            if sid.lower() in path.lower() or sid.lower() == fname.lower():
                return it
        # 4) 模糊包含
        for it in items:
            if sid.lower() in json.dumps(it, ensure_ascii=False).lower():
                return it
        return {}

    @staticmethod
    def public_view(item: dict, index: int = -1) -> dict:
        params = item.get("params", {}) or {}
        return {
            "scan_index": index,
            "scan_id": item.get("patient_id", ""),
            "date": item.get("date", "-"),
            "patient_id": item.get("patient_id", "-"),
            "file_name": params.get("FileName", os.path.basename(str(item.get("file_path", "")))),
            "file_path": item.get("file_path", ""),
            "dimensions": params.get("Dimensions", "-"),
            "voxel_size": params.get("Voxel Size", "-"),
            "tr": params.get("TR", "-"),
            "volumes": params.get("Volumes", "-"),
            "exists": os.path.exists(str(item.get("file_path", ""))),
        }

    # ---------------- 变更 ----------------

    def add(self, entry: dict) -> dict:
        items = self.load()
        items.append(entry)
        self.save(items)
        return entry

    def delete(self, scan_id: str) -> bool:
        items = self.load()
        target = self._match(items, scan_id)
        if not target:
            return False
        key = target.get("file_path", "")
        remained = [
            it for it in items
            if os.path.normpath(str(it.get("file_path", ""))) != os.path.normpath(str(key))
        ]
        if len(remained) == len(items):
            return False
        self.save(remained)
        return True


# ==================== 纯函数：读取 NIfTI 元信息 ====================

def read_nifti_info(path: str) -> dict:
    """读取 NIfTI 头信息，返回与 UI 一致的 params 结构"""
    import nibabel as nib

    img = nib.load(path)
    header = img.header
    dims = header.get_data_shape()
    zooms = header.get_zooms()
    return {
        "FileName": os.path.basename(path),
        "Dimensions": "x".join(str(d) for d in dims[:3]),
        "Voxel Size": "x".join(f"{z:.2f}" for z in zooms[:3]) + " mm" if len(zooms) >= 3 else "-",
        "TR": f"{zooms[3]:.2f} s" if len(zooms) > 3 else "N/A",
        "Volumes": str(dims[3]) if len(dims) > 3 else "1",
    }


def build_scan_entry(path: str, patient_id: str = "") -> dict:
    info = read_nifti_info(path)
    pid = patient_id.strip() or f"PAT_{datetime.now().strftime('%m%d%S')}"
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "patient_id": pid,
        "file_path": os.path.abspath(path),
        "params": info,
    }


# ==================== 工具 ====================

class ListScansTool(BaseTool):
    name = "list_scans"
    category = "scan"
    description = (
        "列出软件中已导入的所有扫描记录（来自 scan_history.json）。"
        "返回序号、患者ID、文件名、维度、TR 等信息。"
        "当用户问「有哪些数据」「有什么扫描」时调用。无需参数。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "按患者ID/文件名过滤，可留空", "default": ""},
            "limit": {"type": "integer", "description": "最多返回条数，默认 20", "default": 20},
        },
        "required": [],
    }

    def run(self, keyword: str = "", limit: int = 20) -> dict:
        store = ScanStore()
        items = store.list(keyword=keyword, limit=limit)
        scans = [ScanStore.public_view(it, i) for i, it in enumerate(items)]
        return {
            "success": True,
            "total": len(scans),
            "storage_path": store.path,
            "scans": scans,
            "hint": "用 get_scan(scan_id=序号或患者ID) 查看单条详情；"
                    "对 func/bold 影像可直接进入特征提取。",
        }

    def summarize(self, result: dict) -> str:
        return f"共 {result.get('total', 0)} 条扫描记录"


class GetScanTool(BaseTool):
    name = "get_scan"
    category = "scan"
    description = (
        "查看单条扫描记录的详细信息。scan_id 可以是 list_scans 返回的序号（如 0 / \"0\"）、"
        "患者ID（如 PAT_041721）或文件名。返回完整参数与文件是否存在。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "scan_id": {"type": "string", "description": "序号 / 患者ID / 文件名 / 路径"},
        },
        "required": ["scan_id"],
    }

    def run(self, scan_id: str) -> dict:
        store = ScanStore()
        item = store.get(scan_id)
        if not item:
            return {"success": False, "error": f"未找到扫描记录: {scan_id}",
                    "hint": "先用 list_scans 查看可用记录"}
        idx = store.load().index(item)
        view = ScanStore.public_view(item, idx)
        view["ready_for_feature_extraction"] = (
            view["exists"] and str(view["file_path"]).lower().endswith((".nii", ".nii.gz"))
        )
        return {"success": True, "scan": view}


class ImportNiftiTool(BaseTool):
    name = "import_nifti"
    category = "scan"
    description = (
        "导入一个 NIfTI 影像（.nii / .nii.gz）到系统：读取维度、体素大小、TR、时间点，"
        "并写入 scan_history.json。等价于在「数据加载」页点导入按钮。"
        "路径必须在已授权目录内（默认项目目录）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "NIfTI 文件绝对路径（.nii / .nii.gz）"},
            "patient_id": {"type": "string", "description": "患者ID，留空则自动生成", "default": ""},
        },
        "required": ["path"],
    }
    timeout = 60

    def validate_args(self, args: dict) -> tuple:
        try:
            args["path"] = safe_path_or_error(args["path"], must_exist=True)
        except ValueError as e:
            return args, str(e)
        if not str(args["path"]).lower().endswith((".nii", ".nii.gz")):
            return args, "仅支持 NIfTI 格式（.nii / .nii.gz）"
        return args, None

    def run(self, path: str, patient_id: str = "") -> dict:
        entry = build_scan_entry(path, patient_id)
        store = ScanStore()
        store.add(entry)
        return {
            "success": True,
            "summary": f"已导入 {entry['params']['FileName']}",
            "scan": ScanStore.public_view(entry, len(store.load()) - 1),
        }


class DeleteScanTool(BaseTool):
    name = "delete_scan"
    category = "scan"
    description = "从 scan_history.json 中删除一条扫描记录（只删记录，不删磁盘文件）。"
    parameters = {
        "type": "object",
        "properties": {
            "scan_id": {"type": "string", "description": "序号 / 患者ID / 文件名"},
        },
        "required": ["scan_id"],
    }

    def run(self, scan_id: str) -> dict:
        store = ScanStore()
        ok = store.delete(scan_id)
        if not ok:
            return {"success": False, "error": f"删除失败，未找到记录: {scan_id}"}
        return {"success": True, "summary": f"已删除记录 {scan_id}", "total": len(store.load())}


# ==================== 数据文件检索 ====================

_DATA_EXTS = (".mat", ".npy", ".nii", ".gz", ".tsv", ".json")
_MAX_WALK_FILES = 40000
# 产物 / 缓存目录不算"数据文件"，搜索时直接剪掉
_SKIP_DIRS = {"report_cache", "build", "dist", "__pycache__", ".git",
              "nilearn_cache", ".index", "logs", "sessions"}


def _kind_of(path: str) -> str:
    """粗判文件用途，帮 LLM 分清 FC / BOLD / 影像"""
    low = path.lower()
    base = os.path.basename(low)
    if low.endswith((".nii", ".nii.gz")):
        return "nifti"
    if "fc" in base or "/fc/" in low or "\\fc\\" in low:
        return "fc"
    if "bold" in low:
        return "bold"
    if low.endswith(".mat"):
        return "mat"
    if low.endswith(".npy"):
        return "npy"
    return "other"


def data_search_roots() -> list:
    """允许检索的数据根目录：项目 Data + 用户显式授权的目录。

    做了包含关系去重：PROJECT_ROOT 与 Data 同时被授权时只扫更小的那个，
    否则同一批文件会被扫两遍、结果里出现重复项。
    """
    from agent.security import load_allowed_roots, PROJECT_ROOT, CONFIG_DIR

    data_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "Data"))
    config_dir = os.path.abspath(CONFIG_DIR)

    candidates = []
    if os.path.isdir(data_dir):
        candidates.append(data_dir)
    for root in load_allowed_roots():
        abs_root = os.path.abspath(root)
        if os.path.isdir(abs_root) and abs_root != config_dir:
            candidates.append(abs_root)
    if not candidates and os.path.isdir(PROJECT_ROOT):
        candidates.append(os.path.abspath(PROJECT_ROOT))

    # 按"短 → 长"归并成最小不重叠集合：
    # 大的 root 先收录，被它完全覆盖的小 root 直接丢弃，避免同一批文件扫两遍
    roots: list = []
    for path in sorted(set(candidates), key=len):
        if any(path == kept or path.startswith(kept + os.sep) for kept in roots):
            continue
        roots = [kept for kept in roots if not kept.startswith(path + os.sep)]
        roots.append(path)

    # Data 目录优先，命中结果才会排在前面
    roots.sort(key=lambda p: (0 if p == data_dir else 1, len(p)))
    return roots


def find_data_files(query: str, kind: str = "", max_results: int = 10) -> list:
    """按关键字搜索数据文件（大小写不敏感），返回 [{path, kind, size_mb}]"""
    query = str(query or "").strip().lower()
    if not query:
        return []
    want = str(kind or "any").lower().strip()
    hits = []
    scanned = 0
    for root in data_search_roots():
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS]
            for name in files:
                scanned += 1
                if scanned > _MAX_WALK_FILES:
                    return hits
                if not name.lower().endswith(_DATA_EXTS):
                    continue
                full = os.path.join(current, name)
                if query not in full.lower():
                    continue
                k = _kind_of(full)
                if want and want != "any" and k != want:
                    continue
                try:
                    size_mb = round(os.path.getsize(full) / 1024 / 1024, 2)
                except Exception:
                    size_mb = 0.0
                hits.append({"path": full, "kind": k, "size_mb": size_mb})
                if len(hits) >= max_results:
                    return hits
    return hits


class FindDataFilesTool(BaseTool):
    name = "find_data_files"
    category = "scan"
    description = (
        "按关键字在 Data 目录（及已授权目录）里搜索数据文件，返回真实路径。"
        "**当用户只给了样本 ID / 受试者编号而没给完整路径时，必须先调这个工具**，"
        "不要凭猜测拼路径去反复试 import_nifti 或 predict_ad_risk。"
        "关键字可以是样本编号（002_S_0729）、子目录名（pMCI）、文件名片段（matr_002）。"
        "返回结果的 kind 字段标明 fc / bold / nifti / mat / npy；"
        "拿到 FC 与 BOLD 两个路径后即可走 check_ready → predict_ad_risk。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键字，如 002_S_0729、sub-01、pMCI"},
            "kind": {"type": "string",
                     "description": "限定类型：fc / bold / nifti / mat / npy / any，默认 any",
                     "default": "any"},
            "max_results": {"type": "integer", "description": "最多返回条数，默认 10", "default": 10},
        },
        "required": ["query"],
    }
    timeout = 60

    def run(self, query: str, kind: str = "any", max_results: int = 10) -> dict:
        if not str(query or "").strip():
            return {"success": False, "error": "query 不能为空"}
        matches = find_data_files(query, kind=kind, max_results=max_results)
        if not matches:
            return {
                "success": True,
                "total": 0,
                "matches": [],
                "searched_roots": data_search_roots(),
                "hint": ("没有找到匹配的数据文件。请确认关键字，"
                         "或确认该目录已通过「授权数据目录」加入白名单。"),
            }
        fc = [m for m in matches if m["kind"] == "fc"]
        bold = [m for m in matches if m["kind"] == "bold"]
        if fc and bold:
            hint = (f"已同时找到 FC 与 BOLD，可直接 "
                    f"check_ready(fc_path='{fc[0]['path']}', bold_path='{bold[0]['path']}') "
                    f"→ predict_ad_risk。")
            next_step = "check_ready"
        else:
            hint = "只找到部分文件；缺 FC 或 BOLD 时再调一次并指定 kind 参数。"
            next_step = "extract_features" if not (fc and bold) else "check_ready"
        return {
            "success": True,
            "total": len(matches),
            "matches": matches,
            "has_fc": bool(fc),
            "has_bold": bool(bold),
            "hint": hint,
            "next_step": next_step,
        }

    def summarize(self, result: dict) -> str:
        if not result.get("success"):
            return f"❌ {str(result.get('error', ''))[:60]}"
        total = result.get("total", 0)
        if not total:
            return "未找到匹配文件"
        return (f"找到 {total} 个文件"
                f"（FC:{'有' if result.get('has_fc') else '无'} "
                f"BOLD:{'有' if result.get('has_bold') else '无'}）")


scan_tools = [
    ListScansTool(),
    GetScanTool(),
    ImportNiftiTool(),
    DeleteScanTool(),
    FindDataFilesTool(),
]

__all__ = ["ScanStore", "scan_tools", "read_nifti_info", "build_scan_entry",
           "find_data_files", "data_search_roots"]
