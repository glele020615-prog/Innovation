"""工具层独立测试（不依赖 LLM / Qt）"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent.tools import (
    TOOLS, TOOL_MAP, TOOL_SCHEMAS, execute_tool, grouped_tools,
    list_scans, describe_pipeline, get_region_info, check_ready,
)
from agent.skills.scan import ScanStore
from agent.util import jsonable, extract_subject_id

print("=== 1. 工具注册表 ===")
print(f"共 {len(TOOLS)} 个工具 / {len(TOOL_SCHEMAS)} 个 schema")
for group, tools in grouped_tools().items():
    print(f"  [{group}] {', '.join(t.name for t in tools)}")

print("\n=== 2. list_scans（ScanStore）===")
store = ScanStore()
print(f"  存储文件: {store.path}")
r = list_scans()
print(f"  总数: {r['total']}")
for s in r["scans"]:
    print(f"  - {s['patient_id']}: {s['file_name']} ({s['dimensions']})")

print("\n=== 3. get_scan ===")
r = execute_tool("get_scan", {"scan_id": "0"})
print(f"  scan_id=0 -> success={r.get('success')} "
      f"{(r.get('scan') or {}).get('file_name', r.get('error'))}")

print("\n=== 4. describe_pipeline ===")
r = describe_pipeline()
for step in r["pipeline"]:
    print(f"  第{step['step']}步 [{step['page']}] {step['name']} -> {step['agent_tools']}")

print("\n=== 5. get_region_info（单一数据源）===")
for name in ["HIP.L", "PCG.R", "海马", "Hippocampus", "AMYG.L", "XXX.Z"]:
    r = get_region_info(name)
    print(f"  {name:<12} -> {r.get('cn_name')} | {str(r.get('ad_relevance'))[:36]}...")

print("\n=== 6. check_ready（含模糊路径解析 / 越界拒绝）===")
from agent.util import PROJECT_ROOT

tmp = os.path.join(PROJECT_ROOT, "_tmp_agent_test")
os.makedirs(tmp, exist_ok=True)
fc_path = os.path.join(tmp, "fc.npy")
bold_path = os.path.join(tmp, "bold.npy")
np.save(fc_path, np.random.randn(116, 116))
np.save(bold_path, np.random.randn(116, 130))
r = check_ready(fc_path, bold_path)
print(f"  合法数据: ready={r['ready']} node_match={r.get('node_match')}")
r = check_ready("Data/pMCI/.../matr_002_S_0729.mat", "Data/pMCI/.../matr_002_S_0729.mat")
print(f"  模糊路径: ready={r['ready']} fc={r['fc'].get('shape')} bold={r['bold'].get('shape')}")
r = check_ready("C:/Windows/System32/config", "")
print(f"  越界路径: ready={r['ready']} error={str(r['fc'].get('error'))[:60]}")

print("\n=== 7. 序列化（jsonable）===")
payload = {"arr": np.random.randn(3, 3), "path": __import__("pathlib").Path("a/b"),
           "s": {1, 2}, "np": np.float32(1.5), "nan": np.nan}
print(" ", json.dumps(jsonable(payload), ensure_ascii=False)[:160])

print("\n=== 8. 受试者 ID 提取 ===")
for p in ["Data/pMCI/FC/matr_002_S_0729_2011-08-16.mat", "sub-01_task-rest_bold.nii.gz"]:
    print(f"  {p} -> {extract_subject_id(p)}")

print("\nAll checks passed.")
