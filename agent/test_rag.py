"""RAG 索引 / 检索回归测试（P2 验收）"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.rag import get_rag, rebuild_index
from agent.knowledge.build_index import build_index, load_index, is_stale
from agent.tools import TOOLS, TOOL_MAP, TOOL_SCHEMAS, describe_groups

print("=== 1. 索引构建 ===")
meta = build_index(force=True)
print(json.dumps(meta, ensure_ascii=False, indent=2)[:600])

chunks, vectors, loaded_meta = load_index()
print(f"chunks={len(chunks) if chunks else 0} vectors={vectors.shape if vectors is not None else None} "
      f"stale={is_stale(loaded_meta)}")

rag = get_rag()
print("\n=== 2. 引擎状态 ===")
print(json.dumps(rag.status, ensure_ascii=False, indent=2))

print("\n=== 3. 固定 query 回归（top-1 文件名 + 段落）===")
FIXED_QUERIES = [
    "MCI 诊断标准",
    "海马 AD",
    "随访频率",
    "Lecanemab 适应症",
    "aMCI vs naMCI",
]
for q in FIXED_QUERIES:
    r = rag.search(q, top_k=3)
    if not r.get("results"):
        print(f"  {q:<20} -> 未覆盖（{r.get('message', '')[:40]}）")
        continue
    top = r["results"][0]
    print(f"  {q:<20} -> [{top['score']:.3f}] {top['source_file']}#{top['section'][:30]}")

print("\n=== 4. 阈值过滤 ===")
r = rag.search("今天股票涨了吗", top_k=3)
print(f"  无关 query: results={len(r.get('results', []))} covered={r.get('covered')}")

print("\n=== 5. 工具注册表 ===")
print(f"  工具总数: {len(TOOLS)}  schema 数: {len(TOOL_SCHEMAS)}")
assert "search_knowledge" in TOOL_MAP
assert "get_region_info" in TOOL_MAP
assert len(TOOL_SCHEMAS) == len([t for t in TOOLS if t.visible])
print(describe_groups())
print("\nAll checks passed.")
