import sys
import os

# 加入 backend/src 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "src"))

from agent.tools import deep_scan

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")
filepath = os.path.join(DOCS_DIR, "problem_contract.txt")

with open(filepath, "r", encoding="utf-8") as f:
    contract = f.read()

print("=" * 70)
print(f"输入文本长度: {len(contract)}")
print(f"前 100 字符:\n{contract[:100]}")
print(f"\n包含 '第一条': {'第一条' in contract}")
print(f"包含换行符 \\n: {chr(10) in contract}")
print("=" * 70)

result = deep_scan(contract)

print(f"\n返回结果:")
print(f"  summary: {result.get('summary')}")
print(f"  total: {result.get('total')}")
print(f"  high_count: {result.get('high_count')}")

if result.get("risk_items"):
    print(f"\n第一条 risk_item 结构:")
    print(f"  clause: {result['risk_items'][0].get('clause', '')[:80]}")
    print(f"  risk: {result['risk_items'][0].get('risk')}")
    print(f"  source: {result['risk_items'][0].get('source', '未标注')}")