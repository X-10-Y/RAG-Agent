"""
误报率测试：在完全合规的合同上跑系统，统计系统报了多少假风险

理想结果：
    合规合同上高风险数 = 0
    误报率 = 0%

如果高风险数 > 0，说明系统在"没问题的地方乱报警"，需要修 Prompt
"""
import os
import json
import ast
import requests

# ==================== 配置 ====================
API_URL = "http://127.0.0.1:8000"
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")

# 合规合同列表（这些合同里，理论上有风险条款应该很少）
COMPLIANT_FILES = [
    "standard_contract.txt",
]


def parse_answer(answer):
    """兼容 JSON / Python dict 字符串"""
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str):
        return answer
    stripped = answer.strip()
    if not stripped.startswith("{"):
        return answer
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except Exception:
        pass
    return answer


def test_one(filename: str) -> dict:
    """测一份合规合同"""
    filepath = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(filepath):
        return {"file": filename, "status": "文件不存在"}

    with open(filepath, "r", encoding="utf-8") as f:
        contract = f.read()

    print(f"\n{'=' * 70}")
    print(f"📄 测试文件: {filename}")
    print(f"   字符数: {len(contract)}")
    print(f"{'=' * 70}")

    try:
        r = requests.post(f"{API_URL}/chat", json={
            "user_input": "帮我仔细审查这份合同",
            "contract_text": contract,
            "history": []
        }, timeout=300)

        result = r.json()
        answer = result.get("answer", "")
        parsed = parse_answer(answer)

        if not isinstance(parsed, dict):
            return {"file": filename, "status": "解析失败", "raw": str(answer)[:200]}

        total = parsed.get("total", 0)
        high = parsed.get("high_count", 0)
        medium = parsed.get("medium_count", 0)
        low = parsed.get("low_count", 0)
        risk_items = parsed.get("risk_items", [])

        # 误报率 = 高风险数 / 总条款数
        false_positive_rate = (high / total * 100) if total > 0 else 0

        print(f"\n📊 系统输出:")
        print(f"   总条款数: {total}")
        print(f"   🔴 高风险: {high}")
        print(f"   🟡 中风险: {medium}")
        print(f"   🟢 低风险: {low}")
        print(f"   误报率(高/总): {false_positive_rate:.1f}%")

        if high > 0:
            print(f"\n⚠️ 系统在合规合同上报了 {high} 条高风险（疑似误报）：")
            for item in risk_items:
                if item.get("risk") == "高":
                    print(f"   - [{item['risk']}] {item.get('clause', '')[:80]}...")
                    print(f"     理由: {item.get('explanation', '')[:120]}...")
        else:
            print(f"\n✅ 合规合同上无高风险误报")

        return {
            "file": filename,
            "status": "成功",
            "total": total,
            "high": high,
            "medium": medium,
            "low": low,
            "false_positive_rate": round(false_positive_rate, 2),
        }

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return {"file": filename, "status": f"失败: {e}"}


def main():
    print("🔍 误报率测试开始")
    print(f"📡 后端: {API_URL}")
    print(f"📂 文档目录: {DOCS_DIR}")
    print(f"📋 待测文件: {COMPLIANT_FILES}")

    results = []
    for filename in COMPLIANT_FILES:
        r = test_one(filename)
        results.append(r)

    # 汇总
    print(f"\n{'=' * 70}")
    print("📊 汇总")
    print(f"{'=' * 70}")

    total_high = sum(r.get("high", 0) for r in results if r.get("status") == "成功")
    total_clauses = sum(r.get("total", 0) for r in results if r.get("status") == "成功")

    if total_clauses > 0:
        overall_fp_rate = total_high / total_clauses * 100
        print(f"合规合同上的高风险总数: {total_high}")
        print(f"合规合同的总条款数: {total_clauses}")
        print(f"📉 误报率: {overall_fp_rate:.1f}%")
    else:
        print("⚠️ 没有成功测试的样本")

    # 保存结果
    out_path = os.path.join(os.path.dirname(__file__), "false_positive_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 结果已保存: {out_path}")


if __name__ == "__main__":
    main()