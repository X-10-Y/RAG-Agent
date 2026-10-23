"""
测试集评估脚本
用法：
    1. 先启动后端：python main.py
    2. 在 tests 目录下运行：python evaluate.py
"""
import json
import os
import ast
import requests
import sys

# ==================== 配置 ====================
API_URL = "http://127.0.0.1:8000"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "test_dataset.json")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_answer(answer):
    """
    解析 Agent 返回的 answer 字段，兼容三种格式：
    1. 标准 JSON（双引号）
    2. Python 字典字符串（单引号）
    3. 纯文本
    """
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str):
        return answer

    stripped = answer.strip()
    if not stripped.startswith("{"):
        return answer

    # 尝试1：标准 JSON
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 尝试2：ast 解析 Python 字典格式
    try:
        return ast.literal_eval(stripped)
    except Exception:
        pass

    # 都失败，返回原始文本
    return answer


def call_deep_scan(contract_text: str) -> dict:
    """调用 /chat 走 deep_scan 工具"""
    payload = {
        "user_input": "帮我仔细审查这份合同",
        "contract_text": contract_text,
        "history": []
    }
    r = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def evaluate_one(item: dict) -> dict:
    """评估单份合同"""
    filepath = os.path.join(DOCS_DIR, item["file"])
    if not os.path.exists(filepath):
        return {
            "id": item["id"],
            "label": item.get("label", ""),
            "status": "文件缺失",
            "hit": 0,
            "total": len(item["expected_risks"]),
            "tool_used": "-",
        }

    with open(filepath, "r", encoding="utf-8") as f:
        contract_text = f.read()

    try:
        result = call_deep_scan(contract_text)
        answer = result.get("answer", "")
        parsed = parse_answer(answer)

        # 解析成功：拿到 risk_items 列表
        if isinstance(parsed, dict):
            risk_items = parsed.get("risk_items", [])
        else:
            risk_items = []

        # 把所有 risk_items 的文本拼起来做关键词匹配
        all_text = " ".join([
            str(r.get("clause", "")) + " " + str(r.get("explanation", ""))
            for r in risk_items
        ])

        # 建立 clause_keyword -> 标注 risk 的映射，以及系统 risk 列表
        keyword_to_label_risk = {r["clause_keyword"]: r.get("risk") for r in item["expected_risks"]}
        keyword_to_sys_risk = {}
        for ri in risk_items:
            clause = str(ri.get("clause", ""))
            for kw in keyword_to_label_risk:
                if kw in clause or kw in str(ri.get("explanation", "")):
                    keyword_to_sys_risk[kw] = ri.get("risk")
                    break

        # 计算召回命中（关键词匹配）
        hit = 0
        risk_match = 0
        risk_checked = 0
        for expected in item["expected_risks"]:
            keyword = expected["clause_keyword"]
            if keyword in all_text:
                hit += 1
            # 只有当系统也返回了这个条款时，才对比 risk 等级
            if keyword in keyword_to_sys_risk:
                risk_checked += 1
                if keyword_to_sys_risk[keyword] == expected.get("risk"):
                    risk_match += 1

        return {
            "id": item["id"],
            "label": item.get("label", ""),
            "status": "成功",
            "hit": hit,
            "total": len(item["expected_risks"]),
            "risk_match": risk_match,
            "risk_checked": risk_checked,
            "tool_used": result.get("tool_used", "未知"),
            "high_count": parsed.get("high_count", 0) if isinstance(parsed, dict) else 0,
        }

    except Exception as e:
        return {
            "id": item["id"],
            "label": item.get("label", ""),
            "status": f"失败: {e}",
            "hit": 0,
            "total": len(item["expected_risks"]),
            "tool_used": "-",
        }


def main():
    dataset = load_dataset()
    print(f"📊 开始评估，共 {len(dataset)} 份合同\n")
    print("=" * 70)

    total_hit = 0
    total_expected = 0
    total_risk_match = 0
    total_risk_checked = 0
    results = []

    for item in dataset:
        r = evaluate_one(item)
        results.append(r)
        total_hit += r["hit"]
        total_expected += r["total"]
        total_risk_match += r.get("risk_match", 0)
        total_risk_checked += r.get("risk_checked", 0)
        rate = r["hit"] / r["total"] * 100 if r["total"] > 0 else 100
        risk_acc = r["risk_match"] / r["risk_checked"] * 100 if r["risk_checked"] > 0 else 100
        print(f"[{r['id']}] {r.get('label', '')}")
        print(f"     状态: {r['status']} | 命中: {r['hit']}/{r['total']} ({rate:.0f}%) | 风险等级: {r['risk_match']}/{r['risk_checked']} ({risk_acc:.0f}%) | 工具: {r.get('tool_used', '-')}")
        print("-" * 70)

    overall_rate = total_hit / total_expected * 100 if total_expected > 0 else 0
    overall_risk_acc = total_risk_match / total_risk_checked * 100 if total_risk_checked > 0 else 0
    print(f"\n✅ 总体召回命中: {total_hit}/{total_expected} = {overall_rate:.1f}%")
    print(f"✅ 风险等级准确率: {total_risk_match}/{total_risk_checked} = {overall_risk_acc:.1f}%")

    # 保存结果
    out_path = os.path.join(os.path.dirname(__file__), "eval_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_hit_rate": round(overall_rate, 2),
            "total_hit": total_hit,
            "total_expected": total_expected,
            "overall_risk_accuracy": round(overall_risk_acc, 2),
            "total_risk_match": total_risk_match,
            "total_risk_checked": total_risk_checked,
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print(f"📁 详细结果已保存到: {out_path}")


if __name__ == "__main__":
    main()