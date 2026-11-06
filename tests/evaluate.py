"""
测试集评估脚本
用法：
    1. 先启动后端：python main.py
    2. 在 tests 目录下运行：python evaluate.py
    3. 首次建立基线：python evaluate.py --update-baseline

指标口径：
    - recall        期望条款的关键词是否出现在返回文本里（召回）
    - risk_accuracy 等级准确率，分母是全部期望条款；漏报直接记为等级错误
    - precision     系统判为中/高的条款中，能对上任意期望条款的比例（过检率反面）
    - citation      返回的 clause_no 能在原文里找到的比例（引用可溯源）
"""
import ast
import hashlib
import json
import os
import sys
import time

import requests

# ==================== 配置 ====================
API_URL = "http://127.0.0.1:8000"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "test_dataset.json")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "eval_result.json")
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline.json")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")
REPO_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "src")

# 指标回退阈值（百分点），超过就判定为退化
REGRESSION_TOLERANCE = 2.0


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

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(stripped)
    except Exception:
        pass

    return answer


def call_deep_scan(contract_text: str) -> tuple:
    """调用 /chat 走 deep_scan 工具，返回 (结果, 耗时毫秒)"""
    payload = {
        "user_input": "帮我仔细审查这份合同",
        "contract_text": contract_text,
        "history": []
    }
    start = time.perf_counter()
    r = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
    cost_ms = (time.perf_counter() - start) * 1000
    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}: {r.text}")
    return r.json(), cost_ms


def normalize(text: str) -> str:
    """去掉空白，便于做条号和关键词的包含判断"""
    return "".join(str(text).split())


def evaluate_one(item: dict) -> dict:
    """评估单份合同"""
    base = {
        "id": item["id"],
        "label": item.get("label", ""),
        "hit": 0,
        "total": len(item["expected_risks"]),
        "risk_correct": 0,
        "predicted": 0,
        "tp": 0,
        "fp": 0,
        "high_predicted": 0,
        "high_tp": 0,
        "cited_ok": 0,
        "cited_total": 0,
        "latency_ms": 0,
        "over_flagged_items": [],
    }

    filepath = os.path.join(DOCS_DIR, item["file"])
    if not os.path.exists(filepath):
        base.update({"status": "文件缺失", "tool_used": "-"})
        return base

    with open(filepath, "r", encoding="utf-8") as f:
        contract_text = f.read()
    contract_norm = normalize(contract_text)

    try:
        result, cost_ms = call_deep_scan(contract_text)
        parsed = parse_answer(result.get("answer", ""))
        risk_items = parsed.get("risk_items", []) if isinstance(parsed, dict) else []

        keyword_to_label = {r["clause_keyword"]: r for r in item["expected_risks"]}

        # ---- 召回 + 等级：按期望条款逐个匹配，一条期望条款可被多条返回项命中
        matched_risks = {}
        for ri in risk_items:
            text = str(ri.get("clause", "")) + " " + str(ri.get("explanation", ""))
            for kw in keyword_to_label:
                if kw in text:
                    matched_risks.setdefault(kw, []).append(ri.get("risk"))

        hit = sum(1 for kw in keyword_to_label if kw in matched_risks)
        risk_correct = sum(
            1
            for kw, expected in keyword_to_label.items()
            if expected.get("risk") in matched_risks.get(kw, [])
        )

        # ---- 精确率：只看系统判为中/高的条款，能不能对上任意期望条款
        predicted = 0
        tp = 0
        high_predicted = 0
        high_tp = 0
        over_flagged_items = []
        for ri in risk_items:
            if ri.get("risk") not in ("高", "中"):
                continue
            predicted += 1
            text = str(ri.get("clause", "")) + " " + str(ri.get("explanation", ""))
            matched = any(kw in text for kw in keyword_to_label)
            if matched:
                tp += 1
            else:
                over_flagged_items.append({
                    "risk": ri.get("risk"),
                    "clause_no": ri.get("clause_no", ""),
                    "clause": str(ri.get("clause", ""))[:80],
                    "explanation": str(ri.get("explanation", ""))[:120],
                })
            if ri.get("risk") == "高":
                high_predicted += 1
                if matched:
                    high_tp += 1

        # ---- 引用可溯源：clause_no 非空且能在原文里找到
        cited_total = 0
        cited_ok = 0
        for ri in risk_items:
            clause_no = normalize(ri.get("clause_no", ""))
            if not clause_no:
                continue
            cited_total += 1
            if clause_no in contract_norm:
                cited_ok += 1

        base.update({
            "status": "成功",
            "hit": hit,
            "risk_correct": risk_correct,
            "predicted": predicted,
            "tp": tp,
            "fp": predicted - tp,
            "high_predicted": high_predicted,
            "high_tp": high_tp,
            "cited_ok": cited_ok,
            "cited_total": cited_total,
            "over_flagged_items": over_flagged_items,
            "latency_ms": round(cost_ms, 1),
            "tool_used": result.get("tool_used", "未知"),
            "high_count": parsed.get("high_count", 0) if isinstance(parsed, dict) else 0,
        })
        return base

    except Exception as e:
        base.update({"status": f"失败: {e}", "tool_used": "-"})
        return base


def read_version() -> dict:
    """记录版本信息：没有版本号，跨时间的数字不可比"""
    commit = "unknown"
    head_path = os.path.join(REPO_DIR, ".git", "HEAD")
    try:
        with open(head_path, "r", encoding="utf-8") as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            ref_path = os.path.join(REPO_DIR, ".git", ref)
            with open(ref_path, "r", encoding="utf-8") as f:
                commit = f.read().strip()[:8]
        else:
            commit = head[:8]
    except Exception:
        pass

    model = "unknown"
    try:
        sys.path.insert(0, SRC_DIR)
        import config  # noqa: F401
        model = getattr(config, "LLM_MODEL", model)
    except Exception:
        pass

    # 提示词或工具逻辑一改，历史数字就不能直接比，所以把关键文件哈希进版本
    prompt_hash = "unknown"
    try:
        h = hashlib.sha256()
        for name in ("agent/tools.py", "agent/agent.py"):
            with open(os.path.join(SRC_DIR, name), "rb") as f:
                h.update(f.read())
        prompt_hash = h.hexdigest()[:8]
    except Exception:
        pass

    return {"commit": commit, "model": model, "prompt_hash": prompt_hash}


def summarize(details: list) -> dict:
    total_hit = sum(d["hit"] for d in details)
    total_expected = sum(d["total"] for d in details)
    total_correct = sum(d["risk_correct"] for d in details)
    total_predicted = sum(d["predicted"] for d in details)
    total_tp = sum(d["tp"] for d in details)
    high_predicted = sum(d["high_predicted"] for d in details)
    high_tp = sum(d["high_tp"] for d in details)
    cited_ok = sum(d["cited_ok"] for d in details)
    cited_total = sum(d["cited_total"] for d in details)
    latencies = sorted(d["latency_ms"] for d in details if d["latency_ms"] > 0)

    def ratio(a, b):
        return round(a / b * 100, 2) if b else 0.0

    def percentile(values, p):
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, int(round(len(values) * p)) - 1))
        return round(values[idx], 1)

    return {
        "recall": ratio(total_hit, total_expected),
        "risk_accuracy": ratio(total_correct, total_expected),
        "precision": ratio(total_tp, total_predicted),
        "precision_high": ratio(high_tp, high_predicted),
        "citation": ratio(cited_ok, cited_total),
        "over_flagged": total_predicted - total_tp,
        "over_flagged_high": high_predicted - high_tp,
        "total_hit": total_hit,
        "total_expected": total_expected,
        "total_risk_correct": total_correct,
        "total_predicted": total_predicted,
        "total_tp": total_tp,
        "total_high_predicted": high_predicted,
        "total_high_tp": high_tp,
        "cited_ok": cited_ok,
        "cited_total": cited_total,
        "latency_p50": percentile(latencies, 0.5),
        "latency_p95": percentile(latencies, 0.95),
        "latency_avg": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
    }


def compare_with_baseline(current: dict, baseline: dict):
    """对比基线，任一核心指标回退超过阈值就返回失败列表"""
    regressions = []
    if baseline.get("version") and current.get("version"):
        if baseline["version"].get("prompt_hash") != current["version"].get("prompt_hash"):
            print("⚠️  提示词/工具代码已变化（prompt_hash 不同），指标对比仅供参考")

    cur_metrics = current.get("metrics", {})
    base_metrics = baseline.get("metrics", {})
    for key, label in (
        ("recall", "召回率"),
        ("risk_accuracy", "等级准确率"),
        ("precision", "精确率(中/高)"),
        ("precision_high", "精确率(高)"),
        ("citation", "引用可溯源率"),
    ):
        if key not in base_metrics:
            print(f"ℹ️  基线里没有 {label}，跳过对比")
            continue
        old = base_metrics[key]
        new = cur_metrics.get(key, old)
        delta = round(new - old, 2)
        flag = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"   {label}: {old}% → {new}% ({flag}{abs(delta):.2f}pt)")
        if delta < -REGRESSION_TOLERANCE:
            regressions.append(f"{label} 回退 {abs(delta):.2f}pt（超过阈值 {REGRESSION_TOLERANCE}pt）")
    return regressions


def main():
    update_baseline = "--update-baseline" in sys.argv
    dataset = load_dataset()
    print(f"📊 开始评估，共 {len(dataset)} 份合同\n")
    print("=" * 78)

    details = []
    for item in dataset:
        r = evaluate_one(item)
        details.append(r)
        print(f"[{r['id']}] {r.get('label', '')}")
        print(
            f"     状态: {r['status']} | 召回: {r['hit']}/{r['total']}"
            f" | 等级: {r['risk_correct']}/{r['total']}"
            f" | 过检: {r['fp']} | 引用: {r['cited_ok']}/{r['cited_total']}"
            f" | {r['latency_ms']}ms"
        )
        print("-" * 78)

    metrics = summarize(details)
    version = read_version()

    print(f"\n📈 召回率（关键词学科口径）: {metrics['total_hit']}/{metrics['total_expected']} = {metrics['recall']}%")
    print(f"📈 等级准确率（分母含漏报）: {metrics['total_risk_correct']}/{metrics['total_expected']} = {metrics['risk_accuracy']}%")
    print(f"📈 精确率（中/高条款命中期望的比例）: {metrics['total_tp']}/{metrics['total_predicted']} = {metrics['precision']}%")
    print(f"📈 高风险精确率（高风险里命中期望的比例）: {metrics['total_high_tp']}/{metrics['total_high_predicted']} = {metrics['precision_high']}%")
    print(f"📈 过检条款数: {metrics['over_flagged']}（其中高风险 {metrics['over_flagged_high']}）")
    print(f"📈 引用可溯源率: {metrics['cited_ok']}/{metrics['cited_total']} = {metrics['citation']}%")
    print(f"⏱️ 耗时: p50={metrics['latency_p50']}ms, p95={metrics['latency_p95']}ms, avg={metrics['latency_avg']}ms")
    print(f"🔖 版本: commit={version['commit']}, model={version['model']}, prompt_hash={version['prompt_hash']}")

    payload = {"metrics": metrics, "version": version, "details": details}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"📁 详细结果已保存到: {RESULT_PATH}")

    if update_baseline:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"✅ 基线已更新: {BASELINE_PATH}")
        return

    if not os.path.exists(BASELINE_PATH):
        print("ℹ️  没有基线文件，本次结果可作为基线：python evaluate.py --update-baseline")
        return

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    print("\n📊 与基线对比:")
    regressions = compare_with_baseline(payload, baseline)
    if regressions:
        print("\n❌ 检测到指标回退:")
        for r in regressions:
            print(f"   - {r}")
        sys.exit(1)
    print("\n✅ 无指标回退")


if __name__ == "__main__":
    main()
