# -*- coding: utf-8 -*-
"""
Review Agent 评估脚本（LangSmith，不需要 LLM Judge）

评估指标：
- Tool Call Accuracy：是否调用了正确的工具
- Review Status Accuracy：approve/revise 判断是否正确
- Completion Rate：是否在限定步骤内完成

用法：
    python tests/eval_review_agent.py
    python tests/eval_review_agent.py --limit 5
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client
from langsmith.evaluation import evaluate

from workflow_service import WorkflowService


# ==================== 测试数据集 ====================

REVIEW_DATASET = [
    # --- 应该 approve 的案例 ---
    {
        "input": {
            "draft": "根据《劳动法》第三十六条，劳动者每日工作时间不超过八小时。",
            "law_context": "第三十六条　国家实行劳动者每日工作时间不超过八小时、平均每周工作时间不超过四十四小时的工时制度。",
        },
        "expected": {
            "tools_called": ["citation_check", "add_disclaimer"],
            "review_status": "approve",
        },
    },
    {
        "input": {
            "draft": "您好，今天天气不错。",
            "law_context": "",
        },
        "expected": {
            "tools_called": [],
            "review_status": "approve",
        },
    },
    {
        "input": {
            "draft": "劳动合同应当以书面形式订立，这是基本要求。",
            "law_context": "第十九条　劳动合同应当以书面形式订立。",
        },
        "expected": {
            "tools_called": ["citation_check", "add_disclaimer"],
            "review_status": "approve",
        },
    },
    # --- 应该 revise 的案例（引用了不存在的法条） ---
    {
        "input": {
            "draft": "根据《劳动法》第一百二十条，用人单位可以随意解除劳动合同。",
            "law_context": "第三十六条　国家实行劳动者每日工作时间不超过八小时。",
        },
        "expected": {
            "tools_called": ["citation_check"],
            "review_status": "revise",
        },
    },
    {
        "input": {
            "draft": "根据《劳动合同法》第五十条和《劳动法》第一百条，用人单位必须缴纳社保。",
            "law_context": "第七十二条　社会保险基金按照保险类型确定资金来源。",
        },
        "expected": {
            "tools_called": ["citation_check", "add_disclaimer"],
            "review_status": "revise",
        },
    },
    # --- PII 检测案例 ---
    {
        "input": {
            "draft": "请联系张三，手机号13800138000，邮箱zhangsan@test.com。",
            "law_context": "",
        },
        "expected": {
            "tools_called": ["pii_check"],
            "review_status": "approve",
        },
    },
    {
        "input": {
            "draft": "员工身份证号为110101199001011234，银行卡号6222021234567890123。",
            "law_context": "",
        },
        "expected": {
            "tools_called": ["pii_check"],
            "review_status": "approve",
        },
    },
    # --- 质量检查案例 ---
    {
        "input": {
            "draft": "是的。",
            "law_context": "",
        },
        "expected": {
            "tools_called": ["quality_check"],
            "review_status": "approve",
        },
    },
    # --- 综合案例：PII + 引用问题 ---
    {
        "input": {
            "draft": "根据《劳动法》第二百条，联系李四13912345678了解详情。",
            "law_context": "第三十六条　国家实行劳动者每日工作时间不超过八小时。",
        },
        "expected": {
            "tools_called": ["pii_check", "citation_check"],
            "review_status": "revise",
        },
    },
    # --- 法律内容需要免责声明 ---
    {
        "input": {
            "draft": "劳动者有权拒绝违章指挥，这是法律赋予的权利。",
            "law_context": "第五十六条　劳动者对用人单位管理人员违章指挥、强令冒险作业，有权拒绝执行。",
        },
        "expected": {
            "tools_called": ["citation_check", "add_disclaimer"],
            "review_status": "approve",
        },
    },
]


# ==================== 评估函数 ====================

def tool_accuracy_evaluator(run, example):
    """评估工具调用准确性：是否调用了预期的工具"""
    expected_tools = set(example.outputs.get("expected", {}).get("tools_called", []))

    # 从 run 的消息中提取实际调用的工具
    actual_tools = set()
    for msg in run.outputs.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                actual_tools.add(tc.get("name", "") if isinstance(tc, dict) else tc.name)

    # 计算准确率
    if not expected_tools and not actual_tools:
        return {"score": 1.0, "comment": "两者都无工具调用"}

    if not expected_tools:
        return {"score": 0.0, "comment": f"不应调用工具但调用了: {actual_tools}"}

    hits = expected_tools & actual_tools
    score = len(hits) / len(expected_tools) if expected_tools else 0.0

    return {
        "score": score,
        "comment": f"期望={sorted(expected_tools)}, 实际={sorted(actual_tools)}, 命中={sorted(hits)}",
    }


def review_status_evaluator(run, example):
    """评估审查状态判断是否正确"""
    expected_status = example.outputs.get("expected", {}).get("review_status", "approve")

    # 从 run 的输出中提取 review_status
    actual_status = "approve"
    for msg in run.outputs.get("messages", []):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if "[引用验证] 发现问题" in msg.content:
                actual_status = "revise"
                break

    score = 1.0 if actual_status == expected_status else 0.0
    return {
        "score": score,
        "comment": f"期望={expected_status}, 实际={actual_status}",
    }


def completion_evaluator(run, example):
    """评估是否正常完成（没有异常/超时）"""
    messages = run.outputs.get("messages", [])
    if not messages:
        return {"score": 0.0, "comment": "无输出消息"}

    # 检查是否有最终 AI 回复
    has_final = False
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
            has_final = True
            break
        if isinstance(msg, dict) and msg.get("type") == "ai" and msg.get("content"):
            has_final = True
            break

    if has_final:
        return {"score": 1.0, "comment": f"正常完成，共 {len(messages)} 条消息"}
    return {"score": 0.0, "comment": f"未找到最终回复，共 {len(messages)} 条消息"}


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="Review Agent 评估")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset = REVIEW_DATASET[:args.limit] if args.limit > 0 else REVIEW_DATASET

    print(f"\n{'='*60}")
    print(f"Review Agent 评估 — {len(dataset)} 条测试数据")
    print(f"指标: Tool Accuracy + Status Accuracy + Completion")
    print(f"{'='*60}\n")

    # 1. 初始化 Review Agent
    print("📦 初始化 WorkflowService...")
    service = WorkflowService()
    review_agent = service.contract_graph.nodes.get("review")

    # 2. 准备 LangSmith 数据集
    client = Client()

    # 创建数据集
    ds_name = "review-agent-eval"
    try:
        existing = client.read_dataset(dataset_name=ds_name)
        print(f"📦 使用已有数据集: {ds_name}")
        dataset_id = existing.id
        # 清空旧数据
        for ex in client.list_examples(dataset_id=dataset_id):
            client.delete_example(ex.id)
    except Exception:
        dataset_obj = client.create_dataset(dataset_name=ds_name)
        dataset_id = dataset_obj.id
        print(f"📦 创建数据集: {ds_name}")

    # 添加测试用例
    for item in dataset:
        client.create_example(
            inputs={"draft": item["input"]["draft"], "law_context": item["input"]["law_context"]},
            outputs={"expected": item["expected"]},
            dataset_id=dataset_id,
        )

    # 3. 创建 Review Agent 节点（直接调用，不走整个图）
    from workflows.review_agent import create_review_agent_node
    review_node = create_review_agent_node(service.llm)

    async def target(inputs: dict) -> dict:
        """直接调用 Review Agent 节点"""
        state = {
            "draft_answer": inputs["draft"],
            "law_context": inputs.get("law_context", ""),
            "query": "",
            "tool_history": [],
            "review_status": "approve",
            "review_issues": [],
            "review_retry": 0,
        }
        result = await review_node(state)
        return result

    # 4. 运行评估
    print(f"\n🔄 运行评估...\n")
    results = evaluate(
        target,
        data=ds_name,
        evaluators=[tool_accuracy_evaluator, review_status_evaluator, completion_evaluator],
        client=client,
    )

    # 5. 输出结果
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")

    df = results.to_pandas()
    for col in ["tool_accuracy_evaluator", "review_status_evaluator", "completion_evaluator"]:
        if col in df.columns:
            mean_score = df[col].mean()
            print(f"  {col}: {mean_score:.4f}")

    print(f"\n✅ 评估完成！")
    service.shutdown()


if __name__ == "__main__":
    main()
