# -*- coding: utf-8 -*-
"""
时效计算器技能（领域状态机 / 时间区间推理）

技术范式：有限状态机（FSM）建模法律时效 + 时间区间推理

为什么不靠 LLM 直接算：
    时效的起算、中断（重新起算）、中止（暂停后继续）、届满是一套确定性的
    时间代数。LLM 算日期会出错且不可复现。这里用状态机显式建模时效的生命周期：

        RUNNING --(中断事件)--> RUNNING(从中断日重新起算)
        RUNNING --(中止开始)--> SUSPENDED
        SUSPENDED --(中止结束)--> RUNNING(扣除中止期间)
        RUNNING --(到达 deadline)--> EXPIRED

    每一步状态转移都是纯函数，给定事件序列输出确定结果，可单测、可解释。
    LLM（若接入）仅负责把"我去年三月被辞退，今年一月公司承诺解决过一次"
    这种自然语言，解析成 (起算日, 事件列表)。

支持的时效类型见 STATUTE_RULES。
"""
import re
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Tuple, Dict, Any

from langchain.tools import tool


# ==================== 时效规则库 ====================

STATUTE_RULES = {
    "劳动仲裁": {
        "period_days": 365, "period_text": "1年",
        "basis": "《劳动争议调解仲裁法》第27条",
        "detail": "劳动争议申请仲裁的时效期间为一年，从当事人知道或应当知道其权利被侵害之日起计算。",
        "keywords": ["仲裁", "劳动仲裁", "劳动争议"],
    },
    "拖欠工资": {
        "period_days": 365, "period_text": "1年（在职不受限）",
        "basis": "《劳动争议调解仲裁法》第27条第4款",
        "detail": "劳动关系存续期间因拖欠劳动报酬发生争议的，不受一年仲裁时效限制；劳动关系终止的，自终止之日起一年内提出。",
        "keywords": ["拖欠工资", "克扣工资", "欠薪", "工资未发", "拖欠", "欠我", "拖欠我的工资", "未发工资"],
        "special": "在职期间不受 1 年时效限制，离职后 1 年内必须申请",
    },
    "违法解除": {
        "period_days": 365, "period_text": "1年",
        "basis": "《劳动争议调解仲裁法》第27条",
        "detail": "因解除劳动合同发生争议的，仲裁时效为 1 年。",
        "keywords": ["违法解除", "违法辞退", "无故辞退", "非法解除", "辞退", "开除"],
    },
    "工伤认定": {
        "period_days": 365, "period_text": "1年（单位30天）",
        "basis": "《工伤保险条例》第17条",
        "detail": "用人单位应自事故伤害发生之日起 30 日内提出工伤认定申请；单位未申请的，工伤职工或近亲属可在 1 年内申请。",
        "keywords": ["工伤", "工伤认定", "因工受伤", "职业病"],
        "special": "单位 30 天内申请，个人 1 年内申请",
    },
    "未签合同": {
        "period_days": 365, "period_text": "1年",
        "basis": "《劳动争议调解仲裁法》第27条",
        "detail": "未签书面劳动合同的双倍工资差额，仲裁时效为 1 年。",
        "keywords": ["未签合同", "没签合同", "双倍工资", "未签劳动合同"],
        "special": "双倍工资最多支持 11 个月",
    },
    "普通民事诉讼": {
        "period_days": 1095, "period_text": "3年",
        "basis": "《民法典》第188条",
        "detail": "向人民法院请求保护民事权利的诉讼时效期间为三年。",
        "keywords": ["诉讼", "民事诉讼", "打官司", "起诉"],
    },
}


# ==================== 状态机核心（确定性，可单测） ====================

@dataclass
class TimelineEvent:
    """时效生命周期事件"""
    kind: str          # "interrupt" 中断 | "suspend_start" 中止开始 | "suspend_end" 中止结束
    at: date
    note: str = ""


@dataclass
class StatuteResult:
    """状态机推理结果"""
    start_date: date
    deadline: date
    is_expired: bool
    days_remaining: int
    days_suspended: int = 0
    interrupted_times: int = 0
    transitions: List[str] = field(default_factory=list)


def run_statute_machine(
    start: date,
    period_days: int,
    events: List[TimelineEvent],
    as_of: Optional[date] = None,
) -> StatuteResult:
    """
    时效状态机：给定起算日、时效天数、事件序列，推算届满日与剩余天数。

    规则（民法时效通则）：
    - 中断（interrupt）：时效从中断之日起**重新计算**（已过时间清零）。
    - 中止（suspend）：中止期间不计入时效，结束后**继续**计算（届满日顺延中止时长）。
    - 届满：有效经过时间 >= period_days。

    Args:
        start: 起算日（知道权利被侵害之日）
        period_days: 时效天数
        events: 按时间排序的事件列表
        as_of: 评估基准日（默认今天）

    Returns:
        StatuteResult
    """
    as_of = as_of or date.today()
    events = sorted(events, key=lambda e: e.at)

    effective_start = start          # 当前这段时效的起算点（中断后会重置）
    suspended_days = 0               # 累计中止天数
    interrupt_count = 0
    suspend_open: Optional[date] = None
    transitions: List[str] = [f"{start.isoformat()} 时效起算 (RUNNING)"]

    for ev in events:
        if ev.at < effective_start:
            continue
        if ev.kind == "interrupt":
            effective_start = ev.at      # 重新起算
            suspended_days = 0           # 重算后此前的中止一并清零
            interrupt_count += 1
            suspend_open = None
            transitions.append(f"{ev.at.isoformat()} 中断→自此重新起算 (RUNNING)")
        elif ev.kind == "suspend_start":
            if suspend_open is None:
                suspend_open = ev.at
                transitions.append(f"{ev.at.isoformat()} 中止开始 (SUSPENDED)")
        elif ev.kind == "suspend_end":
            if suspend_open is not None:
                suspended_days += (ev.at - suspend_open).days
                transitions.append(
                    f"{ev.at.isoformat()} 中止结束，累计中止 {suspended_days} 天 (RUNNING)"
                )
                suspend_open = None

    # 若中止仍未结束，按 as_of 截止计算中止时长
    if suspend_open is not None:
        suspended_days += (as_of - suspend_open).days
        transitions.append(f"{as_of.isoformat()} 中止持续中，截至基准日累计 {suspended_days} 天")

    deadline = effective_start + timedelta(days=period_days + suspended_days)
    days_remaining = (deadline - as_of).days
    is_expired = days_remaining < 0
    transitions.append(
        f"{deadline.isoformat()} 时效届满 ({'EXPIRED' if is_expired else 'RUNNING'})"
    )

    return StatuteResult(
        start_date=effective_start,
        deadline=deadline,
        is_expired=is_expired,
        days_remaining=days_remaining,
        days_suspended=suspended_days,
        interrupted_times=interrupt_count,
        transitions=transitions,
    )


# ==================== 自然语言解析（确定性正则；LLM 可选增强） ====================

def _extract_date(text: str) -> Optional[date]:
    """从文本中提取首个日期"""
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass
    m = re.search(r"(\d{4})\s*年", text)
    if m:
        y = int(m.group(1))
        if 2000 <= y <= 2099:
            return date(y, 1, 1)
    return None


def _match_event_type(text: str) -> Tuple[Optional[str], Optional[dict]]:
    """匹配时效类型"""
    for et, cfg in STATUTE_RULES.items():
        if any(kw in text for kw in cfg["keywords"]):
            return et, cfg
    return None, None


def _extract_lifecycle_events(text: str, start: date) -> List[TimelineEvent]:
    """
    从文本中识别中断/中止事件（确定性关键词 + 日期）。

    中断信号：主张过权利、申请过仲裁、对方承诺履行、寄送催告等。
    中止信号：不可抗力、当事人无民事行为能力等（这里仅做演示性识别）。
    """
    events: List[TimelineEvent] = []
    # 中断：找"曾经主张/催告/承诺"类表述附近的日期
    for kw in ["主张", "催告", "承诺", "申请过", "投诉", "协商"]:
        for m in re.finditer(kw, text):
            seg = text[m.start(): m.start() + 25]
            d = _extract_date(seg)
            if d and d > start:
                events.append(TimelineEvent("interrupt", d, note=f"{kw}"))
    return events


# ==================== @tool / 统一接口 ====================

@tool
def statute_checker(input_text: str) -> str:
    """时效计算器工具。输入事件描述和时间，用状态机推算仲裁/诉讼时效是否届满。

    支持时效的中断（重新起算）与中止（暂停后续算）。

    Args:
        input_text: 事件描述，含类型与时间，如"2024年3月被违法辞退，2024年9月协商过一次"

    Returns:
        JSON 格式的时效计算结果（含状态机转移轨迹）
    """
    if not input_text or not input_text.strip():
        return json.dumps({"error": "输入为空，请描述事件类型和发生时间"}, ensure_ascii=False)

    event_type, cfg = _match_event_type(input_text)
    start = _extract_date(input_text)

    if event_type is None:
        return json.dumps({
            "status": "无法识别事件类型",
            "hint": "请描述具体劳动争议事件，如违法辞退、拖欠工资、工伤、未签合同等。",
            "supported_types": list(STATUTE_RULES.keys()),
        }, ensure_ascii=False, indent=2)

    # 在职拖欠工资：不受时效限制
    if event_type == "拖欠工资" and any(k in input_text for k in ["在职", "还在", "未离职", "仍在"]):
        return json.dumps({
            "event_type": event_type,
            "statute_type": "不受时效限制",
            "is_expired": False,
            "status": "在职期间不受 1 年仲裁时效限制",
            "legal_basis": cfg["basis"],
            "legal_detail": cfg["detail"],
            "special_note": cfg.get("special", ""),
            "suggestion": "在职期间可随时申请仲裁；离职后须在 1 年内提出。",
        }, ensure_ascii=False, indent=2)

    if start is None:
        return json.dumps({
            "status": "无法提取日期",
            "event_type": event_type,
            "hint": "请提供事件发生日期，如 2024年3月15日 / 2024-03-15 / 2024年3月。",
            "statute_info": {"period": cfg["period_text"], "basis": cfg["basis"], "detail": cfg["detail"]},
        }, ensure_ascii=False, indent=2)

    events = _extract_lifecycle_events(input_text, start)
    res = run_statute_machine(start, cfg["period_days"], events)

    if res.is_expired:
        status = "已超过时效"
        suggestion = "已超过仲裁/诉讼时效，建议尽快咨询律师评估是否存在时效中断/中止情形。"
    else:
        status = "仍在时效内"
        if res.days_remaining <= 30:
            suggestion = f"时效即将届满（剩余 {res.days_remaining} 天），请立即申请。"
        elif res.days_remaining <= 90:
            suggestion = f"时效剩余 {res.days_remaining} 天，建议尽快准备材料。"
        else:
            suggestion = f"时效剩余 {res.days_remaining} 天，建议尽早准备。"

    return json.dumps({
        "event_type": event_type,
        "statute_period": cfg["period_text"],
        "original_start": start.isoformat(),
        "effective_start": res.start_date.isoformat(),
        "deadline": res.deadline.isoformat(),
        "days_remaining": res.days_remaining,
        "is_expired": res.is_expired,
        "interrupted_times": res.interrupted_times,
        "days_suspended": res.days_suspended,
        "status": status,
        "legal_basis": cfg["basis"],
        "legal_detail": cfg["detail"],
        "special_note": cfg.get("special", ""),
        "state_transitions": res.transitions,
        "suggestion": suggestion,
    }, ensure_ascii=False, indent=2)


def statute_skill(query: str, law_context: str = "") -> str:
    """统一接口：接受 query（事件描述）"""
    return statute_checker.invoke({"input_text": query})


# 导出
STATUTE_CHECKER_TOOLS = [statute_checker]
