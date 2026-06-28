# -*- coding: utf-8 -*-
"""
时效计算器技能

计算劳动仲裁/诉讼时效，判断是否届满。
"""
import re
import json
from datetime import date, datetime, timedelta
from langchain.tools import tool


# ==================== 时效规则库 ====================

STATUTE_RULES = {
    "劳动仲裁": {
        "period_days": 365,
        "period_text": "1年",
        "basis": "《劳动争议调解仲裁法》第27条",
        "detail": "劳动争议申请仲裁的时效期间为一年，从当事人知道或者应当知道其权利被侵害之日起计算。",
        "keywords": ["仲裁", "劳动仲裁", "劳动争议"],
    },
    "拖欠工资": {
        "period_days": 365,
        "period_text": "1年（在职期间不受限）",
        "basis": "《劳动争议调解仲裁法》第27条第4款",
        "detail": "劳动关系存续期间因拖欠劳动报酬发生争议的，劳动者申请仲裁不受一年的仲裁时效期间限制；但劳动关系终止的，应当自劳动关系终止之日起一年内提出。",
        "keywords": ["拖欠工资", "克扣工资", "欠薪", "工资未发"],
        "special": "在职期间不受1年时效限制，离职后1年内必须申请",
    },
    "违法解除": {
        "period_days": 365,
        "period_text": "1年",
        "basis": "《劳动争议调解仲裁法》第27条",
        "detail": "因解除劳动合同发生争议的，仲裁时效为1年。",
        "keywords": ["违法解除", "违法辞退", "无故辞退", "非法解除"],
    },
    "未签合同": {
        "period_days": 365,
        "period_text": "1年",
        "basis": "《劳动争议调解仲裁法》第27条、《劳动合同法》第82条",
        "detail": "未签书面劳动合同的双倍工资，仲裁时效为1年，从应当签订合同之日起算。",
        "keywords": ["未签合同", "没签合同", "双倍工资", "未签劳动合同"],
        "special": "双倍工资最多支持11个月（从用工第2个月起算）",
    },
    "工伤认定": {
        "period_days": 365,
        "period_text": "1年（单位30天）",
        "basis": "《工伤保险条例》第17条",
        "detail": "用人单位应自事故伤害发生之日起30日内提出工伤认定申请。用人单位未申请的，工伤职工或其近亲属可在1年内直接申请。",
        "keywords": ["工伤", "工伤认定", "因工受伤", "职业病"],
        "special": "单位30天内申请，个人1年内申请",
    },
    "未缴社保": {
        "period_days": None,
        "period_text": "不受时效限制",
        "basis": "《社会保险法》第63条",
        "detail": "用人单位未按时足额缴纳社会保险费的，由社会保险费征收机构责令限期缴纳或者补足，不受仲裁时效限制。",
        "keywords": ["未缴社保", "不缴社保", "社保补缴", "没上保险"],
        "special": "可随时向社保部门投诉要求补缴",
    },
    "加班费": {
        "period_days": 365,
        "period_text": "1年",
        "basis": "《劳动争议调解仲裁法》第27条",
        "detail": "因加班费发生争议的，仲裁时效为1年。但劳动关系存续期间不受限制。",
        "keywords": ["加班费", "加班工资", "未付加班费"],
    },
    "经济补偿": {
        "period_days": 365,
        "period_text": "1年",
        "basis": "《劳动争议调解仲裁法》第27条",
        "detail": "因经济补偿发生争议的，仲裁时效为1年。",
        "keywords": ["经济补偿", "补偿金", "N+1", "赔偿金"],
    },
    "普通民事诉讼": {
        "period_days": 1095,
        "period_text": "3年",
        "basis": "《民法典》第188条",
        "detail": "向人民法院请求保护民事权利的诉讼时效期间为三年。",
        "keywords": ["诉讼", "民事诉讼", "打官司", "起诉"],
    },
    "人身损害赔偿": {
        "period_days": 1095,
        "period_text": "3年",
        "basis": "《民法典》第188条",
        "detail": "人身损害赔偿请求权的诉讼时效期间为三年。",
        "keywords": ["人身损害", "人身伤害", "身体伤害"],
    },
}


def _extract_date(text: str) -> date:
    """从文本中提取日期"""
    # 格式1: YYYY年MM月DD日
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # 格式2: YYYY-MM-DD 或 YYYY/MM/DD
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # 格式3: YYYY年MM月（默认为该月1日）
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass

    # 格式4: YYYY年（默认为1月1日）
    m = re.search(r"(\d{4})\s*年", text)
    if m:
        try:
            year = int(m.group(1))
            if 2000 <= year <= 2099:
                return date(year, 1, 1)
        except ValueError:
            pass

    return None


def _match_event_type(text: str) -> tuple:
    """匹配事件类型"""
    for event_type, config in STATUTE_RULES.items():
        for kw in config["keywords"]:
            if kw in text:
                return event_type, config
    return None, None


@tool
def statute_checker(input_text: str) -> str:
    """
    时效计算器工具。输入事件描述和时间，计算仲裁/诉讼时效是否届满。

    支持的时效类型：劳动仲裁（1年）、拖欠工资（在职不受限）、
    工伤认定（1年）、未签合同双倍工资（1年）、未缴社保（不受限）、
    普通民事诉讼（3年）等。

    Args:
        input_text: 事件描述，应包含事件类型和发生时间。
                   如"2024年3月被公司违法辞退"、"2023年1月开始拖欠工资"

    Returns:
        JSON 格式的时效计算结果
    """
    if not input_text or not input_text.strip():
        return json.dumps({"error": "输入为空，请描述事件类型和发生时间"}, ensure_ascii=False)

    # 匹配事件类型
    event_type, config = _match_event_type(input_text)

    # 提取日期
    event_date = _extract_date(input_text)

    if event_type is None:
        return json.dumps({
            "status": "无法识别事件类型",
            "input": input_text,
            "hint": "请描述具体的劳动争议事件，如：违法辞退、拖欠工资、工伤、未签合同等。",
            "supported_types": list(STATUTE_RULES.keys()),
        }, ensure_ascii=False, indent=2)

    if event_date is None:
        return json.dumps({
            "status": "无法提取日期",
            "event_type": event_type,
            "input": input_text,
            "hint": "请提供事件发生的日期，格式如：2024年3月15日、2024-03-15、2024年3月。",
            "statute_info": {
                "period": config["period_text"],
                "basis": config["basis"],
                "detail": config["detail"],
            },
        }, ensure_ascii=False, indent=2)

    # 计算时效
    today = date.today()

    if config["period_days"] is None:
        # 不受时效限制
        result = {
            "event_type": event_type,
            "event_date": event_date.isoformat(),
            "statute_type": "不受时效限制",
            "statute_period": config["period_text"],
            "start_date": event_date.isoformat(),
            "deadline": "无限制",
            "days_remaining": "无限制",
            "is_expired": False,
            "status": "不受时效限制",
            "legal_basis": config["basis"],
            "legal_detail": config["detail"],
            "special_note": config.get("special", ""),
            "suggestion": "可随时主张权利，但建议尽早处理。",
        }
    else:
        deadline = event_date + timedelta(days=config["period_days"])
        days_remaining = (deadline - today).days
        is_expired = days_remaining < 0

        if is_expired:
            status = "已超过时效"
            suggestion = f"已超过{config['period_text']}仲裁时效。建议尽快咨询律师，评估是否存在时效中断/中止的情形。"
        else:
            status = "仍在时效内"
            if days_remaining <= 30:
                suggestion = f"时效即将届满（剩余{days_remaining}天），请立即申请仲裁/起诉。"
            elif days_remaining <= 90:
                suggestion = f"时效剩余{days_remaining}天，建议尽快准备材料申请仲裁/起诉。"
            else:
                suggestion = f"时效剩余{days_remaining}天，建议尽早准备。"

        result = {
            "event_type": event_type,
            "event_date": event_date.isoformat(),
            "statute_type": "劳动仲裁" if "仲裁" in config["basis"] else "民事诉讼",
            "statute_period": config["period_text"],
            "start_date": event_date.isoformat(),
            "deadline": deadline.isoformat(),
            "days_remaining": days_remaining,
            "is_expired": is_expired,
            "status": status,
            "legal_basis": config["basis"],
            "legal_detail": config["detail"],
            "special_note": config.get("special", ""),
            "suggestion": suggestion,
        }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ==================== 统一接口 ====================

def statute_skill(query: str) -> str:
    """统一接口：接受 query，内部作为事件描述分析"""
    return statute_checker.invoke({"input_text": query})


# 导出的工具列表
STATUTE_CHECKER_TOOLS = [statute_checker]
