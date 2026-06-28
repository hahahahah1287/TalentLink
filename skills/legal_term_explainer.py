# -*- coding: utf-8 -*-
"""
法律术语解释技能

识别文本中的法律术语，用通俗语言解释并附带法条出处。
"""
import json
from langchain.tools import tool


# ==================== 法律术语词典 ====================

LEGAL_TERMS = {
    "经济补偿金": {
        "plain_explanation": "用人单位解除或终止劳动合同时，按工作年限支付给劳动者的补偿。每满一年支付一个月工资。",
        "legal_basis": "《劳动合同法》第47条",
        "example": "小王在公司工作了3年，月薪8000元，被合法辞退时应获得3×8000=24000元经济补偿金。",
    },
    "赔偿金": {
        "plain_explanation": "用人单位违法解除劳动合同时，支付的惩罚性赔偿，标准是经济补偿金的两倍。",
        "legal_basis": "《劳动合同法》第87条",
        "example": "公司无故辞退工作3年的小王，应支付2×3×8000=48000元赔偿金。",
    },
    "代通知金": {
        "plain_explanation": "用人单位没有提前30天书面通知就解除劳动合同，需要额外支付一个月工资作为替代。",
        "legal_basis": "《劳动合同法》第40条",
        "example": "公司当天通知小王离职，除经济补偿外还需多付一个月工资作为代通知金。",
    },
    "竞业限制": {
        "plain_explanation": "离职后一定时间内，不能去竞争对手公司工作或自己开同类公司。单位必须按月给你补偿金。",
        "legal_basis": "《劳动合同法》第23、24条",
        "example": "小王从A互联网公司离职，合同约定2年内不能去B互联网公司，A公司需每月支付小王离职前工资的30%以上作为补偿。",
    },
    "服务期协议": {
        "plain_explanation": "单位出钱给你培训后，约定你必须在单位工作一定年限。如果提前离职，需要按比例退还培训费。",
        "legal_basis": "《劳动合同法》第22条",
        "example": "公司花2万元送小王培训，约定服务期5年，小王工作2年后离职，需退还2万×(5-2)/5=1.2万元。",
    },
    "无固定期限劳动合同": {
        "plain_explanation": "没有确定终止日期的劳动合同，除非出现法定情形，单位不能随意辞退你。",
        "legal_basis": "《劳动合同法》第14条",
        "example": "小王在公司连续工作满10年，有权要求签订无固定期限劳动合同。",
    },
    "劳务派遣": {
        "plain_explanation": "你和派遣公司签合同，但去别的公司干活。派遣公司是你的法律上的雇主。",
        "legal_basis": "《劳动合同法》第57-67条",
        "example": "小王和A派遣公司签合同，被派到B公司工作，工资由A公司发，但日常管理归B公司。",
    },
    "非全日制用工": {
        "plain_explanation": "以小时计算工资的灵活用工形式，每天不超过4小时，每周不超过24小时。可以口头约定，双方随时可以终止。",
        "legal_basis": "《劳动合同法》第68-72条",
        "example": "小王每天在餐厅工作3小时，每周工作6天，按小时结算工资，属于非全日制用工。",
    },
    "工伤保险": {
        "plain_explanation": "工作中受伤或得职业病时，可以获得免费医疗和经济补偿。费用由单位缴纳，个人不用出钱。",
        "legal_basis": "《工伤保险条例》",
        "example": "小王上班途中遭遇交通事故受伤，经认定为工伤，医疗费由工伤保险基金支付。",
    },
    "失业保险": {
        "plain_explanation": "非因本人意愿失业后（如被辞退），可以按月领取失业金，同时享受医疗保险待遇。",
        "legal_basis": "《社会保险法》第45-52条",
        "example": "小王被公司裁员，缴纳失业保险满1年，可领取最长12个月的失业金。",
    },
    "最低工资标准": {
        "plain_explanation": "当地政府规定的最低工资底线，单位给你的工资不能低于这个标准，且不含加班费和补贴。",
        "legal_basis": "《劳动法》第48条",
        "example": "某市最低工资标准为2320元/月，小王基本工资不能低于这个数。",
    },
    "加班费": {
        "plain_explanation": "超出正常工作时间的额外报酬。平日加班1.5倍，休息日加班2倍，法定节假日加班3倍。",
        "legal_basis": "《劳动法》第44条",
        "example": "小王月薪6000元（日薪约276元），国庆加班1天应得276×3=828元。",
    },
    "年休假": {
        "plain_explanation": "每年享有的带薪连续假期。工作满1年有5天，满10年有10天，满20年有15天。未休的要按3倍工资补偿。",
        "legal_basis": "《职工带薪年休假条例》第3、5条",
        "example": "小王工作5年，每年有5天年假。如果公司不安排休假，需按日工资的300%支付补偿。",
    },
    "劳动仲裁": {
        "plain_explanation": "和公司发生劳动纠纷时，先找劳动争议仲裁委员会裁决。这是打劳动官司的前置程序，不收费。",
        "legal_basis": "《劳动争议调解仲裁法》",
        "example": "小王被拖欠工资，可以向当地劳动仲裁委申请仲裁，要求公司支付欠薪。",
    },
    "举证责任": {
        "plain_explanation": "谁主张谁举证。但涉及考勤、工资记录等，由用人单位负责举证（倒置）。",
        "legal_basis": "《劳动争议调解仲裁法》第6条",
        "example": "小王主张公司欠薪，公司否认。因工资记录由公司保管，公司需举证证明已发放。",
    },
    "连续工龄": {
        "plain_explanation": "在同一单位连续工作的时间。影响年假天数、经济补偿金计算、无固定期限合同签订等。",
        "legal_basis": "《劳动合同法》第47条等",
        "example": "小王在公司工作了8年零7个月，经济补偿金按9个月工资计算（超过6个月按1年算）。",
    },
    "违法解除": {
        "plain_explanation": "公司没有合法理由或没有按法定程序辞退你。你可以要求继续上班，或者拿双倍赔偿金走人。",
        "legal_basis": "《劳动合同法》第48条",
        "example": "怀孕的小王被公司辞退，属于违法解除，可要求继续履行合同或支付2N赔偿金。",
    },
    "被迫辞职": {
        "plain_explanation": "因为公司有错（如欠薪、不缴社保、不提供劳动条件），你被迫提出辞职，公司仍需支付经济补偿金。",
        "legal_basis": "《劳动合同法》第38条",
        "example": "公司连续3个月拖欠工资，小王以此为由提出解除合同，公司仍需支付经济补偿金。",
    },
    "双倍工资": {
        "plain_explanation": "公司用工超过1个月还没和你签书面劳动合同，从第2个月起要付双倍工资，最多付11个月。",
        "legal_basis": "《劳动合同法》第82条",
        "example": "小王入职6个月公司都没签合同，公司需额外支付5个月的双倍工资差额。",
    },
    "社会保险": {
        "plain_explanation": "国家强制缴纳的五险：养老、医疗、失业、工伤、生育保险。单位和个人各承担一部分（工伤和生育全由单位出）。",
        "legal_basis": "《社会保险法》",
        "example": "小王月薪10000元，个人需缴纳养老8%+医疗2%+失业0.5%=1050元，单位缴纳约30%。",
    },
    "住房公积金": {
        "plain_explanation": "单位和你各出一部分钱存起来，用于买房、租房、装修等。缴存比例一般5%-12%。",
        "legal_basis": "《住房公积金管理条例》",
        "example": "小王月薪10000元，公积金比例12%，则个人和单位各缴1200元，每月共存入2400元。",
    },
    "劳动能力鉴定": {
        "plain_explanation": "工伤治疗稳定后，鉴定你的劳动功能障碍程度（1-10级伤残）和生活自理障碍等级。",
        "legal_basis": "《工伤保险条例》第21-26条",
        "example": "小王工伤治愈后，经鉴定为8级伤残，可获得一次性伤残补助金（11个月本人工资）。",
    },
    "N+1": {
        "plain_explanation": "经济补偿的通俗说法。N是工作年限（每满1年算1个月），+1是未提前30天通知时多付的1个月工资（代通知金）。",
        "legal_basis": "《劳动合同法》第40、46、47条",
        "example": "小王工作5年，月薪10000元，公司未提前通知即辞退，应付(5+1)×10000=60000元。",
    },
    "2N": {
        "plain_explanation": "违法解除的赔偿金，即经济补偿金的两倍。不需要+1。",
        "legal_basis": "《劳动合同法》第87条",
        "example": "小王工作5年，月薪10000元，公司违法辞退，应付2×5×10000=100000元。",
    },
}


@tool
def legal_term_explainer(text: str) -> str:
    """
    法律术语解释工具。识别文本中的法律术语，用通俗语言解释。

    自动识别劳动法、合同法等领域的专业术语，提供通俗解释、
    法条出处和实际案例说明。

    Args:
        text: 包含法律术语的文本（合同条款、法律条文、用户问题等）

    Returns:
        JSON 格式的术语解释结果
    """
    if not text or not text.strip():
        return json.dumps({"error": "输入文本为空"}, ensure_ascii=False)

    found_terms = []
    for term, info in LEGAL_TERMS.items():
        if term in text:
            found_terms.append({
                "term": term,
                "plain_explanation": info["plain_explanation"],
                "legal_basis": info["legal_basis"],
                "example": info["example"],
            })

    if not found_terms:
        # 尝试模糊匹配
        fuzzy_matches = []
        for term, info in LEGAL_TERMS.items():
            # 检查部分匹配（至少2个字）
            for i in range(len(term) - 1):
                if term[i:i+2] in text and term not in fuzzy_matches:
                    fuzzy_matches.append({
                        "term": term,
                        "plain_explanation": info["plain_explanation"],
                        "legal_basis": info["legal_basis"],
                        "example": info["example"],
                    })
                    break

        if fuzzy_matches:
            result = {
                "found_terms": len(fuzzy_matches),
                "terms": fuzzy_matches,
                "note": "以上为模糊匹配结果，可能不完全准确。具体以实际法律条文为准。",
            }
        else:
            result = {
                "found_terms": 0,
                "terms": [],
                "note": "未识别到已知法律术语。可尝试输入包含劳动法相关术语的文本。",
                "supported_terms_count": len(LEGAL_TERMS),
            }
    else:
        result = {
            "found_terms": len(found_terms),
            "terms": found_terms,
            "note": "以上解释仅供参考，具体以实际法律条文为准。如需了解详情，可进一步咨询。",
        }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ==================== 统一接口 ====================

def legal_term_skill(query: str) -> str:
    """统一接口：接受 query，内部作为文本分析"""
    return legal_term_explainer.invoke({"text": query})


# 导出的工具列表
LEGAL_TERM_TOOLS = [legal_term_explainer]
