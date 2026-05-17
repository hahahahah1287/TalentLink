# -*- coding: utf-8 -*-
"""
Agent 输出 Guardrails（输出防护管线）

职责:
1. PII 检测与脱敏 — 过滤 LLM 输出中的手机号、身份证号等敏感信息
2. 法律免责声明注入 — 当回复涉及法律建议时自动追加免责声明
3. 内容质量检查 — 检测回复是否过短、是否包含幻觉性表述

设计理念:
  采用管线（Pipeline）模式，每个 Guard 是一个独立的处理单元，
  按顺序通过所有 Guard 后才算最终输出。
  新增 Guard 只需实现 process() 方法并注册到管线。
"""
import re
from typing import List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class GuardResult:
    """单个 Guard 的处理结果"""
    modified: bool = False           # 是否修改了内容
    guard_name: str = ""             # Guard 名称
    details: str = ""                # 修改细节（日志用）
    output: str = ""                 # 处理后的文本


class BaseGuard:
    """Guard 基类"""
    name: str = "BaseGuard"

    def process(self, text: str, context: Optional[dict] = None) -> GuardResult:
        """
        处理文本

        Args:
            text: 待处理文本
            context: 可选的上下文信息（如 intent、user_id 等）

        Returns:
            GuardResult
        """
        raise NotImplementedError


class PIIGuard(BaseGuard):
    """
    PII (Personally Identifiable Information) 脱敏 Guard

    检测并遮蔽:
    - 手机号 (11 位)
    - 身份证号 (18 位)
    - 银行卡号 (16-19 位)
    - 邮箱地址
    """
    name = "PII脱敏"

    # 预编译正则提升性能
    PATTERNS = [
        # 手机号: 1[3-9] 开头的 11 位数字
        (re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)'), r'\g<0>'[:3] + '****' + r'\g<0>'[-4:], "手机号"),
        # 身份证号: 18 位（最后一位可能是 X）
        (re.compile(r'(?<!\d)(\d{6})((?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))(\d{3}[\dXx])(?!\d)'), None, "身份证号"),
        # 银行卡号: 16-19 位连续数字
        (re.compile(r'(?<!\d)(\d{4})(\d{8,11})(\d{4})(?!\d)'), None, "银行卡号"),
        # 邮箱
        (re.compile(r'([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'), None, "邮箱"),
    ]

    def process(self, text: str, context: Optional[dict] = None) -> GuardResult:
        modified_text = text
        modifications = []

        # 手机号脱敏: 138****1234
        phone_pattern = re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)')
        for match in phone_pattern.finditer(modified_text):
            phone = match.group(1)
            masked = phone[:3] + '****' + phone[-4:]
            modified_text = modified_text.replace(phone, masked, 1)
            modifications.append(f"手机号 {phone[:3]}***")

        # 身份证脱敏: 110***********1234
        id_pattern = re.compile(r'(?<!\d)(\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)')
        for match in id_pattern.finditer(modified_text):
            id_num = match.group(1)
            masked = id_num[:3] + '*' * 11 + id_num[-4:]
            modified_text = modified_text.replace(id_num, masked, 1)
            modifications.append("身份证号")

        # 银行卡脱敏: 6222 **** **** 1234
        bank_pattern = re.compile(r'(?<!\d)(\d{16,19})(?!\d)')
        for match in bank_pattern.finditer(modified_text):
            card = match.group(1)
            # 排除已经处理的手机号和身份证号
            if len(card) >= 16:
                masked = card[:4] + ' **** **** ' + card[-4:]
                modified_text = modified_text.replace(card, masked, 1)
                modifications.append("银行卡号")

        # 邮箱脱敏: u***@example.com
        email_pattern = re.compile(r'([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
        for match in email_pattern.finditer(modified_text):
            email = match.group(0)
            local = match.group(1)
            domain = match.group(2)
            masked = local[0] + '***@' + domain
            modified_text = modified_text.replace(email, masked, 1)
            modifications.append("邮箱")

        return GuardResult(
            modified=bool(modifications),
            guard_name=self.name,
            details=f"脱敏: {', '.join(modifications)}" if modifications else "",
            output=modified_text
        )


class DisclaimerGuard(BaseGuard):
    """
    法律免责声明注入 Guard

    当 AI 回复涉及法律建议时，自动追加标准免责声明，
    降低法律风险。
    """
    name = "免责声明"

    # 触发免责声明的关键词
    LEGAL_KEYWORDS = [
        "建议", "法律", "法规", "条款", "合同", "劳动法",
        "赔偿", "仲裁", "诉讼", "法院", "维权", "工伤",
        "社保", "公积金", "解除劳动", "经济补偿"
    ]

    DISCLAIMER = (
        "\n\n---\n"
        "⚠️ **免责声明**: 以上内容由 AI 生成，仅供参考，不构成专业法律意见。"
        "具体法律问题请咨询持证律师或前往当地法律援助中心。"
    )

    def process(self, text: str, context: Optional[dict] = None) -> GuardResult:
        # 检查是否已包含免责声明（避免重复添加）
        if "免责声明" in text:
            return GuardResult(output=text, guard_name=self.name)

        # 检查是否涉及法律内容
        contains_legal = any(kw in text for kw in self.LEGAL_KEYWORDS)

        # 也可通过 context 中的 intent 判断
        intent = (context or {}).get("intent", "")
        is_legal_intent = intent in ("job_search", "contract_critique")

        if contains_legal or is_legal_intent:
            return GuardResult(
                modified=True,
                guard_name=self.name,
                details="检测到法律相关内容，已追加免责声明",
                output=text + self.DISCLAIMER
            )

        return GuardResult(output=text, guard_name=self.name)


class QualityGuard(BaseGuard):
    """
    输出质量检查 Guard

    检测:
    - 回复过短（可能是模型截断或异常）
    - 重复内容
    """
    name = "质量检查"

    MIN_LENGTH = 20  # 最小回复长度

    def process(self, text: str, context: Optional[dict] = None) -> GuardResult:
        warnings = []

        # 检查回复长度
        if len(text.strip()) < self.MIN_LENGTH:
            warnings.append(f"回复过短({len(text.strip())}字)")

        # 检查重复内容（简单启发式：同一句子出现 3 次以上）
        sentences = text.split("。")
        seen = {}
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                seen[s] = seen.get(s, 0) + 1
                if seen[s] >= 3:
                    warnings.append("检测到重复内容")
                    break

        if warnings:
            return GuardResult(
                modified=False,
                guard_name=self.name,
                details="; ".join(warnings),
                output=text
            )

        return GuardResult(output=text, guard_name=self.name)


class CitationGuard(BaseGuard):
    """
    引用验证 Guard

    检查 LLM 输出中引用的法律条文是否存在于检索结果中。
    如果 LLM 引用了一条检索结果里没有的法律，标记为疑似幻觉。

    工作原理：
    1. 从 LLM 输出中提取法律引用（《法律名》第X条）
    2. 在检索上下文中查找这些引用
    3. 未找到的引用 → 标记为疑似幻觉

    通过 context["law_context"] 传入检索到的法律条文。
    """
    name = "引用验证"

    # 提取《法律名》
    LAW_NAME_PATTERN = re.compile(r'《([^》]+)》')
    # 提取第X条
    ARTICLE_PATTERN = re.compile(r'第(\d+)条')

    def process(self, text: str, context: Optional[dict] = None) -> GuardResult:
        law_context = (context or {}).get("law_context", "")

        # 没有检索上下文时跳过验证（比如普通聊天场景）
        if not law_context or len(law_context) < 10:
            return GuardResult(output=text, guard_name=self.name)

        warnings = []

        # 1. 检查法律名引用（仅当检索结果中也包含法律名时才检查）
        # 如果检索结果只有条文内容没有法律名字符串，跳过法律名检查避免误报
        cited_laws = set(self.LAW_NAME_PATTERN.findall(text))
        context_laws = set(self.LAW_NAME_PATTERN.findall(law_context))
        if context_laws:
            for law_name in cited_laws:
                if law_name not in law_context:
                    warnings.append(f"《{law_name}》未在检索结果中找到")

        # 2. 检查条文号引用（只在有法律名引用时才检查条文号）
        # 如果 LLM 说"第38条"但检索结果里没有第38条的内容，可能是编造
        cited_articles = set(self.ARTICLE_PATTERN.findall(text))
        if cited_articles and len(cited_articles) >= 3:
            # 如果引用了 3 条以上条文，抽查是否有对应内容
            missing_count = 0
            for article_num in cited_articles:
                # 在检索结果中查找该条文号
                if f"第{article_num}条" not in law_context:
                    missing_count += 1
            if missing_count > 0 and missing_count >= len(cited_articles) // 2:
                warnings.append(f"引用了{len(cited_articles)}个条文，其中{missing_count}个未在检索结果中找到")

        if warnings:
            detail_str = "; ".join(warnings)
            # 在输出前插入警告标记
            warning_text = (
                f"\n\n⚠️ [引用验证] 以下内容可能不准确：{detail_str}。"
                f"请以检索到的法律条文为准，以上建议仅供参考。"
            )
            return GuardResult(
                modified=True,
                guard_name=self.name,
                details=detail_str,
                output=text + warning_text
            )

        return GuardResult(output=text, guard_name=self.name)


class GuardrailsPipeline:
    """
    Guardrails 管线

    将多个 Guard 按顺序串联，文本依次通过每个 Guard 处理。

    使用示例:
        pipeline = GuardrailsPipeline()
        result = pipeline.run("这是AI的回复...", context={"intent": "job_search"})
        final_text = result["output"]
        print(result["guards_triggered"])  # 查看哪些 Guard 被触发
    """

    def __init__(self, guards: Optional[List[BaseGuard]] = None):
        """
        Args:
            guards: Guard 列表，按顺序执行。默认使用全部内置 Guard。
        """
        self.guards = guards or [
            PIIGuard(),
            QualityGuard(),
            CitationGuard(),      # 引用验证（需 context 中传入 law_context）
            DisclaimerGuard(),    # 免责声明放最后，因为它追加内容
        ]

    def run(self, text: str, context: Optional[dict] = None) -> dict:
        """
        执行 Guardrails 管线

        Args:
            text: 原始 LLM 输出
            context: 上下文信息

        Returns:
            {
                "output": 最终文本,
                "modified": 是否有任何修改,
                "guards_triggered": 被触发的 Guard 列表及详情
            }
        """
        current_text = text
        guards_triggered = []

        for guard in self.guards:
            try:
                result = guard.process(current_text, context)
                current_text = result.output

                if result.modified:
                    guards_triggered.append({
                        "guard": result.guard_name,
                        "details": result.details
                    })
                    print(f"🛡️ [Guardrails:{result.guard_name}] {result.details}")

            except Exception as e:
                print(f"⚠️ [Guardrails:{guard.name}] 执行异常: {e}")
                # Guard 异常不应阻止输出
                continue

        return {
            "output": current_text,
            "modified": bool(guards_triggered),
            "guards_triggered": guards_triggered
        }
