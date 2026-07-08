# -*- coding: utf-8 -*-
"""
Skill Registry — 技能注册表

统一登记法务 skill 的元信息，供确定性法务图的 run_skills 节点按名查找调用。

与旧版的区别：
- 不再服务于 LLM Planner（已删），因此去掉 get_planner_options / complexity 等
  "供 9B 自由调度"的概念。
- 注册时显式声明 skill 是否消费检索到的法条（uses_law_context），run_skills 据此传参，
  避免直调 skill 时把 law_context 丢掉。
"""
from typing import Callable, Optional, Dict, List


class SkillRegistry:
    """
    技能注册表

    职责：
    - 登记 skill（名称 → 统一接口函数 fn(query, law_context="") + 元信息）
    - 按名称查找 skill 函数 / 是否消费 law_context / 展示标题

    使用方式：
        registry = SkillRegistry()
        registry.register(
            "compliance_check", compliance_skill,
            "劳动法合规检查", uses_law_context=True, label="合规检查",
        )
        fn = registry.get_skill_fn("compliance_check")
    """

    def __init__(self):
        self._skills: Dict[str, dict] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., str],
        description: str,
        uses_law_context: bool = False,
        label: Optional[str] = None,
    ) -> None:
        """
        注册一个 skill

        Args:
            name: 唯一标识（如 "risk_clause_detector"）
            fn: 统一接口函数 fn(query: str, law_context: str = "") -> str
            description: 功能描述（文档/调试用）
            uses_law_context: 调用时是否需要把检索到的法条透传进去
            label: 给合成 LLM 的中文小标题（默认用 name）
        """
        self._skills[name] = {
            "fn": fn,
            "description": description,
            "uses_law_context": uses_law_context,
            "label": label or name,
        }

    def get_skill_fn(self, name: str) -> Optional[Callable[..., str]]:
        """获取 skill 的实现函数"""
        entry = self._skills.get(name)
        return entry["fn"] if entry else None

    def get_description(self, name: str) -> Optional[str]:
        """获取 skill 的描述"""
        entry = self._skills.get(name)
        return entry["description"] if entry else None

    def uses_law_context(self, name: str) -> bool:
        """该 skill 是否消费检索到的法条"""
        entry = self._skills.get(name)
        return bool(entry and entry["uses_law_context"])

    def get_label(self, name: str) -> Optional[str]:
        """获取 skill 的展示标题"""
        entry = self._skills.get(name)
        return entry["label"] if entry else None

    def get_all_skill_names(self) -> List[str]:
        """获取所有已注册的 skill 名称"""
        return list(self._skills.keys())

    def is_registered(self, name: str) -> bool:
        """检查 skill 是否已注册"""
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        names = ", ".join(self._skills.keys())
        return f"SkillRegistry({len(self._skills)} skills: [{names}])"
