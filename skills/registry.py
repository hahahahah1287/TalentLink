# -*- coding: utf-8 -*-
"""
Skill Registry — 技能注册表

统一管理所有 skills 的注册、发现和调用。
添加新 skill 只需一行 registry.register(...)，无需修改 Planner、execute 节点或其他文件。
"""
from typing import Callable, Optional


class SkillRegistry:
    """
    技能注册表

    职责：
    - 注册 skill（名称 → 实现函数 + 描述）
    - 按名称查找 skill 函数
    - 生成 Planner prompt 用的选项列表
    - 提供所有 skill 的 LangChain Tool 列表

    使用方式：
        registry = SkillRegistry()
        registry.register("risk_clause_detector", skill_fn, "风险条款识别")

        # 在 Planner prompt 中使用
        options = registry.get_planner_options()

        # 在 execute 节点中动态调用
        fn = registry.get_skill_fn(step_name)
        if fn:
            result = fn(query)
    """

    def __init__(self):
        self._skills: dict[str, dict] = {}

    def register(
        self,
        name: str,
        fn: Callable[[str], str],
        description: str,
        category: str = "legal",
        complexity: str = "simple",
    ) -> None:
        """
        注册一个 skill

        Args:
            name: 唯一标识（如 "risk_clause_detector"）
            fn: 统一接口函数 fn(query: str) -> str
            description: 供 Planner 理解的功能描述
            category: 分类（"legal" | "search" | "utility"）
            complexity: 复杂度（"simple" 直接调用 | "complex" 需要 ReAct 执行）
        """
        self._skills[name] = {
            "fn": fn,
            "description": description,
            "category": category,
            "complexity": complexity,
        }

    def get_skill_fn(self, name: str) -> Optional[Callable[[str], str]]:
        """获取 skill 的实现函数"""
        entry = self._skills.get(name)
        return entry["fn"] if entry else None

    def get_description(self, name: str) -> Optional[str]:
        """获取 skill 的描述"""
        entry = self._skills.get(name)
        return entry["description"] if entry else None

    def get_all_skill_names(self) -> list[str]:
        """获取所有已注册的 skill 名称"""
        return list(self._skills.keys())

    def get_planner_options(self) -> str:
        """
        生成 Planner prompt 用的选项列表

        返回格式：
            - risk_clause_detector: 识别合同中的风险条款...
            - compliance_check: 检查用工场景是否合法...
        """
        lines = []
        for name, info in self._skills.items():
            lines.append(f"- {name}: {info['description']}")
        return "\n".join(lines)

    def is_registered(self, name: str) -> bool:
        """检查 skill 是否已注册"""
        return name in self._skills

    def is_complex(self, name: str) -> bool:
        """判断某个 skill 是否需要 ReAct 执行"""
        entry = self._skills.get(name)
        return entry.get("complexity", "simple") == "complex" if entry else False

    def get_complex_skills(self) -> list[str]:
        """返回需要 ReAct 执行的复杂 skill 名称"""
        return [name for name, info in self._skills.items() if info.get("complexity") == "complex"]

    def get_simple_skills(self) -> list[str]:
        """返回直接调用的简单 skill 名称"""
        return [name for name, info in self._skills.items() if info.get("complexity", "simple") == "simple"]

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        names = ", ".join(self._skills.keys())
        return f"SkillRegistry({len(self._skills)} skills: [{names}])"
