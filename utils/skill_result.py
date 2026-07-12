# -*- coding: utf-8 -*-
"""
Unified SkillResult schema helpers.

Skills keep their legacy top-level fields for backward compatibility, but also
expose the same audit-oriented envelope so downstream code can render, validate
and evaluate skill output without knowing each skill's private JSON shape.
"""
import json
from typing import Any, Dict, List, Optional


SKILL_RESULT_SCHEMA_VERSION = "skill-result-v1"


def make_skill_result(
    *,
    skill_name: str,
    facts: Optional[Dict[str, Any]] = None,
    findings: Optional[List[Dict[str, Any]]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    display_text: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    legacy: Optional[Dict[str, Any]] = None,
    status: str = "ok",
) -> Dict[str, Any]:
    """Build a unified skill result while preserving legacy fields."""
    result: Dict[str, Any] = {
        "schema_version": SKILL_RESULT_SCHEMA_VERSION,
        "skill_name": skill_name,
        "status": status,
        "facts": facts or {},
        "findings": findings or [],
        "evidence": evidence or [],
        "provenance": provenance or {},
        "display_text": display_text,
        "metrics": metrics or {},
    }
    if legacy:
        result["legacy"] = legacy
        # Backward compatibility: tests and old prompt rendering still read the
        # previous top-level keys such as risk_clauses / violations / cases.
        for key, value in legacy.items():
            result.setdefault(key, value)
    return result


def is_skill_result(value: Any) -> bool:
    return isinstance(value, dict) and value.get("schema_version") == SKILL_RESULT_SCHEMA_VERSION


def parse_skill_result(raw_output: str) -> Optional[Dict[str, Any]]:
    """Parse raw JSON skill output. Return None for legacy plain text outputs."""
    try:
        parsed = json.loads(raw_output)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_skill_result(skill_name: str, raw_output: Any) -> Dict[str, Any]:
    """Normalize arbitrary skill output into the unified SkillResult envelope."""
    if isinstance(raw_output, dict):
        parsed = raw_output
    else:
        parsed = parse_skill_result(str(raw_output))

    if is_skill_result(parsed):
        result = dict(parsed)
        if result.get("skill_name") != skill_name:
            result["skill_name"] = skill_name
        result.setdefault("status", "ok")
        result.setdefault("facts", {})
        result.setdefault("findings", [])
        result.setdefault("evidence", [])
        result.setdefault("provenance", {})
        result.setdefault("display_text", "")
        result.setdefault("metrics", {})
        return result

    if isinstance(parsed, dict):
        return make_skill_result(
            skill_name=skill_name,
            facts=parsed.get("facts") if isinstance(parsed.get("facts"), dict) else {},
            findings=parsed.get("findings") if isinstance(parsed.get("findings"), list) else [],
            evidence=parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else [],
            provenance=parsed.get("provenance") if isinstance(parsed.get("provenance"), dict) else {},
            display_text=parsed.get("display_text") or parsed.get("summary") or json.dumps(parsed, ensure_ascii=False),
            metrics=parsed.get("metrics") if isinstance(parsed.get("metrics"), dict) else {},
            legacy=parsed,
            status=parsed.get("status", "ok") if isinstance(parsed.get("status"), str) else "ok",
        )

    return make_skill_result(
        skill_name=skill_name,
        display_text=str(raw_output or ""),
        legacy={"raw_output": str(raw_output or "")},
        status="ok",
    )


def validate_and_clean_skill_result(
    result: Dict[str, Any],
    *,
    allowed_statuses: Optional[set] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate structure and drop malformed findings/evidence before synthesis."""
    allowed_statuses = allowed_statuses or {"ok", "warning", "error", "skipped"}
    clean = dict(result)
    corrections: List[Dict[str, Any]] = []
    dropped_findings: List[Dict[str, Any]] = []
    validation_errors: List[str] = []

    if clean.get("schema_version") != SKILL_RESULT_SCHEMA_VERSION:
        clean["schema_version"] = SKILL_RESULT_SCHEMA_VERSION
        corrections.append({"action": "set_schema_version", "value": SKILL_RESULT_SCHEMA_VERSION})

    status = clean.get("status")
    if status not in allowed_statuses:
        validation_errors.append(f"invalid_status:{status}")
        clean["status"] = "warning"
        corrections.append({"action": "normalize_status", "from": status, "to": "warning"})

    if not isinstance(clean.get("facts"), dict):
        clean["facts"] = {}
        corrections.append({"action": "drop_malformed_facts"})

    if not isinstance(clean.get("metrics"), dict):
        clean["metrics"] = {}
        corrections.append({"action": "drop_malformed_metrics"})

    if not isinstance(clean.get("provenance"), dict):
        clean["provenance"] = {}
        corrections.append({"action": "drop_malformed_provenance"})

    findings = clean.get("findings")
    valid_findings: List[Dict[str, Any]] = []
    if not isinstance(findings, list):
        validation_errors.append("findings_not_list")
        findings = []
        corrections.append({"action": "drop_malformed_findings"})

    for idx, finding in enumerate(findings):
        if not isinstance(finding, dict):
            dropped_findings.append({"index": idx, "reason": "finding_not_object"})
            continue
        has_content = any(str(finding.get(k) or "").strip() for k in ("title", "type", "message", "description", "risk", "issue"))
        if not has_content:
            dropped_findings.append({"index": idx, "reason": "empty_finding"})
            continue
        valid_findings.append(finding)
    clean["findings"] = valid_findings

    evidence = clean.get("evidence")
    valid_evidence: List[Dict[str, Any]] = []
    if not isinstance(evidence, list):
        validation_errors.append("evidence_not_list")
        evidence = []
        corrections.append({"action": "drop_malformed_evidence"})
    for idx, item in enumerate(evidence):
        if isinstance(item, dict):
            valid_evidence.append(item)
        else:
            corrections.append({"action": "drop_malformed_evidence_item", "index": idx})
    clean["evidence"] = valid_evidence

    report = {
        "specialist": clean.get("skill_name"),
        "status": clean.get("status", "ok"),
        "corrections": corrections,
        "dropped_findings": dropped_findings,
        "validation_errors": validation_errors,
        "input_finding_count": len(findings),
        "output_finding_count": len(valid_findings),
    }
    return clean, report


def render_skill_result_for_prompt(skill_name: str, raw_output: str) -> str:
    """Render a skill result compactly for the synthesis prompt."""
    parsed = parse_skill_result(raw_output)
    if not is_skill_result(parsed):
        return raw_output

    display = (parsed.get("display_text") or "").strip()
    findings = parsed.get("findings") or []
    facts = parsed.get("facts") or {}
    metrics = parsed.get("metrics") or {}

    parts: List[str] = []
    if display:
        parts.append(display)
    if findings:
        parts.append("结构化 findings：" + json.dumps(findings, ensure_ascii=False))
    if facts:
        parts.append("抽取 facts：" + json.dumps(facts, ensure_ascii=False))
    if metrics:
        parts.append("metrics：" + json.dumps(metrics, ensure_ascii=False))

    return "\n".join(parts) if parts else raw_output
