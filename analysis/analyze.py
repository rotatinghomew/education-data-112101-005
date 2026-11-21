"""按策略版本段、分层与组别汇总状态并计算效应估计。

所有结果组织为可 JSON 序列化的字典，随运行保存；报告（report.py）
与运行间比较（store.py）都只消费这份结构化结果。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date

from .config import AnalysisConfig
from .estimators import (
    cluster_robust_se,
    manski_bounds,
    newcombe_rd_interval,
    pooled_effect,
    wilson_interval,
)
from .model import EVENT_STATES, NON_EVENT_STATES, UNRESOLVED_STATES, OutcomeState
from .prepare import Prepared, UnitAnalysis

STATE_ORDER = [state.value for state in OutcomeState]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _arm_block(units: list[UnitAnalysis], config: AnalysisConfig) -> dict:
    n = len(units)
    states = Counter(u.outcome_state.value for u in units)
    events = sum(states[s.value] for s in EVENT_STATES)
    non_events = sum(states[s.value] for s in NON_EVENT_STATES)
    unresolved = sum(states[s.value] for s in UNRESOLVED_STATES)
    evaluable = events + non_events
    rate = events / evaluable if evaluable else None
    exposed = sum(1 for u in units if u.exposure_state.value == "exposed")
    delays = [
        (u.outcome_at - u.first_exposure_at).days
        for u in units
        if u.outcome_state in EVENT_STATES
        and u.outcome_at is not None
        and u.first_exposure_at is not None
    ]
    return {
        "n_assigned": n,
        "n_exposed": exposed,
        "exposure_rate": exposed / n if n else None,
        "exposure_rate_ci": list(wilson_interval(exposed, n, config.z)) if n else None,
        "states": {key: states.get(key, 0) for key in STATE_ORDER},
        "n_events": events,
        "n_evaluable": evaluable,
        "event_rate": rate,
        "event_rate_ci": (
            list(wilson_interval(events, evaluable, config.z)) if evaluable else None
        ),
        "manski_bounds": list(manski_bounds(events, non_events, unresolved)),
        "event_delay_days": (
            {
                "n": len(delays),
                "min": min(delays),
                "median": _median(delays),
                "max": max(delays),
            }
            if delays
            else None
        ),
        "n_cross_policy_exposed": sum(1 for u in units if u.cross_policy_refs),
    }


def _effect_block(
    index_units: list[UnitAnalysis],
    ref_units: list[UnitAnalysis],
    config: AnalysisConfig,
) -> dict:
    """index 臂相对 reference 臂的效应（率差）。"""
    index = _arm_block(index_units, config)
    ref = _arm_block(ref_units, config)
    block: dict = {"index": index, "reference": ref}
    k1, m1 = index["n_events"], index["n_evaluable"]
    k0, m0 = ref["n_events"], ref["n_evaluable"]
    if not m1 or not m0:
        block["rd"] = None
        block["note"] = "至少一臂没有可评估单位，无法计算率差"
        return block

    rd, lo, hi = newcombe_rd_interval(k1, m1, k0, m0, config.z)
    block["rd"] = rd
    block["newcombe_ci"] = [lo, hi]
    block["manski_rd_bounds"] = [
        index["manski_bounds"][0] - ref["manski_bounds"][1],
        index["manski_bounds"][1] - ref["manski_bounds"][0],
    ]

    samples = [
        (u.cluster_key, 1, 1.0 if u.outcome_state in EVENT_STATES else 0.0)
        for u in index_units
        if u.outcome_state in EVENT_STATES | NON_EVENT_STATES
    ] + [
        (u.cluster_key, 0, 1.0 if u.outcome_state in EVENT_STATES else 0.0)
        for u in ref_units
        if u.outcome_state in EVENT_STATES | NON_EVENT_STATES
    ]
    cr = cluster_robust_se(samples)
    if cr is None:
        block["cluster_robust"] = None
        block["cluster_note"] = "聚类过少或单臂为空，无法计算聚类稳健标准误"
    else:
        se, n_clusters = cr
        block["cluster_robust"] = {
            "se": se,
            "n_clusters": n_clusters,
            "ci": [rd - config.z * se, rd + config.z * se],
        }
        if se == 0.0:
            block["cluster_note"] = (
                "可评估单位的结果在聚类内无变异，标准误为 0；"
                "这不代表效果确定，请以 Newcombe 区间与 Manski 界限为准"
            )
        elif n_clusters < config.min_clusters:
            block["cluster_note"] = (
                f"聚类数 {n_clusters} 少于 {config.min_clusters}，"
                "聚类稳健标准误可能低估不确定度，请谨慎解释"
            )
    return block


def _stratified_pooled(
    stratum_effects: dict[str, dict], config: AnalysisConfig
) -> dict | None:
    """按方差倒数加权合并各层率差（层内 Newcombe 区间反推方差）。"""
    rows = []
    for name, effect in sorted(stratum_effects.items()):
        if effect.get("rd") is None:
            continue
        lo, hi = effect["newcombe_ci"]
        var = ((hi - lo) / (2.0 * config.z)) ** 2
        rows.append((name, effect["rd"], var))
    if not rows:
        return None
    pooled = pooled_effect([(rd, var) for _, rd, var in rows])
    if pooled is None:
        return None
    estimate, var = pooled
    signs = {1 if rd > 0 else -1 if rd < 0 else 0 for _, rd, _ in rows}
    return {
        "rd": estimate,
        "se": var**0.5,
        "ci": [estimate - config.z * var**0.5, estimate + config.z * var**0.5],
        "weights": {name: round(1.0 / var, 6) for name, _, var in rows},
        "heterogeneity_note": (
            "各层效应方向不一致，合并估计掩盖了层间差异，请按层解读"
            if len(signs - {0}) > 1
            else None
        ),
    }


def analyze(prepared: Prepared, inputs, config: AnalysisConfig) -> dict:
    versions_by_policy: dict[str, dict[str, str]] = defaultdict(dict)
    for version in inputs.policy_versions:
        versions_by_policy[version.policy_ref][version.version] = version.action

    units_by_segment: dict[str, list[UnitAnalysis]] = defaultdict(list)
    for unit in prepared.units:
        # version_segment 形如 "P-2@v1"；无有效版本时为 "P-2@无有效版本"。
        units_by_segment[unit.version_segment].append(unit)

    segments: dict[str, dict] = {}
    for segment, units in sorted(units_by_segment.items()):
        policy_ref = units[0].policy_ref
        version_key = segment.split("@", 1)[1]
        version_num = version_key[1:] if version_key.startswith("v") else None
        arms = sorted({u.group for u in units})
        reference = "control" if "control" in arms else (arms[0] if arms else None)
        index_arms = [a for a in arms if a != reference]

        strata = sorted({u.stratum for u in units})
        comparable: dict[str, dict] = {}
        non_comparable: dict[str, dict] = {}
        for stratum in strata:
            in_stratum = [u for u in units if u.stratum == stratum]
            counts = Counter(u.group for u in in_stratum)
            if reference is None or len(counts) < 2 or len(arms) != 2:
                non_comparable[stratum] = {
                    "counts": dict(counts),
                    "reason": "层内缺少可比较的组别，无法构成对照",
                }
                continue
            index_units = [u for u in in_stratum if u.group == index_arms[0]]
            ref_units = [u for u in in_stratum if u.group == reference]
            comparable[stratum] = _effect_block(index_units, ref_units, config)

        overall = None
        pooled = None
        if len(arms) == 2 and comparable:
            index_all = [
                u for u in units if u.group == index_arms[0] and u.stratum in comparable
            ]
            ref_all = [u for u in units if u.group == reference and u.stratum in comparable]
            overall = _effect_block(index_all, ref_all, config)
            pooled = _stratified_pooled(
                {name: eff for name, eff in comparable.items()}, config
            )

        segments[segment] = {
            "policy_ref": policy_ref,
            "version": version_num,
            "action": versions_by_policy.get(policy_ref, {}).get(version_num),
            "arms": arms,
            "reference_arm": reference,
            "n_assigned": len(units),
            "strata": comparable,
            "non_comparable_strata": non_comparable,
            "overall_unadjusted": overall,
            "pooled_stratified": pooled,
        }

    limitations = _build_limitations(prepared, inputs, segments, config)
    return {
        "cutoff": prepared.cutoff.isoformat(),
        "config": config.to_dict(),
        "units": {u.assignment_ref: u.to_dict() for u in prepared.units},
        "sessions": {s.session_ref: s.to_dict() for s in prepared.sessions},
        "segments": segments,
        "anomalies": prepared.anomalies,
        "limitations": limitations,
    }


def _build_limitations(prepared, inputs, segments, config) -> list[str]:
    limitations: list[str] = []

    # 不同策略的分配时间是否重叠：不重叠则跨策略比较存在时间混杂。
    ranges: dict[str, tuple[date, date]] = {}
    for unit in prepared.units:
        if unit.assigned_at is None:
            continue
        lo, hi = ranges.get(
            unit.policy_ref, (unit.assigned_at, unit.assigned_at)
        )
        ranges[unit.policy_ref] = (min(lo, unit.assigned_at), max(hi, unit.assigned_at))
    refs = sorted(ranges)
    for i, left in enumerate(refs):
        for right in refs[i + 1 :]:
            if ranges[left][1] < ranges[right][0] or ranges[right][1] < ranges[left][0]:
                limitations.append(
                    f"策略 {left}（{ranges[left][0]}~{ranges[left][1]}）与 {right}"
                    f"（{ranges[right][0]}~{ranges[right][1]}）的分配时间不重迭，"
                    "两者效果差异不能归因于措施本身（时间混杂）"
                )

    # 同一策略的不同版本是不同干预。
    for segment, info in sorted(segments.items()):
        others = [s for s in segments if s.startswith(info["policy_ref"] + "@") and s != segment]
        if others:
            limitations.append(
                f"策略 {info['policy_ref']} 存在多个版本段（{', '.join(sorted([segment] + others))}），"
                "版本间差异是前后对照的观察性对比，不是随机分配的结果"
            )
            break

    # 适用人群：各段只覆盖部分层。
    for segment, info in sorted(segments.items()):
        if info["non_comparable_strata"]:
            missing = "、".join(sorted(info["non_comparable_strata"]))
            limitations.append(
                f"{segment} 的估计仅适用于可比较层"
                f"（{', '.join(sorted(info['strata'])) or '无'}）；"
                f"层 {missing} 内缺少对照，结论不能外推"
            )

    # 未决结果。
    unresolved = sum(1 for u in prepared.units if u.outcome_state in UNRESOLVED_STATES)
    if unresolved:
        limitations.append(
            f"有 {unresolved} 个单位的结果未决（待复核/窗口未闭合/缺记录/截止后），"
            "主要估计未包含这些单位；其影响以 Manski 界限呈现"
        )

    # 聚类数。
    for segment, info in sorted(segments.items()):
        overall = info.get("overall_unadjusted")
        if overall and overall.get("cluster_robust"):
            n_clusters = overall["cluster_robust"]["n_clusters"]
            if n_clusters < config.min_clusters:
                limitations.append(
                    f"{segment} 的聚类数（{n_clusters}）较少，"
                    "聚类稳健标准误为小样本近似"
                )

    if any(u.cross_policy_refs for u in prepared.units):
        limitations.append(
            "存在用户在被分配策略之外接触其他策略的记录；"
            "按曝光条件化（仅看已曝光者）的比较会破坏分配的可比性，"
            "本报告仅将其作为描述性信息列出"
        )
    limitations.append("输入为合成数据，所有数值仅用于演示分析流程，不构成任何业务结论")
    return limitations
