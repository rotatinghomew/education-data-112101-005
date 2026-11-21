"""把输入记录整理为可分析的单位记录与场次记录。

关键规则：
- 一个单位可能在多个场次接触不同策略；按分配（ITT）分析时不剔除，
  但跨策略曝光会记录为异常并单独统计。
- 场次可能跨越策略版本切换；按曝光发生时的有效版本归属。
- 未曝光、无投诉记录、结果未成熟、缺结果行是不同状态（见 model.OutcomeState）。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from .config import AnalysisConfig
from .loader import Inputs
from .model import (
    Exposure,
    ExposureState,
    Outcome,
    OutcomeState,
    PolicyVersion,
    SourceRef,
)


@dataclass
class UnitAnalysis:
    """一条分配记录（用户 × 策略）的完整分析视图。"""

    assignment_ref: str
    unit_ref: str
    policy_ref: str
    group: str
    stratum: str
    assigned_at: date | None
    version_segment: str  # 分配时刻的策略版本，如 "P-2@v1"；无有效版本为 "P-?@无有效版本"
    exposure_state: ExposureState
    first_exposure_at: date | None
    exposure_version: str | None  # 首次曝光时刻的策略版本
    impressions: int | None
    outcome_state: OutcomeState
    outcome_ref: str | None
    outcome_at: date | None
    window_end: date | None
    cluster_key: str
    cross_policy_refs: tuple[str, ...]
    anomalies: tuple[str, ...]
    source: SourceRef

    def to_dict(self) -> dict:
        return {
            "assignment_ref": self.assignment_ref,
            "unit_ref": self.unit_ref,
            "policy_ref": self.policy_ref,
            "group": self.group,
            "stratum": self.stratum,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "version_segment": self.version_segment,
            "exposure_state": self.exposure_state.value,
            "first_exposure_at": (
                self.first_exposure_at.isoformat() if self.first_exposure_at else None
            ),
            "exposure_version": self.exposure_version,
            "impressions": self.impressions,
            "outcome_state": self.outcome_state.value,
            "outcome_ref": self.outcome_ref,
            "outcome_at": self.outcome_at.isoformat() if self.outcome_at else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "cluster_key": self.cluster_key,
            "cross_policy_refs": list(self.cross_policy_refs),
            "anomalies": list(self.anomalies),
            "source": {"file": self.source.file, "line": self.source.line},
        }


@dataclass
class SessionAnalysis:
    session_ref: str
    policies: tuple[str, ...]
    versions: tuple[str, ...]  # 形如 "P-2@v1"
    n_exposures: int
    n_users: int
    spans_policy_switch: bool
    spans_version_switch: bool

    def to_dict(self) -> dict:
        return {
            "session_ref": self.session_ref,
            "policies": list(self.policies),
            "versions": list(self.versions),
            "n_exposures": self.n_exposures,
            "n_users": self.n_users,
            "spans_policy_switch": self.spans_policy_switch,
            "spans_version_switch": self.spans_version_switch,
        }


@dataclass
class Prepared:
    units: list[UnitAnalysis]
    sessions: list[SessionAnalysis]
    anomalies: list[str]
    cutoff: date


def _version_at(versions: list[PolicyVersion], day: date | None) -> PolicyVersion | None:
    if day is None:
        return None
    for version in versions:
        if version.covers(day):
            return version
    return None


def _segment_label(policy_ref: str, version: PolicyVersion | None) -> str:
    if version is None:
        return f"{policy_ref}@无有效版本"
    return f"{policy_ref}@v{version.version}"


def _classify_outcome(
    rows: list[Outcome], window_closed: bool, config: AnalysisConfig, cutoff: date
) -> tuple[OutcomeState, Outcome | None]:
    """按优先级判定单位的结果状态，返回（状态， 决定性记录）。"""
    observable = [r for r in rows if r.event_at is not None and r.event_at <= cutoff]
    if rows and not observable:
        return OutcomeState.AFTER_CUTOFF, None
    for row in observable:
        if (
            row.kind in config.event_kinds
            and row.maturity in config.event_maturity
            and row.review in config.final_review
        ):
            return OutcomeState.MATURE_EVENT, row
    for row in observable:
        if row.review in config.pending_review:
            return OutcomeState.PENDING_REVIEW, row
        if row.kind in config.event_kinds and row.maturity in config.pending_maturity:
            return OutcomeState.PENDING_REVIEW, row
    if observable:
        # 有最终记录但不是终点事件（如窗口已关闭的 no_record）。
        return (OutcomeState.NON_EVENT if window_closed else OutcomeState.WINDOW_OPEN), observable[0]
    # 没有任何结果行：缺失与“无投诉记录”不是同一状态。
    return (OutcomeState.MISSING if window_closed else OutcomeState.WINDOW_OPEN), None


def prepare(inputs: Inputs, config: AnalysisConfig) -> Prepared:
    all_dates = [
        d
        for d in (
            [a.assigned_at for a in inputs.assignments]
            + [e.seen_at for e in inputs.exposures]
            + [o.event_at for o in inputs.outcomes]
        )
        if d is not None
    ]
    if config.cutoff is not None:
        cutoff = config.cutoff
    elif all_dates:
        cutoff = max(all_dates)
    else:
        raise ValueError("无法推断分析截止日：输入没有任何日期，请在配置中指定 cutoff")

    versions_by_policy: dict[str, list[PolicyVersion]] = defaultdict(list)
    for version in inputs.policy_versions:
        versions_by_policy[version.policy_ref].append(version)

    exposures_by_user: dict[str, list[Exposure]] = defaultdict(list)
    for exposure in inputs.exposures:
        exposures_by_user[exposure.user_ref].append(exposure)
    outcomes_by_user: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in inputs.outcomes:
        outcomes_by_user[outcome.user_ref].append(outcome)

    anomalies: list[str] = []

    # 曝光日期是否落在该策略任一版本有效期内（对全部曝光检查，含跨策略曝光）。
    exposure_version: dict[str, PolicyVersion | None] = {}
    for exposure in inputs.exposures:
        version = _version_at(versions_by_policy.get(exposure.policy_ref, []), exposure.seen_at)
        exposure_version[exposure.exposure_ref] = version
        if version is None:
            anomalies.append(
                f"曝光 {exposure.exposure_ref}（{exposure.source.label()}）日期 "
                f"{exposure.seen_at} 不在策略 {exposure.policy_ref} 任何版本的有效期内"
            )

    units: list[UnitAnalysis] = []
    for assignment in inputs.assignments:
        unit_anomalies: list[str] = []
        versions = versions_by_policy.get(assignment.policy_ref, [])
        assigned_version = _version_at(versions, assignment.assigned_at)
        if assigned_version is None:
            unit_anomalies.append(
                f"分配 {assignment.assignment_ref} 日期 {assignment.assigned_at} 不在策略 "
                f"{assignment.policy_ref} 任何版本有效期内"
            )

        user_exposures = exposures_by_user.get(assignment.unit_ref, [])
        own = [
            e
            for e in user_exposures
            if e.policy_ref == assignment.policy_ref
            and e.seen_at is not None
            and e.seen_at <= cutoff
        ]
        cross = tuple(
            sorted(e.exposure_ref for e in user_exposures if e.policy_ref != assignment.policy_ref)
        )
        if cross:
            unit_anomalies.append(
                f"单位 {assignment.unit_ref} 存在跨策略曝光：{', '.join(cross)}"
            )
        for e in user_exposures:
            # 只有“被分配策略”的曝光早于分配时间才是记录矛盾；
            # 跨策略曝光早于分配是正常的历史接触，已单独标记。
            if (
                e.policy_ref == assignment.policy_ref
                and assignment.assigned_at is not None
                and e.seen_at is not None
                and e.seen_at < assignment.assigned_at
            ):
                unit_anomalies.append(
                    f"曝光 {e.exposure_ref}（{e.source.label()}）早于分配时间 "
                    f"{assignment.assigned_at}"
                )

        own.sort(key=lambda e: (e.seen_at, e.exposure_ref))
        exposure_state = ExposureState.EXPOSED if own else ExposureState.UNEXPOSED
        first_exposure_at = own[0].seen_at if own else None
        first_exposure_version = (
            exposure_version.get(own[0].exposure_ref) if own else None
        )
        if (
            own
            and assigned_version is not None
            and first_exposure_version is not None
            and first_exposure_version.version != assigned_version.version
        ):
            unit_anomalies.append(
                f"单位 {assignment.unit_ref} 分配时版本为 v{assigned_version.version}，"
                f"首次曝光时版本为 v{first_exposure_version.version}"
            )
        impressions = [e.impression_count for e in own if e.impression_count is not None]
        impression_total = sum(impressions) if impressions else None

        # 观察窗口自首次曝光起算；无曝光单位自分配起算（ITT 随访起点）。
        window_start = first_exposure_at or assignment.assigned_at
        window_end = (
            window_start + timedelta(days=config.maturity_window_days)
            if window_start is not None
            else None
        )
        window_closed = window_end is not None and window_end <= cutoff

        user_outcomes = outcomes_by_user.get(assignment.unit_ref, [])
        outcome_state, decisive = _classify_outcome(user_outcomes, window_closed, config, cutoff)
        if user_outcomes and exposure_state is ExposureState.UNEXPOSED:
            refs = ", ".join(o.outcome_ref for o in user_outcomes)
            unit_anomalies.append(
                f"单位 {assignment.unit_ref} 无曝光记录但存在结果记录：{refs}"
            )

        if own:
            cluster_key = f"S:{own[0].session_ref}"
        elif user_outcomes and user_outcomes[0].session_ref:
            cluster_key = f"S:{user_outcomes[0].session_ref}"
        else:
            cluster_key = f"U:{assignment.unit_ref}"

        units.append(
            UnitAnalysis(
                assignment_ref=assignment.assignment_ref,
                unit_ref=assignment.unit_ref,
                policy_ref=assignment.policy_ref,
                group=assignment.group,
                stratum=assignment.stratum,
                assigned_at=assignment.assigned_at,
                version_segment=_segment_label(assignment.policy_ref, assigned_version),
                exposure_state=exposure_state,
                first_exposure_at=first_exposure_at,
                exposure_version=(
                    f"v{first_exposure_version.version}" if first_exposure_version else None
                ),
                impressions=impression_total,
                outcome_state=outcome_state,
                outcome_ref=decisive.outcome_ref if decisive else None,
                outcome_at=decisive.event_at if decisive else None,
                window_end=window_end,
                cluster_key=cluster_key,
                cross_policy_refs=cross,
                anomalies=tuple(unit_anomalies),
                source=assignment.source,
            )
        )

    sessions: list[SessionAnalysis] = []
    exposures_by_session: dict[str, list[Exposure]] = defaultdict(list)
    for exposure in inputs.exposures:
        exposures_by_session[exposure.session_ref].append(exposure)
    for session_ref in sorted(exposures_by_session):
        rows = exposures_by_session[session_ref]
        policies = sorted({e.policy_ref for e in rows})
        versions = sorted(
            {
                f"{e.policy_ref}@v{exposure_version[e.exposure_ref].version}"
                for e in rows
                if exposure_version.get(e.exposure_ref) is not None
            }
        )
        versioned_policies = defaultdict(set)
        for e in rows:
            v = exposure_version.get(e.exposure_ref)
            if v is not None:
                versioned_policies[e.policy_ref].add(v.version)
        sessions.append(
            SessionAnalysis(
                session_ref=session_ref,
                policies=tuple(policies),
                versions=tuple(versions),
                n_exposures=len(rows),
                n_users=len({e.user_ref for e in rows}),
                spans_policy_switch=len(policies) > 1,
                spans_version_switch=any(len(v) > 1 for v in versioned_policies.values()),
            )
        )

    anomalies.extend(
        anomaly for unit in units for anomaly in unit.anomalies
    )
    return Prepared(units=units, sessions=sessions, anomalies=anomalies, cutoff=cutoff)
