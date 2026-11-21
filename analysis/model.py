"""领域模型与状态枚举。

状态设计原则：未曝光、无投诉记录、结果尚未成熟、缺少结果行是不同状态，
不得合并。所有记录保留来源位置（文件与行号），便于复核与更正后解释。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SourceRef:
    """一条记录在输入文件中的位置（行号为文件物理行，表头为第 1 行）。"""

    file: str
    line: int

    def label(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class Assignment:
    assignment_ref: str
    unit_ref: str
    policy_ref: str
    assigned_at: date | None
    stratum: str
    group: str
    source: SourceRef


@dataclass(frozen=True)
class Exposure:
    exposure_ref: str
    user_ref: str
    session_ref: str
    policy_ref: str
    seen_at: date | None
    impression_count: int | None
    source: SourceRef


@dataclass(frozen=True)
class Outcome:
    outcome_ref: str
    user_ref: str
    session_ref: str
    event_at: date | None
    kind: str
    maturity: str
    review: str
    source: SourceRef


@dataclass(frozen=True)
class PolicyVersion:
    policy_ref: str
    version: str
    valid_from: date | None
    valid_to: date | None
    action: str
    source: SourceRef

    def covers(self, day: date) -> bool:
        if self.valid_from is not None and day < self.valid_from:
            return False
        if self.valid_to is not None and day > self.valid_to:
            return False
        return True


class ExposureState(enum.Enum):
    """单位在分析截止日前是否有过被分配策略的曝光机会。"""

    EXPOSED = "exposed"
    UNEXPOSED = "unexposed"


class OutcomeState(enum.Enum):
    """单位的结果状态（针对主要终点，默认“成熟投诉”）。

    - MATURE_EVENT：有符合终点定义的最终事件记录（如成熟且已确认/已更正的投诉）。
    - NON_EVENT：有最终的非事件记录（如窗口已关闭的 no_record）。
    - PENDING_REVIEW：有记录但分类或成熟度未最终确定（复核中、未成熟）。
    - WINDOW_OPEN：观察窗口在截止日尚未关闭，事件仍可能到达。
    - MISSING：窗口已关闭但没有任何结果行（缺失 ≠ 无投诉）。
    - AFTER_CUTOFF：结果记录的事件时间晚于分析截止日，本次运行不可观测。
    """

    MATURE_EVENT = "mature_event"
    NON_EVENT = "non_event"
    PENDING_REVIEW = "pending_review"
    WINDOW_OPEN = "window_open"
    MISSING = "missing"
    AFTER_CUTOFF = "after_cutoff"


# 进入主要率估计的状态（分子/分母）；其余状态进入敏感性界限。
EVENT_STATES = frozenset({OutcomeState.MATURE_EVENT})
NON_EVENT_STATES = frozenset({OutcomeState.NON_EVENT})
UNRESOLVED_STATES = frozenset(
    {
        OutcomeState.PENDING_REVIEW,
        OutcomeState.WINDOW_OPEN,
        OutcomeState.MISSING,
        OutcomeState.AFTER_CUTOFF,
    }
)
