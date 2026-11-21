"""分析配置：终点定义、成熟窗口、截止日与不确定度参数。

配置可整体序列化为 JSON 并随每次运行保存，保证同一配置 + 同一输入
复算得到同一结果。未知配置键会报错，避免拼写错误被静默忽略。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path

# 正态分布 0.975 分位点（95% 双侧区间）。
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class AnalysisConfig:
    # 分析截止日；None 表示由输入数据中的最大日期推断（保证同数据复算一致）。
    cutoff: date | None = None
    # 结果成熟窗口（天）：自首次曝光（无曝光则自分配）起算。
    maturity_window_days: int = 30
    # 主要终点：事件记录需同时满足 kind / maturity / review 三组取值。
    event_kinds: tuple[str, ...] = ("complaint",)
    event_maturity: tuple[str, ...] = ("mature",)
    final_review: tuple[str, ...] = ("confirmed", "corrected")
    # 分类未最终确定的标记（任何 kind 适用）。
    pending_review: tuple[str, ...] = ("pending",)
    pending_maturity: tuple[str, ...] = ("immature", "pending_review")
    # 区间估计的 z 值与聚类数下限（低于该值时提示谨慎解释）。
    z: float = Z_95
    min_clusters: int = 30

    def to_dict(self) -> dict:
        return {
            "cutoff": self.cutoff.isoformat() if self.cutoff else None,
            "maturity_window_days": self.maturity_window_days,
            "event_kinds": list(self.event_kinds),
            "event_maturity": list(self.event_maturity),
            "final_review": list(self.final_review),
            "pending_review": list(self.pending_review),
            "pending_maturity": list(self.pending_maturity),
            "z": self.z,
            "min_clusters": self.min_clusters,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "AnalysisConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"配置包含未知键：{', '.join(unknown)}")
        data = dict(raw)
        if data.get("cutoff"):
            value = data["cutoff"]
            # 同时接受 ISO 字符串（配置文件）与 date 对象（命令行覆盖）。
            data["cutoff"] = value if isinstance(value, date) else date.fromisoformat(value)
        for key in (
            "event_kinds",
            "event_maturity",
            "final_review",
            "pending_review",
            "pending_maturity",
        ):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "AnalysisConfig":
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
