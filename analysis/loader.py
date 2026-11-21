"""输入读取：CSV → 带出处（文件与行号）的领域记录，并生成输入清单。

清单（manifest）记录每个输入文件的 SHA-256 与行数，随运行保存；
原始资料被更正后，新旧运行的差异可定位到具体文件与记录。
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .model import Assignment, Exposure, Outcome, PolicyVersion, SourceRef

ASSIGNMENTS_FILE = "assignments.csv"
EXPOSURES_FILE = "exposures.csv"
OUTCOMES_FILE = "outcomes.csv"
POLICY_VERSIONS_FILE = "policy_versions.csv"

REQUIRED_COLUMNS = {
    ASSIGNMENTS_FILE: {
        "assignment_ref",
        "unit_ref",
        "policy_ref",
        "assigned_at",
        "stratum",
        "assignment_group",
    },
    EXPOSURES_FILE: {
        "exposure_ref",
        "user_ref",
        "session_ref",
        "policy_ref",
        "seen_at",
        "impression_count",
    },
    OUTCOMES_FILE: {
        "outcome_ref",
        "user_ref",
        "session_ref",
        "event_at",
        "outcome_kind",
        "maturity_state",
        "review_state",
    },
    POLICY_VERSIONS_FILE: {
        "policy_ref",
        "version",
        "valid_from",
        "valid_to",
        "action_description",
    },
}


class DataError(Exception):
    """输入资料不满足分析前提（缺列、主键重复、日期格式错误等）。"""


@dataclass
class Inputs:
    assignments: list[Assignment]
    exposures: list[Exposure]
    outcomes: list[Outcome]
    policy_versions: list[PolicyVersion]
    manifest: dict


def _parse_date(value: str | None, *, where: SourceRef, column: str) -> date | None:
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DataError(f"{where.label()} 列 {column} 的日期无法解析：{value!r}") from exc


def _read_rows(path: Path) -> list[tuple[dict, int]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise DataError(f"{path.name} 缺少表头")
        missing = REQUIRED_COLUMNS[path.name] - set(reader.fieldnames)
        if missing:
            raise DataError(f"{path.name} 缺少必需列：{', '.join(sorted(missing))}")
        # 表头为第 1 行，数据从第 2 行起。
        return [(row, line) for line, row in enumerate(reader, start=2)]


def _check_unique(refs: list[str], *, filename: str, column: str) -> None:
    seen: set[str] = set()
    for ref in refs:
        if ref in seen:
            raise DataError(f"{filename} 中 {column}={ref!r} 重复出现")
        seen.add(ref)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(data_dir: str | Path) -> Inputs:
    data_dir = Path(data_dir)
    tables: dict[str, list[tuple[dict, int]]] = {}
    manifest_files: dict[str, dict] = {}
    for name in REQUIRED_COLUMNS:
        path = data_dir / name
        if not path.is_file():
            raise DataError(f"缺少输入文件：{path}")
        tables[name] = _read_rows(path)
        manifest_files[name] = {
            "sha256": _file_digest(path),
            "rows": len(tables[name]),
        }

    assignments = [
        Assignment(
            assignment_ref=row["assignment_ref"],
            unit_ref=row["unit_ref"],
            policy_ref=row["policy_ref"],
            assigned_at=_parse_date(
                row["assigned_at"], where=SourceRef(ASSIGNMENTS_FILE, line), column="assigned_at"
            ),
            stratum=row["stratum"],
            group=row["assignment_group"],
            source=SourceRef(ASSIGNMENTS_FILE, line),
        )
        for row, line in tables[ASSIGNMENTS_FILE]
    ]
    exposures = [
        Exposure(
            exposure_ref=row["exposure_ref"],
            user_ref=row["user_ref"],
            session_ref=row["session_ref"],
            policy_ref=row["policy_ref"],
            seen_at=_parse_date(
                row["seen_at"], where=SourceRef(EXPOSURES_FILE, line), column="seen_at"
            ),
            impression_count=(
                int(row["impression_count"]) if row["impression_count"] not in (None, "") else None
            ),
            source=SourceRef(EXPOSURES_FILE, line),
        )
        for row, line in tables[EXPOSURES_FILE]
    ]
    outcomes = [
        Outcome(
            outcome_ref=row["outcome_ref"],
            user_ref=row["user_ref"],
            session_ref=row["session_ref"],
            event_at=_parse_date(
                row["event_at"], where=SourceRef(OUTCOMES_FILE, line), column="event_at"
            ),
            kind=row["outcome_kind"],
            maturity=row["maturity_state"],
            review=row["review_state"],
            source=SourceRef(OUTCOMES_FILE, line),
        )
        for row, line in tables[OUTCOMES_FILE]
    ]
    policy_versions = [
        PolicyVersion(
            policy_ref=row["policy_ref"],
            version=row["version"],
            valid_from=_parse_date(
                row["valid_from"], where=SourceRef(POLICY_VERSIONS_FILE, line), column="valid_from"
            ),
            valid_to=_parse_date(
                row["valid_to"], where=SourceRef(POLICY_VERSIONS_FILE, line), column="valid_to"
            ),
            action=row["action_description"],
            source=SourceRef(POLICY_VERSIONS_FILE, line),
        )
        for row, line in tables[POLICY_VERSIONS_FILE]
    ]

    _check_unique([a.assignment_ref for a in assignments], filename=ASSIGNMENTS_FILE, column="assignment_ref")
    _check_unique([e.exposure_ref for e in exposures], filename=EXPOSURES_FILE, column="exposure_ref")
    _check_unique([o.outcome_ref for o in outcomes], filename=OUTCOMES_FILE, column="outcome_ref")

    manifest = {"files": manifest_files}
    return Inputs(assignments, exposures, outcomes, policy_versions, manifest)
