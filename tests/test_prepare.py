import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.analyze import analyze
from analysis.config import AnalysisConfig
from analysis.loader import load_inputs
from analysis.model import ExposureState, OutcomeState
from analysis.prepare import prepare

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _make_dataset(
    tmp: Path,
    assignments: list[list[str]],
    exposures: list[list[str]],
    outcomes: list[list[str]],
) -> Path:
    _write_csv(
        tmp / "policy_versions.csv",
        ["policy_ref", "version", "valid_from", "valid_to", "action_description"],
        [["P-1", "1", "2026-01-01", "2026-03-31", "直播提示"]],
    )
    _write_csv(
        tmp / "assignments.csv",
        ["assignment_ref", "unit_ref", "policy_ref", "assigned_at", "stratum", "assignment_group"],
        assignments,
    )
    _write_csv(
        tmp / "exposures.csv",
        ["exposure_ref", "user_ref", "session_ref", "policy_ref", "seen_at", "impression_count"],
        exposures,
    )
    _write_csv(
        tmp / "outcomes.csv",
        ["outcome_ref", "user_ref", "session_ref", "event_at", "outcome_kind", "maturity_state", "review_state"],
        outcomes,
    )
    return tmp


class RealDataStateTests(unittest.TestCase):
    """在仓库自带的合成数据上验证关键记录关系被正确区分。"""

    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs(ROOT / "data")
        cls.prepared = prepare(cls.inputs, AnalysisConfig())
        cls.units = {u.unit_ref: u for u in cls.prepared.units}

    def test_every_assigned_unit_is_analysed(self):
        self.assertEqual(len(self.prepared.units), 60)

    def test_unexposed_unit_with_outcome_is_flagged(self):
        u3 = self.units["U-3"]
        self.assertEqual(u3.exposure_state, ExposureState.UNEXPOSED)
        self.assertEqual(u3.outcome_state, OutcomeState.MATURE_EVENT)  # ITT 仍计入
        self.assertTrue(any("无曝光记录但存在结果记录" in a for a in u3.anomalies))

    def test_immature_complaint_is_pending_not_event(self):
        self.assertEqual(self.units["U-2"].outcome_state, OutcomeState.PENDING_REVIEW)

    def test_corrected_no_record_with_closed_window_is_non_event(self):
        self.assertEqual(self.units["U-5"].outcome_state, OutcomeState.NON_EVENT)

    def test_pending_refund_is_pending_review(self):
        self.assertEqual(self.units["U-4"].outcome_state, OutcomeState.PENDING_REVIEW)

    def test_mature_confirmed_complaint_is_event(self):
        self.assertEqual(self.units["U-1"].outcome_state, OutcomeState.MATURE_EVENT)

    def test_exposure_outside_validity_flagged(self):
        self.assertTrue(
            any("X-2" in a and "不在策略 P-2 任何版本" in a for a in self.prepared.anomalies)
        )

    def test_cross_policy_exposure_flagged(self):
        self.assertIn("X-4-B", self.units["U-4"].cross_policy_refs)

    def test_sessions_spanning_version_switch_detected(self):
        spanning = {s.session_ref for s in self.prepared.sessions if s.spans_version_switch}
        self.assertEqual(spanning, {"L-2", "L-3", "L-5", "L-10", "L-15", "L-17"})

    def test_policy_versions_split_segments(self):
        results = analyze(self.prepared, self.inputs, AnalysisConfig())
        self.assertIn("P-1@v1", results["segments"])
        self.assertIn("P-2@v1", results["segments"])
        self.assertIn("P-2@v2", results["segments"])
        # P-2 的 returning 层只有处理组一个单位，不可比较。
        self.assertIn("returning", results["segments"]["P-2@v1"]["non_comparable_strata"])
        self.assertEqual(
            sorted(results["segments"]["P-2@v1"]["strata"]), ["new"]
        )


class SyntheticStateTests(unittest.TestCase):
    """用小规模合成数据验证各状态分支（含仓库数据未覆盖的情形）。"""

    def _prepare(self, tmp: Path, cutoff: str):
        inputs = load_inputs(tmp)
        config = AnalysisConfig.from_dict({"cutoff": cutoff})
        prepared = prepare(inputs, config)
        return {u.unit_ref: u for u in prepared.units}

    def test_missing_window_open_after_cutoff_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = _make_dataset(
                Path(d),
                assignments=[
                    ["AS-1", "U-1", "P-1", "2026-01-01", "new", "treatment"],
                    ["AS-2", "U-2", "P-1", "2026-01-01", "new", "control"],
                    ["AS-3", "U-3", "P-1", "2026-02-20", "new", "treatment"],
                    ["AS-4", "U-4", "P-1", "2026-01-01", "new", "control"],
                    ["AS-5", "U-5", "P-1", "2026-01-01", "new", "treatment"],
                    ["AS-6", "U-6", "P-1", "2026-01-01", "new", "control"],
                ],
                exposures=[
                    ["X-1", "U-1", "L-1", "P-1", "2026-01-02", "1"],
                    ["X-2", "U-2", "L-1", "P-1", "2026-01-02", "1"],
                    ["X-3", "U-3", "L-2", "P-1", "2026-02-21", "1"],
                    ["X-4", "U-4", "L-1", "P-1", "2026-01-02", "1"],
                    ["X-5", "U-5", "L-1", "P-1", "2026-01-02", "1"],
                    ["X-6", "U-6", "L-1", "P-1", "2026-01-02", "1"],
                ],
                outcomes=[
                    # U-2：窗口已关闭但没有任何结果行 → MISSING
                    # U-3：窗口在截止日后才关闭且无记录 → WINDOW_OPEN
                    ["O-4", "U-4", "L-1", "2026-01-05", "no_record", "pending_review", "corrected"],
                    ["O-5", "U-5", "L-1", "2026-04-01", "complaint", "mature", "confirmed"],
                    ["O-6", "U-6", "L-1", "2026-01-05", "refund_request", "immature", "pending"],
                ],
            )
            units = self._prepare(tmp, cutoff="2026-03-01")
            self.assertEqual(units["U-2"].outcome_state, OutcomeState.MISSING)
            self.assertEqual(units["U-3"].outcome_state, OutcomeState.WINDOW_OPEN)
            self.assertEqual(units["U-4"].outcome_state, OutcomeState.NON_EVENT)
            # 事件日晚于截止日：本次运行不可观测，不能当作无投诉。
            self.assertEqual(units["U-5"].outcome_state, OutcomeState.AFTER_CUTOFF)
            self.assertEqual(units["U-6"].outcome_state, OutcomeState.PENDING_REVIEW)

    def test_unexposed_unit_uses_assignment_date_for_window(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = _make_dataset(
                Path(d),
                assignments=[["AS-1", "U-1", "P-1", "2026-01-01", "new", "treatment"]],
                exposures=[["X-9", "U-9", "L-1", "P-1", "2026-01-02", "1"]],
                outcomes=[],  # 仅表头：没有任何结果记录
            )
            units = self._prepare(tmp, cutoff="2026-03-01")
            self.assertEqual(units["U-1"].exposure_state, ExposureState.UNEXPOSED)
            self.assertEqual(units["U-1"].outcome_state, OutcomeState.MISSING)


if __name__ == "__main__":
    unittest.main()
