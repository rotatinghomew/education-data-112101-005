import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def _run_analysis(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "analysis", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _load_results(run_dir: Path) -> dict:
    return json.loads((run_dir / "results.json").read_text(encoding="utf-8"))


def _strip_volatile(payload: dict) -> dict:
    """去掉每次运行必然不同的字段后比较，验证复算一致性。"""
    return {k: v for k, v in payload.items() if k not in {"run_id", "generated_at"}}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="analysis-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = self.tmp / "data"
        shutil.copytree(ROOT / "data", self.data)
        self.out = self.tmp / "runs"

    def _run(self, run_id: str) -> Path:
        proc = _run_analysis(
            "run", "--data", str(self.data), "--out", str(self.out), "--run-id", run_id
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return self.out / run_id

    def test_run_writes_complete_run_directory(self):
        run_dir = self._run("run-a")
        for name in ("manifest.json", "config.json", "results.json", "report.md"):
            self.assertTrue((run_dir / name).is_file(), name)
        results = _load_results(run_dir)
        self.assertEqual(len(results["units"]), 60)
        self.assertIn("P-2@v2", results["segments"])
        self.assertTrue(results["limitations"])

    def test_rerun_on_same_data_is_reproducible(self):
        first = _strip_volatile(_load_results(self._run("run-a")))
        second = _strip_volatile(_load_results(self._run("run-b")))
        self.assertEqual(first, second)

    def test_outcome_reclassification_is_explained_by_compare(self):
        run_a = self._run("run-a")
        # 复核完成：O-2 由“未成熟/待复核”更正为“成熟/已确认”。
        outcomes = self.data / "outcomes.csv"
        text = outcomes.read_text(encoding="utf-8")
        text = text.replace(
            "O-2,U-2,L-3,2026-01-10,complaint,immature,pending",
            "O-2,U-2,L-3,2026-01-10,complaint,mature,confirmed",
        )
        outcomes.write_text(text, encoding="utf-8")
        run_b = self._run("run-b")

        proc = _run_analysis("compare", str(run_a), str(run_b))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("outcomes.csv 内容已变化", proc.stdout)
        self.assertIn("AS-2", proc.stdout)
        self.assertIn("pending_review → mature_event", proc.stdout)

    def test_assignment_correction_is_explained_by_compare(self):
        run_a = self._run("run-a")
        # 分配资料更正：AS-4 由对照改为处理。
        assignments = self.data / "assignments.csv"
        text = assignments.read_text(encoding="utf-8")
        text = text.replace(
            "AS-4,U-4,P-1,2026-01-05,returning,control",
            "AS-4,U-4,P-1,2026-01-05,returning,treatment",
        )
        assignments.write_text(text, encoding="utf-8")
        run_b = self._run("run-b")

        proc = _run_analysis("compare", str(run_a), str(run_b))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("assignments.csv 内容已变化", proc.stdout)
        self.assertIn("AS-4", proc.stdout)
        self.assertIn("分配组别：control → treatment", proc.stdout)

    def test_new_batch_extends_analysis(self):
        run_a = self._run("run-a")
        # 新一批观测：追加一个分配、曝光与成熟投诉。
        with (self.data / "assignments.csv").open("a", encoding="utf-8") as stream:
            stream.write("AS-61,U-61,P-1,2026-01-26,returning,treatment\n")
        with (self.data / "exposures.csv").open("a", encoding="utf-8") as stream:
            stream.write("X-61,U-61,L-1,P-1,2026-01-27,2\n")
        with (self.data / "outcomes.csv").open("a", encoding="utf-8") as stream:
            stream.write("O-61,U-61,L-1,2026-02-10,complaint,mature,confirmed\n")
        run_b = self._run("run-b")

        results = _load_results(run_b)
        self.assertEqual(len(results["units"]), 61)
        self.assertEqual(results["units"]["AS-61"]["outcome_state"], "mature_event")
        proc = _run_analysis("compare", str(run_a), str(run_b))
        self.assertIn("新增 1 个分配记录", proc.stdout)

    def test_states_command_runs_without_writing(self):
        proc = _run_analysis("states", "--data", str(self.data))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("U-3", proc.stdout)
        self.assertFalse(self.out.exists())

    def test_early_cutoff_marks_window_open_and_after_cutoff(self):
        proc = _run_analysis("states", "--data", str(self.data), "--cutoff", "2026-02-10")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("window_open", proc.stdout)
        self.assertIn("after_cutoff", proc.stdout)

    def test_bad_data_dir_returns_error_code(self):
        proc = _run_analysis("run", "--data", str(self.tmp / "nope"), "--out", str(self.out))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("错误", proc.stderr)


if __name__ == "__main__":
    unittest.main()
