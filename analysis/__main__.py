"""命令行入口。

    python3 -m analysis run --data data --out runs [--cutoff 2026-04-01]
                            [--maturity-days 30] [--config config.json] [--run-id ID]
    python3 -m analysis compare runs/<旧运行> runs/<新运行> [--out diff.md]
    python3 -m analysis states --data data [--cutoff 2026-04-01]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from .analyze import analyze
from .config import AnalysisConfig
from .loader import DataError, load_inputs
from .prepare import prepare
from .report import STATE_LABELS, render_report
from .store import compare_runs, new_run_id, write_run


def _build_config(args: argparse.Namespace) -> AnalysisConfig:
    if getattr(args, "config", None):
        config = AnalysisConfig.from_json_file(args.config)
    else:
        config = AnalysisConfig()
    overrides = {}
    if getattr(args, "cutoff", None):
        overrides["cutoff"] = date.fromisoformat(args.cutoff)
    if getattr(args, "maturity_days", None) is not None:
        overrides["maturity_window_days"] = args.maturity_days
    if overrides:
        config = AnalysisConfig.from_dict({**config.to_dict(), **overrides})
    return config


def _cmd_run(args: argparse.Namespace) -> int:
    config = _build_config(args)
    inputs = load_inputs(args.data)
    prepared = prepare(inputs, config)
    results = analyze(prepared, inputs, config)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = args.run_id or new_run_id(inputs.manifest)
    report_md = render_report(results, run_id, inputs.manifest, generated_at)
    run_dir = write_run(
        args.out,
        run_id,
        manifest=inputs.manifest,
        config=config.to_dict(),
        results=results,
        report_md=report_md,
        generated_at=generated_at,
    )

    print(f"运行 ID：{run_id}")
    print(f"分析截止日：{results['cutoff']}；单位数：{len(results['units'])}")
    for segment, info in sorted(results["segments"].items()):
        pooled = info.get("pooled_stratified")
        if pooled:
            print(
                f"  {segment}：分层合并率差 {pooled['rd']:.4f}，"
                f"95% 区间 [{pooled['ci'][0]:.4f}, {pooled['ci'][1]:.4f}]"
            )
        else:
            print(f"  {segment}：无可比较的层，未给出效应估计")
    print(f"记录级异常 {len(results['anomalies'])} 条；详见报告。")
    print(f"报告：{run_dir / 'report.md'}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    text = compare_runs(args.run_a, args.run_b)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"比较结果已写入 {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_states(args: argparse.Namespace) -> int:
    config = _build_config(args)
    inputs = load_inputs(args.data)
    prepared = prepare(inputs, config)
    print(f"分析截止日：{prepared.cutoff}")
    header = ("分配记录", "单位", "策略版本段", "组别", "曝光状态", "结果状态")
    print(("{:<14}{:<10}{:<18}{:<12}{:<14}{:<14}").format(*header))
    for unit in prepared.units:
        print(
            ("{:<14}{:<10}{:<18}{:<12}{:<14}{:<14}").format(
                unit.assignment_ref,
                unit.unit_ref,
                unit.version_segment,
                unit.group,
                unit.exposure_state.value,
                f"{unit.outcome_state.value}（{STATE_LABELS[unit.outcome_state.value]}）",
            )
        )
    if prepared.anomalies:
        print("\n异常：")
        for item in prepared.anomalies:
            print(f"  - {item}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="对一批观测运行完整分析并保存运行目录")
    p_run.add_argument("--data", required=True, help="输入 CSV 所在目录")
    p_run.add_argument("--out", required=True, help="运行结果输出根目录")
    p_run.add_argument("--cutoff", help="分析截止日（ISO 日期）；缺省取数据中的最大日期")
    p_run.add_argument("--maturity-days", type=int, help="结果成熟窗口（天），默认 30")
    p_run.add_argument("--config", help="分析配置 JSON 文件")
    p_run.add_argument("--run-id", help="指定运行 ID（缺省按时间与输入哈希生成）")
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="比较两次运行，解释数据更正或新批次的影响")
    p_cmp.add_argument("run_a", help="旧运行目录")
    p_cmp.add_argument("run_b", help="新运行目录")
    p_cmp.add_argument("--out", help="将比较结果写入文件（缺省输出到标准输出）")
    p_cmp.set_defaults(func=_cmd_compare)

    p_states = sub.add_parser("states", help="只输出每个单位的状态核算表，不保存运行")
    p_states.add_argument("--data", required=True, help="输入 CSV 所在目录")
    p_states.add_argument("--cutoff", help="分析截止日（ISO 日期）")
    p_states.add_argument("--maturity-days", type=int, help="结果成熟窗口（天），默认 30")
    p_states.add_argument("--config", help="分析配置 JSON 文件")
    p_states.set_defaults(func=_cmd_states)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DataError, FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
