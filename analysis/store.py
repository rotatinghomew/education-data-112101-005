"""运行存档与运行间比较。

每次运行写入独立目录：

    runs/<run_id>/manifest.json   输入文件哈希与行数
    runs/<run_id>/config.json     分析配置
    runs/<run_id>/results.json    结构化结果（含每个单位的状态与出处）
    runs/<run_id>/report.md       报告

原始资料更正或新批次到达后，用同一命令产生新运行，再用 compare
逐项解释两次运行的差异，而不是覆盖旧结论。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_FILE = "manifest.json"
CONFIG_FILE = "config.json"
RESULTS_FILE = "results.json"
REPORT_FILE = "report.md"

_UNIT_COMPARE_FIELDS = (
    ("group", "分配组别"),
    ("stratum", "分层"),
    ("policy_ref", "策略"),
    ("version_segment", "策略版本段"),
    ("exposure_state", "曝光状态"),
    ("outcome_state", "结果状态"),
)


def new_run_id(manifest: dict, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    digest = manifest_digest(manifest)
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{digest}"


def manifest_digest(manifest: dict) -> str:
    import hashlib

    payload = json.dumps(manifest["files"], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def write_run(
    out_root: str | Path,
    run_id: str,
    *,
    manifest: dict,
    config: dict,
    results: dict,
    report_md: str,
    generated_at: str,
) -> Path:
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    envelope = {"run_id": run_id, "generated_at": generated_at, **results}
    _write_json(run_dir / MANIFEST_FILE, manifest)
    _write_json(run_dir / CONFIG_FILE, config)
    _write_json(run_dir / RESULTS_FILE, envelope)
    (run_dir / REPORT_FILE).write_text(report_md, encoding="utf-8")
    # 记录最近一次运行，便于脚本引用（纯文本，避免符号链接的平台差异）。
    (Path(out_root) / "LATEST").write_text(run_id + "\n", encoding="utf-8")
    return run_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    for name in (MANIFEST_FILE, CONFIG_FILE, RESULTS_FILE):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{run_dir} 不是完整的运行目录：缺少 {name}")
    return {
        "dir": str(run_dir),
        "manifest": json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8")),
        "config": json.loads((run_dir / CONFIG_FILE).read_text(encoding="utf-8")),
        "results": json.loads((run_dir / RESULTS_FILE).read_text(encoding="utf-8")),
    }


def compare_runs(dir_a: str | Path, dir_b: str | Path) -> str:
    """生成两次运行的差异说明（Markdown）。A 为旧运行，B 为新运行。"""
    a = load_run(dir_a)
    b = load_run(dir_b)
    lines: list[str] = []
    add = lines.append
    id_a = a["results"].get("run_id", str(dir_a))
    id_b = b["results"].get("run_id", str(dir_b))

    add("# 运行比较")
    add("")
    add(f"- 旧运行 A：`{id_a}`（截止日 {a['results'].get('cutoff')}）")
    add(f"- 新运行 B：`{id_b}`（截止日 {b['results'].get('cutoff')}）")
    add("")

    _compare_manifests(add, a["manifest"], b["manifest"])
    _compare_configs(add, a["config"], b["config"])
    changed_units = _compare_units(add, a["results"], b["results"])
    _compare_estimates(add, a["results"], b["results"])

    add("## 结论性说明")
    add("")
    if changed_units:
        add(
            f"共有 {len(changed_units)} 个单位的分析状态在两次运行间发生变化；"
            "基于旧运行作出的判断应在新运行下重新评估。两次运行的完整内容均保留在磁盘上，"
            "可逐单位复查（results.json 中每个单位记录了来源文件与行号）。"
        )
    else:
        add("两次运行的单位级状态一致；若输入哈希有变化，请核对是否为无分析影响的格式调整。")
    add("")
    return "\n".join(lines) + "\n"


def _compare_manifests(add, manifest_a: dict, manifest_b: dict) -> None:
    add("## 输入文件变化")
    add("")
    files_a = manifest_a.get("files", {})
    files_b = manifest_b.get("files", {})
    changed = False
    for name in sorted(set(files_a) | set(files_b)):
        in_a, in_b = files_a.get(name), files_b.get(name)
        if in_a is None:
            add(f"- 新增文件 {name}（{in_b['rows']} 行）")
            changed = True
        elif in_b is None:
            add(f"- 文件 {name} 已移除")
            changed = True
        elif in_a["sha256"] != in_b["sha256"]:
            add(
                f"- {name} 内容已变化（行数 {in_a['rows']} → {in_b['rows']}，"
                f"哈希 {in_a['sha256'][:12]} → {in_b['sha256'][:12]}）"
            )
            changed = True
    if not changed:
        add("- 输入文件哈希一致：数据未发生变化")
    add("")


def _compare_configs(add, config_a: dict, config_b: dict) -> None:
    if config_a == config_b:
        return
    add("## 分析配置变化")
    add("")
    for key in sorted(set(config_a) | set(config_b)):
        va, vb = config_a.get(key), config_b.get(key)
        if va != vb:
            add(f"- {key}: {va!r} → {vb!r}")
    add("")
    add("配置不同会改变状态判定与估计口径，比较结果时请先确认配置差异是有意为之。")
    add("")


def _compare_units(add, results_a: dict, results_b: dict) -> list[str]:
    add("## 单位级状态变化")
    add("")
    units_a = results_a.get("units", {})
    units_b = results_b.get("units", {})
    changed: list[str] = []
    only_b = sorted(set(units_b) - set(units_a))
    only_a = sorted(set(units_a) - set(units_b))
    for ref in sorted(set(units_a) & set(units_b)):
        ua, ub = units_a[ref], units_b[ref]
        diffs = [
            f"{label}：{ua.get(field)} → {ub.get(field)}"
            for field, label in _UNIT_COMPARE_FIELDS
            if ua.get(field) != ub.get(field)
        ]
        if diffs:
            src = ub.get("source", {})
            where = f"{src.get('file')}:{src.get('line')}" if src else "出处未知"
            changed.append(ref)
            add(f"- {ref}（单位 {ub.get('unit_ref')}，{where}）：" + "；".join(diffs))
    if only_b:
        add(f"- 新增 {len(only_b)} 个分配记录：" + "、".join(only_b[:20]) + ("……" if len(only_b) > 20 else ""))
        changed.extend(only_b)
    if only_a:
        add(f"- 移除 {len(only_a)} 个分配记录：" + "、".join(only_a[:20]) + ("……" if len(only_a) > 20 else ""))
        changed.extend(only_a)
    if not changed:
        add("- 无单位级变化")
    add("")
    return changed


def _compare_estimates(add, results_a: dict, results_b: dict) -> None:
    add("## 效果估计变化")
    add("")
    seg_a = results_a.get("segments", {})
    seg_b = results_b.get("segments", {})
    any_row = False
    for segment in sorted(set(seg_a) | set(seg_b)):
        pa = (seg_a.get(segment) or {}).get("pooled_stratified")
        pb = (seg_b.get(segment) or {}).get("pooled_stratified")
        if pa is None and pb is None:
            continue
        any_row = True
        fmt = lambda p: (  # noqa: E731
            "无可比层" if p is None else f"{p['rd']:.4f} [{p['ci'][0]:.4f}, {p['ci'][1]:.4f}]"
        )
        add(f"- {segment} 分层合并率差：A {fmt(pa)} → B {fmt(pb)}")
    if not any_row:
        add("- 两次运行均无可合并的效应估计")
    add("")
