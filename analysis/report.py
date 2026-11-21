"""把结构化结果渲染为 Markdown 报告（中文）。

报告只消费 analyze() 产出的字典，不重新计算；同一 results.json
重新渲染应得到同一份报告。
"""
from __future__ import annotations

from .model import OutcomeState

STATE_LABELS = {
    OutcomeState.MATURE_EVENT.value: "成熟事件",
    OutcomeState.NON_EVENT.value: "最终非事件",
    OutcomeState.PENDING_REVIEW.value: "待复核/未成熟",
    OutcomeState.WINDOW_OPEN.value: "窗口未闭合",
    OutcomeState.MISSING.value: "缺结果行",
    OutcomeState.AFTER_CUTOFF.value: "截止日后",
}


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _ci(bounds: list[float] | None) -> str:
    if not bounds:
        return "—"
    return f"[{_num(bounds[0])}, {_num(bounds[1])}]"


def render_report(results: dict, run_id: str, manifest: dict, generated_at: str) -> str:
    lines: list[str] = []
    add = lines.append
    config = results["config"]

    add("# 内容治理措施效果分析报告")
    add("")
    add(f"- 运行 ID：`{run_id}`")
    add(f"- 生成时间（UTC）：{generated_at}")
    add(f"- 分析截止日：{results['cutoff']}")
    add(f"- 结果成熟窗口：{config['maturity_window_days']} 天")
    add(
        "- 主要终点："
        + "/".join(config["event_kinds"])
        + " 且成熟度为 "
        + "/".join(config["event_maturity"])
        + " 且复核状态为 "
        + "/".join(config["final_review"])
    )
    add("")
    add("## 输入清单")
    add("")
    add("| 文件 | 行数 | SHA-256（前 12 位） |")
    add("| --- | ---: | --- |")
    for name, info in sorted(manifest["files"].items()):
        add(f"| {name} | {info['rows']} | `{info['sha256'][:12]}` |")
    add("")
    add("输入文件内容的任何更正都会改变哈希值；比较两次运行即可定位变化来源。")
    add("")

    _section_states(add, results)
    _section_scope(add, results)
    _section_effects(add, results)
    _section_exposure(add, results)
    _section_anomalies(add, results)
    _section_limitations(add, results)
    _section_reproduce(add, run_id)
    return "\n".join(lines) + "\n"


def _section_states(add, results: dict) -> None:
    add("## 1. 结果状态核算")
    add("")
    add(
        "未曝光、无投诉记录、结果尚未成熟与缺少结果行是不同状态，分别核算；"
        "主要率估计只使用“成熟事件”与“最终非事件”两类单位。"
    )
    add("")
    header = "| 策略版本段 | 组别 | 分配 | 曝光 | " + " | ".join(STATE_LABELS.values()) + " |"
    add(header)
    add("| --- | --- | ---: | ---: | " + " | ".join("---:" for _ in STATE_LABELS) + " |")
    for segment, info in sorted(results["segments"].items()):
        arms_units: dict[str, dict[str, int]] = {}
        for unit in results["units"].values():
            if unit["version_segment"] != segment:
                continue
            bucket = arms_units.setdefault(
                unit["group"], {"n": 0, "exposed": 0, **{k: 0 for k in STATE_LABELS}}
            )
            bucket["n"] += 1
            bucket["exposed"] += 1 if unit["exposure_state"] == "exposed" else 0
            bucket[unit["outcome_state"]] += 1
        for group in info["arms"]:
            counts = arms_units.get(group)
            if counts is None:
                continue
            cells = " | ".join(str(counts[key]) for key in STATE_LABELS)
            add(
                f"| {segment} | {group} | {counts['n']} | {counts['exposed']} | {cells} |"
            )
    add("")


def _section_scope(add, results: dict) -> None:
    add("## 2. 可比范围与适用人群")
    add("")
    for segment, info in sorted(results["segments"].items()):
        action = info.get("action") or "（无版本说明）"
        add(f"### {segment}（{action}）")
        add("")
        add("| 层 | " + " | ".join(info["arms"]) + " | 可比较 |")
        add("| --- | " + " | ".join("---:" for _ in info["arms"]) + " | --- |")
        strata = set(info["strata"]) | set(info["non_comparable_strata"])
        counts: dict[tuple[str, str], int] = {}
        for unit in results["units"].values():
            if unit["version_segment"] == segment:
                key = (unit["stratum"], unit["group"])
                counts[key] = counts.get(key, 0) + 1
        for stratum in sorted(strata):
            cells = " | ".join(str(counts.get((stratum, arm), 0)) for arm in info["arms"])
            mark = "是" if stratum in info["strata"] else "否（层内缺少对照）"
            add(f"| {stratum} | {cells} | {mark} |")
        add("")
        covered = "、".join(sorted(info["strata"])) or "无"
        add(f"该段估计仅适用于层：{covered}。")
        add("")


def _section_effects(add, results: dict) -> None:
    add("## 3. 效果估计（按分配组别，ITT）")
    add("")
    add(
        "率差 = 处理组成熟事件率 − 对照组成熟事件率（负值表示处理组事件更少）；"
        "区间为 95% Newcombe 混合区间。Manski 界限给出未决结果在任意填补规则下"
        "率差的可能范围。"
    )
    add("")
    for segment, info in sorted(results["segments"].items()):
        add(f"### {segment}")
        add("")
        if not info["strata"]:
            add("该段没有可比较的层，无法给出因果对比。")
            add("")
            continue
        add("| 层 | 处理组事件/可评估 | 对照组事件/可评估 | 率差 | 95% 区间 | Manski 界限 |")
        add("| --- | ---: | ---: | ---: | --- | --- |")
        for stratum, effect in sorted(info["strata"].items()):
            idx, ref = effect["index"], effect["reference"]
            add(
                f"| {stratum} | {idx['n_events']}/{idx['n_evaluable']} "
                f"| {ref['n_events']}/{ref['n_evaluable']} "
                f"| {_num(effect.get('rd'))} | {_ci(effect.get('newcombe_ci'))} "
                f"| {_ci(effect.get('manski_rd_bounds'))} |"
            )
        add("")
        pooled = info.get("pooled_stratified")
        if pooled:
            add(
                f"分层加权合并率差：**{_num(pooled['rd'])}**，"
                f"95% 区间 {_ci(pooled['ci'])}。"
            )
            if pooled.get("heterogeneity_note"):
                add(f"注意：{pooled['heterogeneity_note']}。")
            add("")
        overall = info.get("overall_unadjusted")
        if overall and overall.get("cluster_robust"):
            cr = overall["cluster_robust"]
            add(
                f"按场次聚类的稳健敏感性：率差标准误 {_num(cr['se'])}"
                f"（{cr['n_clusters']} 个聚类），95% 区间 {_ci(cr['ci'])}。"
            )
            if overall.get("cluster_note"):
                add(f"{overall['cluster_note']}。")
            add("")


def _section_exposure(add, results: dict) -> None:
    add("## 4. 曝光机会与剂量")
    add("")
    add(
        "曝光机会本身可能受措施影响（如推荐降权）。下表为各臂曝光率；"
        "曝光率是分配后的中间变量，其组间差异描述机制，不作为效果估计。"
    )
    add("")
    add("| 策略版本段 | 组别 | 曝光率 | 95% 区间 | 跨策略曝光单位数 | 事件延迟（天，中位/范围） |")
    add("| --- | --- | ---: | --- | ---: | --- |")
    for segment, info in sorted(results["segments"].items()):
        for stratum_effect in [info.get("overall_unadjusted")]:
            if not stratum_effect:
                continue
            index_arm = next(a for a in info["arms"] if a != info["reference_arm"])
            for key, label in (("index", index_arm), ("reference", info["reference_arm"])):
                arm = stratum_effect[key]
                delay = arm["event_delay_days"]
                delay_text = (
                    f"{_num(delay['median'], 1)}（{delay['min']}~{delay['max']}）"
                    if delay
                    else "—"
                )
                add(
                    f"| {segment} | {label} | {_pct(arm['exposure_rate'])} "
                    f"| {_ci(arm['exposure_rate_ci'])} | {arm['n_cross_policy_exposed']} "
                    f"| {delay_text} |"
                )
    add("")
    add("注：上表按可比较层合并；逐层曝光率见 results.json。")
    add("")


def _section_anomalies(add, results: dict) -> None:
    add("## 5. 记录关系与异常")
    add("")
    sessions = results["sessions"]
    spanning = [s for s in sessions.values() if s["spans_version_switch"]]
    mixed = [s for s in sessions.values() if s["spans_policy_switch"]]
    add(
        f"- 跨策略版本切换的场次：{len(spanning)} 个"
        + ("（" + "、".join(sorted(s['session_ref'] for s in spanning)) + "）" if spanning else "")
    )
    add(
        f"- 同一场次出现多种策略的场次：{len(mixed)} 个；"
        "同场次用户的结果可能相关，已按场次聚类评估不确定度"
    )
    anomalies = results["anomalies"]
    add(f"- 记录级异常：{len(anomalies)} 条")
    if anomalies:
        add("")
        for item in anomalies:
            add(f"  - {item}")
    add("")


def _section_limitations(add, results: dict) -> None:
    add("## 6. 因果解释限制")
    add("")
    for item in results["limitations"]:
        add(f"- {item}")
    add("")


def _section_reproduce(add, run_id: str) -> None:
    add("## 7. 复算与更新")
    add("")
    add("```bash")
    add("# 用同一批数据复算（结果应一致）")
    add("python3 -m analysis run --data data --out runs")
    add("")
    add("# 新一批观测或更正后的资料：替换/更新 data/ 后重新运行，再与本次运行比较")
    add(f"python3 -m analysis compare runs/<旧运行ID> runs/{run_id}")
    add("```")
    add("")
    add("每次运行保存 manifest.json（输入哈希）、config.json（分析配置）、")
    add("results.json（结构化结果）与本报告；原始分配或投诉分类被更正后，")
    add("新旧运行的差异可由 compare 子命令逐项解释。")
    add("")
