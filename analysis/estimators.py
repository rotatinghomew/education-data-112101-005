"""区间与不确定度估计（仅标准库）。

- Wilson 得分区间：单臂比例。
- Newcombe 混合区间（方法 10）：两臂比例差。
- Manski 式界限：把待定/缺失结果全部视为事件或全部视为非事件，
  给出在任意填补规则下都成立的取值范围。
- 按场次聚类的三明治标准误（CR1 小样本修正）：同一场次内不同用户
  接受不同策略、同一用户跨场次出现，记录不独立，不能按独立样本计算。
"""
from __future__ import annotations

import math
from collections import defaultdict


def wilson_interval(k: int, n: int, z: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    # 边界情形取精确值，避免浮点误差使区间越过 0/1。
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (lo, hi)


def newcombe_rd_interval(
    k1: int, n1: int, k0: int, n0: int, z: float
) -> tuple[float, float, float]:
    """返回（比例差， 下界， 上界），差 = 处理组比例 - 对照组比例。"""
    p1 = k1 / n1
    p0 = k0 / n0
    l1, u1 = wilson_interval(k1, n1, z)
    l0, u0 = wilson_interval(k0, n0, z)
    rd = p1 - p0
    lo = rd - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = rd + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return (rd, max(-1.0, lo), min(1.0, hi))


def manski_bounds(n_event: int, n_non_event: int, n_unresolved: int) -> tuple[float, float]:
    """在任意填补规则下事件率的取值范围（分母为全部分配单位）。"""
    n = n_event + n_non_event + n_unresolved
    if n <= 0:
        return (0.0, 1.0)
    return (n_event / n, (n_event + n_unresolved) / n)


def cluster_robust_se(samples: list[tuple[str, int, float]]) -> tuple[float, int] | None:
    """对 y ~ 1 + t 的 OLS 估计计算按 cluster 聚类的 CR1 标准误。

    samples: (cluster_key, t, y)，t ∈ {0, 1}。
    返回（处理系数的标准误， 聚类数）；无法计算时返回 None。
    """
    n = len(samples)
    n1 = sum(t for _, t, _ in samples)
    n0 = n - n1
    if n < 3 or n1 == 0 or n0 == 0:
        return None
    y1 = sum(y for _, t, y in samples if t == 1) / n1
    y0 = sum(y for _, t, y in samples if t == 0) / n0

    clusters: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for key, t, y in samples:
        e = y - (y0 + (y1 - y0) * t)
        clusters[key].append((t, e))
    g = len(clusters)
    if g < 2:
        return None

    # meat = Σ_g u_g u_gᵀ，u_g = Σ_{i∈g} x_i e_i，x_i = (1, t_i)
    m00 = m01 = m11 = 0.0
    for rows in clusters.values():
        u0 = sum(e for _, e in rows)
        u1 = sum(t * e for t, e in rows)
        m00 += u0 * u0
        m01 += u0 * u1
        m11 += u1 * u1

    # (XᵀX)⁻¹，XᵀX = [[n, n1], [n1, n1]]
    det = n * n1 - n1 * n1
    i00, i01 = n1 / det, -n1 / det
    i11 = n / det

    # V = (XᵀX)⁻¹ · meat · (XᵀX)⁻¹，取处理系数（第 2 个）的方差。
    v11 = (
        i01 * (m00 * i01 + m01 * i11)
        + i11 * (m01 * i01 + m11 * i11)
    )
    if v11 < 0.0:
        # 数值误差导致的微小负值按 0 处理，明显负值说明输入异常。
        if v11 < -1e-9:
            return None
        v11 = 0.0
    cr1 = (g / (g - 1.0)) * ((n - 1.0) / (n - 2.0))
    return (math.sqrt(v11 * cr1), g)


def pooled_effect(
    strata: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """按方差倒数加权合并若干层的（效应， 方差），返回（合并效应， 合并方差）。"""
    weights = [(1.0 / var, rd) for rd, var in strata if var > 0.0]
    if not weights:
        return None
    total_w = sum(w for w, _ in weights)
    pooled = sum(w * rd for w, rd in weights) / total_w
    return (pooled, 1.0 / total_w)
