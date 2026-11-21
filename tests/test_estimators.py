import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import Z_95
from analysis.estimators import (
    cluster_robust_se,
    manski_bounds,
    newcombe_rd_interval,
    pooled_effect,
    wilson_interval,
)


class WilsonIntervalTests(unittest.TestCase):
    def test_zero_events_lower_bound_is_zero(self):
        lo, hi = wilson_interval(0, 10, Z_95)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 1.0)

    def test_all_events_upper_bound_is_one(self):
        lo, hi = wilson_interval(10, 10, Z_95)
        self.assertEqual(hi, 1.0)
        self.assertGreater(lo, 0.0)

    def test_interval_contains_point_estimate_and_widens_when_smaller(self):
        lo, hi = wilson_interval(3, 10, Z_95)
        self.assertLess(lo, 0.3)
        self.assertGreater(hi, 0.3)
        lo_big, hi_big = wilson_interval(30, 100, Z_95)
        self.assertLess(hi_big - lo_big, hi - lo)

    def test_empty_denominator_returns_unit_interval(self):
        self.assertEqual(wilson_interval(0, 0, Z_95), (0.0, 1.0))


class NewcombeTests(unittest.TestCase):
    def test_rd_and_interval_order(self):
        rd, lo, hi = newcombe_rd_interval(4, 10, 2, 10, Z_95)
        self.assertAlmostEqual(rd, 0.2)
        self.assertLess(lo, rd)
        self.assertGreater(hi, rd)

    def test_interval_within_minus_one_one(self):
        rd, lo, hi = newcombe_rd_interval(10, 10, 0, 10, Z_95)
        self.assertAlmostEqual(rd, 1.0)
        self.assertGreaterEqual(lo, -1.0)
        self.assertLessEqual(hi, 1.0)


class ManskiTests(unittest.TestCase):
    def test_bounds_and_width(self):
        lo, hi = manski_bounds(2, 3, 5)
        self.assertAlmostEqual(lo, 0.2)
        self.assertAlmostEqual(hi, 0.7)
        self.assertAlmostEqual(hi - lo, 0.5)  # 宽度 = 未决比例

    def test_no_unresolved_is_degenerate(self):
        lo, hi = manski_bounds(2, 3, 0)
        self.assertAlmostEqual(lo, hi)


class ClusterRobustTests(unittest.TestCase):
    def test_hand_computed_two_cluster_case(self):
        # 手算：n=4, n1=2, 残差 ±0.5，meat[1][1]=0.5，(X'X)^-1[1][1]=1，
        # V=0.5，CR1 = 2/1 * 3/2 = 3 → se = sqrt(1.5)。
        samples = [("A", 1, 1.0), ("A", 0, 0.0), ("B", 1, 0.0), ("B", 0, 1.0)]
        result = cluster_robust_se(samples)
        self.assertIsNotNone(result)
        se, n_clusters = result
        self.assertEqual(n_clusters, 2)
        self.assertAlmostEqual(se, math.sqrt(1.5))

    def test_perfect_within_cluster_correlation_inflates_se(self):
        # 同一聚类内结果完全相同的复制样本：聚类数不变、观测数翻倍，
        # 标准误应明显大于把每条记录当独立聚类。
        base = [("A", 1, 1.0), ("A", 0, 0.0), ("B", 1, 0.0), ("B", 0, 1.0)]
        duplicated = [(key, t, y) for key, t, y in base for _ in range(2)]
        as_singletons = [(f"{key}-{i}", t, y) for i, (key, t, y) in enumerate(duplicated)]
        se_clustered, _ = cluster_robust_se(duplicated)
        se_singleton, _ = cluster_robust_se(as_singletons)
        self.assertGreater(se_clustered, se_singleton)

    def test_returns_none_when_not_computable(self):
        self.assertIsNone(cluster_robust_se([("A", 1, 1.0), ("A", 1, 0.0)]))  # 单聚类
        self.assertIsNone(
            cluster_robust_se([("A", 1, 1.0), ("B", 1, 0.0), ("C", 1, 1.0)])
        )  # 没有对照


class PooledEffectTests(unittest.TestCase):
    def test_equal_variance_weights_average(self):
        pooled = pooled_effect([(0.1, 0.04), (0.3, 0.04)])
        self.assertIsNotNone(pooled)
        estimate, var = pooled
        self.assertAlmostEqual(estimate, 0.2)
        self.assertAlmostEqual(var, 0.02)

    def test_more_precise_stratum_dominates(self):
        estimate, _ = pooled_effect([(0.1, 0.01), (0.9, 0.25)])
        self.assertLess(estimate, 0.5)

    def test_empty_returns_none(self):
        self.assertIsNone(pooled_effect([]))


if __name__ == "__main__":
    unittest.main()
