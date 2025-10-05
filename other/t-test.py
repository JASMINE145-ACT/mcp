"""
T-Test Statistical Analysis Module
Provides comprehensive t-test functions for data analysis
Compatible with agent.py tool system
"""
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Literal, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

Tail = Literal["two-sided", "less", "greater"]

@dataclass
class TTestResult:
    kind: str                       # one-sample / independent / paired
    t: float
    df: float
    p_value: float
    tail: Tail
    ci: Tuple[float, float]         # 均值差/均值 的置信区间
    effect_size: float              # Cohen's d（独立样本为 Hedges g）
    effect_label: str               # small/medium/large
    assumptions: Dict[str, Any]     # 正态性/方差齐性等
    sample_sizes: Tuple[int, ...]   # 各样本有效 n
    estimate: Dict[str, float]      # 样本均值/差、标准差等
    interpretation: str             # 自动中文解释模板

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "t": self.t,
            "df": self.df,
            "p_value": self.p_value,
            "tail": self.tail,
            "ci": self.ci,
            "effect_size": self.effect_size,
            "effect_label": self.effect_label,
            "assumptions": self.assumptions,
            "sample_sizes": self.sample_sizes,
            "estimate": self.estimate,
            "interpretation": self.interpretation,
        }

# ----------------------------- #
# 工具函数
# ----------------------------- #
def _cohens_d_one_sample(x: np.ndarray, mu0: float) -> float:
    return (np.nanmean(x) - mu0) / np.nanstd(x, ddof=1)

def _cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    return np.nanmean(d) / np.nanstd(d, ddof=1)

def _hedges_g_independent(x: np.ndarray, y: np.ndarray) -> float:
    # pooled SD (unbiased with Hedges' correction)
    nx, ny = len(x), len(y)
    sx2, sy2 = np.nanvar(x, ddof=1), np.nanvar(y, ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    d = (np.nanmean(x) - np.nanmean(y)) / s_pooled
    # Hedges' correction
    J = 1 - 3 / (4 * (nx + ny) - 9)
    return d * J

def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2: return "trivial"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"

def _t_ci(mean_diff: float, se: float, df: float, alpha: float) -> Tuple[float, float]:
    q = stats.t.ppf(1 - alpha / 2, df)
    return mean_diff - q * se, mean_diff + q * se

def _to_array(v: Iterable) -> np.ndarray:
    if isinstance(v, (pd.Series, pd.DataFrame)):
        v = v.squeeze()
    arr = np.asarray(list(v), dtype=float)
    return arr[~np.isnan(arr)]  # drop NaN


# ----------------------------- #
# 1) 单样本 t 检验
# ----------------------------- #
def ttest_one_sample(
    x: Iterable, 
    mu0: float = 0.0, 
    tail: Tail = "two-sided", 
    alpha: float = 0.05
) -> TTestResult:
    x = _to_array(x)
    n = len(x)
    m, s = np.nanmean(x), np.nanstd(x, ddof=1)
    se = s / np.sqrt(n)
    df = n - 1

    # 正态性（Shapiro 在 n>5000 时意义不大，这里做提示）
    sw_stat, sw_p = stats.shapiro(x) if 3 <= n <= 5000 else (np.nan, np.nan)

    # SciPy 双尾 t 与单尾转换
    t_stat, p_two = stats.ttest_1samp(x, popmean=mu0, alternative="two-sided")
    if tail == "two-sided":
        p = p_two
    elif tail == "greater":
        p = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:  # less
        p = p_two / 2 if t_stat < 0 else 1 - p_two / 2

    # CI（针对均值 - mu0 的差）
    ci = _t_ci(mean_diff=m - mu0, se=se, df=df, alpha=alpha)

    d = _cohens_d_one_sample(x, mu0)
    interpretation = (
        f"单样本 t 检验：t({df:.0f})={t_stat:.3f}, p={p:.4f}；"
        f"样本均值={m:.3f}，与假设均值 {mu0:.3f} 的差为 {m - mu0:.3f}，"
        f"{(1-alpha)*100:.0f}% CI [{ci[0]:.3f}, {ci[1]:.3f}]。"
        f"效应量 d={d:.3f}（{_effect_label(d)}）。"
    )

    return TTestResult(
        kind="one-sample",
        t=float(t_stat),
        df=float(df),
        p_value=float(p),
        tail=tail,
        ci=ci,
        effect_size=float(d),
        effect_label=_effect_label(d),
        assumptions={
            "normality_shapiro_W": float(sw_stat) if not np.isnan(sw_stat) else None,
            "normality_p": float(sw_p) if not np.isnan(sw_p) else None,
            "note": "Shapiro p>0.05 表示未拒绝正态；大样本时中心极限定理通常可放宽要求。",
        },
        sample_sizes=(n,),
        estimate={
            "mean": float(m),
            "std": float(s),
            "se": float(se),
            "mean_minus_mu0": float(m - mu0),
        },
        interpretation=interpretation,
    )


# ----------------------------- #
# 2) 独立样本 t 检验（可 Welch）
# ----------------------------- #
def ttest_independent(
    x: Iterable,
    y: Iterable,
    equal_var: bool = False,   # 默认 Welch（工业/互联网常用）
    tail: Tail = "two-sided",
    alpha: float = 0.05
) -> TTestResult:
    x, y = _to_array(x), _to_array(y)
    nx, ny = len(x), len(y)
    mx, my = np.nanmean(x), np.nanmean(y)
    sx, sy = np.nanstd(x, ddof=1), np.nanstd(y, ddof=1)

    # 方差齐性检验
    lev_stat, lev_p = stats.levene(x, y, center="median")

    # t 与 p
    alternative = "two-sided"
    t_stat, p_two = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
    if tail == "two-sided":
        p = p_two
    elif tail == "greater":
        p = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:
        p = p_two / 2 if t_stat < 0 else 1 - p_two / 2

    # 自由度与 SE（用于 CI）
    if equal_var:
        df = nx + ny - 2
        sp2 = ((nx - 1) * sx**2 + (ny - 1) * sy**2) / df
        se = np.sqrt(sp2 * (1 / nx + 1 / ny))
    else:  # Welch
        se = np.sqrt(sx**2 / nx + sy**2 / ny)
        df = (sx**2 / nx + sy**2 / ny) ** 2 / (
            (sx**2 / nx) ** 2 / (nx - 1) + (sy**2 / ny) ** 2 / (ny - 1)
        )

    mean_diff = mx - my
    ci = _t_ci(mean_diff=mean_diff, se=se, df=df, alpha=alpha)

    g = _hedges_g_independent(x, y)
    interpretation = (
        f"独立样本 t 检验（{'等方差' if equal_var else 'Welch'}）："
        f"t({df:.1f})={t_stat:.3f}, p={p:.4f}；"
        f"均值差={mean_diff:.3f}（组1 {mx:.3f} vs 组2 {my:.3f}），"
        f"{(1-alpha)*100:.0f}% CI [{ci[0]:.3f}, {ci[1]:.3f}]。"
        f"效应量 Hedges g={g:.3f}（{_effect_label(g)}）。"
    )

    return TTestResult(
        kind="independent",
        t=float(t_stat),
        df=float(df),
        p_value=float(p),
        tail=tail,
        ci=ci,
        effect_size=float(g),
        effect_label=_effect_label(g),
        assumptions={
            "equal_variance_levene_stat": float(lev_stat),
            "equal_variance_p": float(lev_p),
            "note": "Levene p>0.05 表示未拒绝方差齐性。默认 equal_var=False 使用 Welch 更稳健。",
        },
        sample_sizes=(nx, ny),
        estimate={
            "mean_group1": float(mx),
            "std_group1": float(sx),
            "mean_group2": float(my),
            "std_group2": float(sy),
            "mean_diff": float(mean_diff),
            "se_diff": float(se),
        },
        interpretation=interpretation,
    )


# ----------------------------- #
# 3) 配对样本 t 检验
# ----------------------------- #
def ttest_paired(
    x: Iterable, 
    y: Iterable, 
    tail: Tail = "two-sided", 
    alpha: float = 0.05
) -> TTestResult:
    x, y = _to_array(x), _to_array(y)
    # 对齐长度（配对：逐对删除缺失）
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    d = x - y
    n_eff = np.sum(~np.isnan(d))
    d = d[~np.isnan(d)]

    md, sd = np.nanmean(d), np.nanstd(d, ddof=1)
    se = sd / np.sqrt(n_eff)
    df = n_eff - 1

    # 正态性（对差值 d 做 Shapiro）
    sw_stat, sw_p = stats.shapiro(d) if 3 <= n_eff <= 5000 else (np.nan, np.nan)

    t_stat, p_two = stats.ttest_rel(x, y, alternative="two-sided")
    if tail == "two-sided":
        p = p_two
    elif tail == "greater":  # H1: x - y > 0
        p = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:                    # H1: x - y < 0
        p = p_two / 2 if t_stat < 0 else 1 - p_two / 2

    ci = _t_ci(mean_diff=md, se=se, df=df, alpha=alpha)
    d_eff = _cohens_d_paired(x, y)

    interpretation = (
        f"配对样本 t 检验：t({df:.0f})={t_stat:.3f}, p={p:.4f}；"
        f"配对均值差={md:.3f}，{(1-alpha)*100:.0f}% CI [{ci[0]:.3f}, {ci[1]:.3f}]。"
        f"效应量 d={d_eff:.3f}（{_effect_label(d_eff)}）。"
    )

    return TTestResult(
        kind="paired",
        t=float(t_stat),
        df=float(df),
        p_value=float(p),
        tail=tail,
        ci=ci,
        effect_size=float(d_eff),
        effect_label=_effect_label(d_eff),
        assumptions={
            "normality_of_difference_shapiro_W": float(sw_stat) if not np.isnan(sw_stat) else None,
            "normality_p": float(sw_p) if not np.isnan(sw_p) else None,
            "note": "对差值 d 的正态性检验；n 大时可依赖中心极限定理。",
        },
        sample_sizes=(int(n_eff),),
        estimate={
            "mean_diff": float(md),
            "std_diff": float(sd),
            "se_diff": float(se),
        },
        interpretation=interpretation,
    )


# ----------------------------- #
# 使用示例（作为 RAG 提示片段）
# ----------------------------- #
if __name__ == "__main__":
    # 构造示例数据
    rng = np.random.default_rng(42)
    a = rng.normal(loc=10, scale=2, size=30)
    b = rng.normal(loc=9.3, scale=2.1, size=28)
    pre = rng.normal(70, 8, 25)
    post = pre + rng.normal(3, 5, 25)

    r1 = ttest_one_sample(a, mu0=9.5)
    r2 = ttest_independent(a, b, equal_var=False)
    r3 = ttest_paired(pre, post)

    for r in [r1, r2, r3]:
        print(r.kind, r.to_dict()["interpretation"])
