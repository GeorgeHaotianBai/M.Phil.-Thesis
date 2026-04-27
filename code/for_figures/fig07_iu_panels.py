#!/usr/bin/env python3
"""Inverted-U security-premium 4-panel robustness.

Replaces the prior single-panel ``security_premium_empirical.jpg`` with
four panels that address referee concerns:

  (a) Main spec: 3-bin coefficient plot, residualised log(1 + violent
      events) by right-wing winner-share bin.  Same as the existing main.
  (b) Finer 5-bin spec: validates the result is not bin-cutoff sensitive.
  (c) Raw-events spec: outcome is raw violent_events count instead of
      log, validating the result is not a log-transform artefact.
  (d) Pre-2014 subsample: validates the result is not Modi-era-driven.

Construction follows the upstream Stata file
``vote_35_65_current_exact.do`` and the existing
``code/for_figures/fig07_security_premium_empirical.py`` for the base
specification.

Output:
  output/figures/iu_main.jpg
  output/figures/iu_5bin.jpg
  output/figures/iu_raw_events.jpg
  output/figures/iu_pre2014.jpg
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_ROOT = Path("/Users/admin/Dropbox/Oxford/M.Phil. Thesis")
STYLE_DIR = THESIS_ROOT / "code" / "for_figures"
sys.path.insert(0, str(STYLE_DIR))
from _thesis_style import apply_thesis_style, THESIS_COLORS, save_jpg  # noqa: E402

apply_thesis_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PANEL_CSV = Path(
    "/Users/admin/Dropbox/PE of Cattle _ Shared/india beef and religion/"
    "data/regression/gdelt_with_control_and_trend.csv"
)
VOTE_DTA = Path(
    "/Users/admin/Dropbox/Oxford/M.Phil. Thesis/empirical_writing/PE_of_India/"
    "replication_package_India_slightly_different/data/figures/vote/vote_clean_state.dta"
)
OUT_DIR = THESIS_ROOT / "output" / "figures"

# Bin sets for the four panels.
BINS_3 = [(0.35, 0.45, "35-45%"),
          (0.45, 0.55, "45-55%"),
          (0.55, 0.65, "55-65%")]
BINS_5 = [(0.30, 0.40, "30-40%"),
          (0.40, 0.50, "40-50%"),
          (0.50, 0.60, "50-60%"),
          (0.60, 0.70, "60-70%"),
          (0.70, 0.80, "70-80%")]


# ---------------------------------------------------------------------------
# 1. Load panel + vote data
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_CSV, low_memory=False)
    df["lnevent_all"] = np.log1p(df["violent_events"].astype(float).fillna(0.0))
    df["raw_events"] = df["violent_events"].astype(float).fillna(0.0)
    return df


def load_vote() -> pd.DataFrame:
    vote, _ = pyreadstat.read_dta(str(VOTE_DTA))
    vote["year"] = vote["year"].astype(int)
    return vote[["clean_state", "year", "right_wing_winner_prop"]]


def merge_with_lastwinner(panel: pd.DataFrame, vote: pd.DataFrame) -> pd.DataFrame:
    base = panel.merge(vote, on=["clean_state", "year"], how="left")
    base = base.sort_values(["clean_state", "year"])
    base["right_wing_winner_prop"] = (
        base.groupby("clean_state")["right_wing_winner_prop"].ffill()
    )
    return base


# ---------------------------------------------------------------------------
# 2. Residualisation (mirrors fig07_security_premium_empirical.py)
# ---------------------------------------------------------------------------
WEATHER = ["precip_mean", "temp_mean", "precip_std", "temp_std",
           "precip_mean_sq", "temp_mean_sq"]


def make_within_year_quartile(df: pd.DataFrame, col: str, out: str) -> pd.DataFrame:
    s = df.groupby("year")[col].transform(
        lambda x: pd.qcut(x.rank(method="first"), q=4, labels=False, duplicates="drop") + 1
    )
    df[out] = s
    return df


def residualise(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    work = df.copy()
    for src, q in [
        ("muslim_share_quartile", None),
        ("log_population_quartile", None),
        ("gdp_share", "gdp_share_quartile"),
        ("literacy_rate", "literacy_rate_quartile"),
    ]:
        if q is None:
            continue
        if src in work.columns and q not in work.columns:
            work = make_within_year_quartile(work, src, q)

    fe_quartiles = [c for c in [
        "muslim_share_quartile", "log_population_quartile",
        "gdp_share_quartile", "literacy_rate_quartile",
    ] if c in work.columns]

    keep = [ycol, "clean_state", "year", "right_wing_winner_prop"] + WEATHER + fe_quartiles
    keep = [c for c in keep if c in work.columns]
    work = work[keep].dropna(subset=[ycol, "clean_state", "year"] + WEATHER)

    parts = [work[WEATHER].astype(float)]
    parts.append(pd.get_dummies(work["clean_state"], prefix="state",
                                drop_first=True).astype(float))
    parts.append(pd.get_dummies(work["year"].astype(int), prefix="year",
                                drop_first=True).astype(float))
    for q in fe_quartiles:
        inter = work[q].astype("Int64").astype(str) + "x" + work["year"].astype(int).astype(str)
        parts.append(pd.get_dummies(inter, prefix=q + "_y",
                                    drop_first=True).astype(float))

    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X, has_constant="add")
    X = X.loc[:, (X.abs().sum(axis=0) > 0)]
    y = work[ycol].astype(float)
    m = sm.OLS(y, X).fit()
    work["residual"] = m.resid
    return work[["clean_state", "year", "right_wing_winner_prop", "residual"]]


# ---------------------------------------------------------------------------
# 3. Bin and bootstrap
# ---------------------------------------------------------------------------
def assign_bin(prop: float, bins) -> int | None:
    if pd.isna(prop):
        return None
    for i, (lo, hi, _) in enumerate(bins, start=1):
        if lo < prop <= hi:
            return i
    return None


def cluster_bootstrap_mean(values: pd.Series, clusters: pd.Series,
                            B: int = 2000, rng=None) -> tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng(20260422)
    df = pd.DataFrame({"v": values.values, "c": clusters.values}).dropna()
    if df.empty:
        return float("nan"), float("nan"), float("nan")
    states = df["c"].unique()
    means = []
    boot_index = {s: df[df["c"] == s]["v"].values for s in states}
    for _ in range(B):
        sample = rng.choice(states, size=len(states), replace=True)
        vals = np.concatenate([boot_index[s] for s in sample])
        means.append(vals.mean())
    means = np.asarray(means)
    return df["v"].mean(), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# 4. Plot one panel (no in-image title)
# ---------------------------------------------------------------------------
def plot_panel(points: list[dict], outpath: Path, ylabel: str) -> None:
    xs = np.arange(len(points))
    means = np.array([p["mean"] for p in points])
    lows = np.array([p["lo"] for p in points])
    highs = np.array([p["hi"] for p in points])
    labels = [p["label"] for p in points]
    ns = [p["n"] for p in points]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.axhline(0, color=THESIS_COLORS["grey"], linewidth=0.7,
               linestyle=(0, (1, 2)), alpha=0.7)
    ax.errorbar(
        xs, means,
        yerr=[means - lows, highs - means],
        fmt="o", color=THESIS_COLORS["blue"],
        ecolor=THESIS_COLORS["red"], elinewidth=1.4, capsize=4,
        markersize=7, markerfacecolor=THESIS_COLORS["blue"],
        markeredgecolor=THESIS_COLORS["blue"], zorder=3,
    )
    for i, n in enumerate(ns):
        ax.annotate(f"N = {n}", xy=(xs[i], means[i]), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.0, color=THESIS_COLORS["grey"])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Right-wing winner vote share")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(points) - 0.5)
    span = highs.max() - lows.min() if highs.size else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    ax.set_ylim(lows.min() - 0.10 * span, highs.max() + 0.10 * span)
    fig.tight_layout()
    save_jpg(fig, outpath)
    plt.close(fig)


def run_one(merged: pd.DataFrame, ycol: str, bins, year_filter,
            outpath: Path, ylabel: str) -> None:
    """Residualise, bin, bootstrap, and plot one panel."""
    if year_filter is not None:
        df = merged[merged["year"].astype(int) < year_filter].copy()
    else:
        df = merged.copy()
    res = residualise(df, ycol)
    res["bin"] = res["right_wing_winner_prop"].apply(lambda p: assign_bin(p, bins))
    res = res.dropna(subset=["bin"])
    if res.empty:
        # Empty panel placeholder.
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        ax.text(0.5, 0.5, "no observations in window", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xlabel("Right-wing winner vote share"); ax.set_ylabel(ylabel)
        fig.tight_layout(); save_jpg(fig, outpath); plt.close(fig)
        return
    res["bin"] = res["bin"].astype(int)

    points = []
    for i, (lo, hi, label) in enumerate(bins, start=1):
        sub = res[res["bin"] == i]
        if len(sub) == 0:
            continue
        m, clo, chi = cluster_bootstrap_mean(sub["residual"], sub["clean_state"])
        points.append({"label": label, "mean": m, "lo": clo, "hi": chi, "n": len(sub)})
    plot_panel(points, outpath, ylabel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    panel = load_panel()
    vote = load_vote()
    merged = merge_with_lastwinner(panel, vote)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # (a) main: 3 bins, log outcome, full sample
    out = OUT_DIR / "iu_main.jpg"
    run_one(merged, "lnevent_all", BINS_3, year_filter=None,
            outpath=out, ylabel="Residual log(1 + violent events)")
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"  iu_main.jpg  sha256_16={h}")

    # (b) 5-bin
    out = OUT_DIR / "iu_5bin.jpg"
    run_one(merged, "lnevent_all", BINS_5, year_filter=None,
            outpath=out, ylabel="Residual log(1 + violent events)")
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"  iu_5bin.jpg  sha256_16={h}")

    # (c) raw-events outcome
    out = OUT_DIR / "iu_raw_events.jpg"
    run_one(merged, "raw_events", BINS_3, year_filter=None,
            outpath=out, ylabel="Residual violent-event count")
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"  iu_raw_events.jpg  sha256_16={h}")

    # (d) pre-2014 subsample
    out = OUT_DIR / "iu_pre2014.jpg"
    run_one(merged, "lnevent_all", BINS_3, year_filter=2014,
            outpath=out, ylabel="Residual log(1 + violent events)")
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"  iu_pre2014.jpg  sha256_16={h}")


if __name__ == "__main__":
    main()
