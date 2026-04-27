#!/usr/bin/env python3
"""Paradox-of-competence 4-panel robustness.

Replaces the prior single-panel ``paradox_of_competence.jpg`` with four
panels that address referee concerns about (i) GDELT-specificity, (ii)
the Modi era, (iii) and (b) being separately identifiable.  Each panel is
a 5-bin binned scatter of BJP retention probability against tenure-
average violence, with bootstrap 95% CI bars and a clustered LPM fit
overlay.  Specs:

  (a) Main spec: tenure-average log(1 + violent Hindu-Muslim events) using
      GDELT primary + Varshney-Wilkinson pre-1995 fallback.  Same as the
      original ``paradox_of_competence.jpg`` but re-rendered without an
      in-image title.
  (b) VW-only: Varshney-Wilkinson incidents instead of GDELT, validating
      that the result is not GDELT-specific.
  (c) Pre-2014 subsample: validates the result is not a Modi-era artefact.
  (d) Post-2014 subsample: presents the post-2014 sub-pattern.

Output:
  output/figures/paradox_main.jpg
  output/figures/paradox_vw.jpg
  output/figures/paradox_pre2014.jpg
  output/figures/paradox_post2014.jpg
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_ROOT = Path("/Users/admin/Dropbox/Oxford/M.Phil. Thesis")
STYLE_DIR = THESIS_ROOT / "code" / "for_figures"
sys.path.insert(0, str(STYLE_DIR))
from _thesis_style import apply_thesis_style, save_jpg  # noqa: E402

apply_thesis_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = THESIS_ROOT / "data" / "for_analysis"
ELEC_CSV = DATA_DIR / "fig04_state_year_elections.csv"
CONF_CSV = DATA_DIR / "fig04_conflict_state_year.csv"
VW_CSV = DATA_DIR / "fig04_vw_state_year.csv"

OUT_DIR = THESIS_ROOT / "output" / "figures"
RNG = np.random.default_rng(20260422)


# ---------------------------------------------------------------------------
# Data loaders -- adapted from fig04_paradox_of_competence.py
# ---------------------------------------------------------------------------
def _norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    return s.replace({"kashmir": "jammu and kashmir", "orissa": "odisha"})


def load_state_year_panel(use_vw_only: bool = False) -> pd.DataFrame:
    """Build the state-year panel of conflict events.

    Row: one (state, year). Unique level: (state, year).
    Conflict series: GDELT Hindu-Muslim violent events as primary,
    Varshney-Wilkinson incidents as pre-1995 fallback (default), or
    VW alone (when ``use_vw_only`` is True).
    """
    elec = pd.read_csv(ELEC_CSV, comment="#")
    conf = pd.read_csv(CONF_CSV)
    vw = pd.read_csv(VW_CSV)

    elec["key"] = _norm(elec["st_name"])
    conf["key"] = _norm(conf["clean_state"])
    vw["key"] = _norm(vw["clean_state"])

    sy = elec[["st_name", "key", "year", "ruling_party", "regime_class",
               "bjp_vote_share", "bjp_seat_share", "margin_pp",
               "election_year"]].copy()
    sy = sy.merge(
        conf[["key", "year", "violent_hindu_muslim_events"]]
            .rename(columns={"violent_hindu_muslim_events": "gdelt_hm"}),
        on=["key", "year"], how="left",
    )
    sy = sy.merge(vw[["key", "year", "vw_incidents"]], on=["key", "year"],
                  how="left")

    if use_vw_only:
        sy["conflict_events"] = sy["vw_incidents"]
    else:
        sy["conflict_events"] = sy["gdelt_hm"]
        pre95 = (sy["year"] <= 1995) & (sy["gdelt_hm"].isna())
        sy.loc[pre95, "conflict_events"] = sy.loc[pre95, "vw_incidents"]
    sy["conflict_events"] = sy["conflict_events"].fillna(0.0)
    sy["log_conflict"] = np.log1p(sy["conflict_events"])
    return sy.sort_values(["st_name", "year"]).reset_index(drop=True)


def build_defense_panel(sy: pd.DataFrame) -> pd.DataFrame:
    """One row per BJP defense election. Sample: every (state, election_year)
    where BJP held the state in election_year - 1."""
    ep = (sy[["st_name", "key", "election_year", "ruling_party",
              "bjp_vote_share", "bjp_seat_share", "regime_class"]]
          .drop_duplicates(["st_name", "election_year"])
          .sort_values(["st_name", "election_year"])
          .reset_index(drop=True))
    ep["bjp_vote_prev"] = ep.groupby("st_name")["bjp_vote_share"].shift(1)
    ep["bjp_seat_prev"] = ep.groupby("st_name")["bjp_seat_share"].shift(1)
    ep["ruling_prev"] = ep.groupby("st_name")["ruling_party"].shift(1)
    ep["election_year_prev"] = ep.groupby("st_name")["election_year"].shift(1)

    def _window(row):
        s, e = row["st_name"], int(row["election_year"])
        if pd.isna(row["election_year_prev"]):
            start = e - 4
        else:
            start = int(row["election_year_prev"]) + 1
        end = max(start, e - 1)
        return s, start, end

    def _fold(metric):
        def f(row):
            s, lo, hi = _window(row)
            w = sy[(sy["st_name"] == s) & (sy["year"] >= lo) & (sy["year"] <= hi)]
            if w.empty:
                return np.nan
            return float(w[metric].mean())
        return f

    ep["tenure_events_annual"] = ep.apply(_fold("conflict_events"), axis=1)
    ep["tenure_logc"] = ep.apply(_fold("log_conflict"), axis=1)

    defense = ep[ep["ruling_prev"] == "BJP"].copy()
    defense["bjp_retained"] = (defense["ruling_party"] == "BJP").astype(int)
    return defense.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helper -- 5-bin binned scatter with bootstrap CIs and LPM fit
# ---------------------------------------------------------------------------
def binned_scatter(ax, df: pd.DataFrame, xvar: str, yvar: str,
                   xlabel: str, ylabel: str,
                   nbins: int = 5, nboot: int = 2000) -> None:
    """One panel of binned scatter.  Drops rows missing xvar or yvar."""
    d = df.dropna(subset=[xvar, yvar]).copy()
    if d.empty:
        ax.text(0.5, 0.5, "no obs", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        return

    d["bin"] = pd.qcut(d[xvar], nbins, duplicates="drop")
    xs, ys, los, his = [], [], [], []
    for _, g in d.groupby("bin", observed=True):
        if len(g) == 0:
            continue
        xs.append(float(g[xvar].mean()))
        ys.append(float(g[yvar].mean()))
        boot = np.array([
            RNG.choice(g[yvar].values, size=len(g), replace=True).mean()
            for _ in range(nboot)
        ])
        los.append(float(np.percentile(boot, 2.5)))
        his.append(float(np.percentile(boot, 97.5)))
    xs, ys = np.array(xs), np.array(ys)
    los, his = np.array(los), np.array(his)

    # LPM fit with state-clustered SE.
    X = sm.add_constant(d[[xvar]].astype(float))
    res = sm.OLS(d[yvar].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["st_name"].values}
    )
    xgrid = np.linspace(d[xvar].min(), d[xvar].max(), 100)
    Xg = sm.add_constant(pd.DataFrame({xvar: xgrid}))
    pred = res.get_prediction(Xg).summary_frame(alpha=0.05)
    yhat = pred["mean"].values
    ylo = pred["mean_ci_lower"].values
    yhi = pred["mean_ci_upper"].values

    ax.fill_between(xgrid, ylo, yhi, color="#4c78a8", alpha=0.18, linewidth=0)
    ax.plot(xgrid, yhat, color="#4c78a8", linewidth=1.4, label="LPM fit (95% CI)")
    ax.errorbar(
        xs, ys,
        yerr=[ys - los, his - ys],
        fmt="o-", color="black",
        markersize=6, linewidth=1.3, capsize=3,
        label=f"{nbins} bins (95% bootstrap CI)",
    )
    ax.axhline(d[yvar].mean(), color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_panel(df: pd.DataFrame, xvar: str, yvar: str,
               xlabel: str, ylabel: str, outpath: Path,
               nbins: int = 5) -> None:
    """One PDF/JPG per panel, no in-image title."""
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    binned_scatter(ax, df, xvar, yvar, xlabel, ylabel, nbins=nbins)
    fig.tight_layout()
    save_jpg(fig, outpath)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sy_main = load_state_year_panel(use_vw_only=False)
    sy_vw = load_state_year_panel(use_vw_only=True)
    defense_main = build_defense_panel(sy_main)
    defense_vw = build_defense_panel(sy_vw)

    print(f"Main sample: N = {len(defense_main)}, retention = {defense_main['bjp_retained'].mean():.3f}")
    print(f"VW sample:   N = {len(defense_vw)}, retention = {defense_vw['bjp_retained'].mean():.3f}")

    pre = defense_main[defense_main["election_year"] < 2014].copy()
    post = defense_main[defense_main["election_year"] >= 2014].copy()
    print(f"Pre-2014:  N = {len(pre)}")
    print(f"Post-2014: N = {len(post)}")

    panels = {
        "paradox_main.jpg": (
            defense_main, "tenure_logc", "bjp_retained",
            r"Tenure-average $\ln(1 + \text{violent H-M events})$",
            "P(BJP retained at next election)",
            5,
        ),
        "paradox_vw.jpg": (
            defense_vw, "tenure_logc", "bjp_retained",
            r"Tenure-average $\ln(1 + \text{VW H-M incidents})$",
            "P(BJP retained at next election)",
            5,
        ),
        "paradox_pre2014.jpg": (
            pre, "tenure_logc", "bjp_retained",
            r"Tenure-average $\ln(1 + \text{violent H-M events})$",
            "P(BJP retained at next election)",
            min(5, max(2, pre["tenure_logc"].notna().sum() // 4)),
        ),
        "paradox_post2014.jpg": (
            post, "tenure_logc", "bjp_retained",
            r"Tenure-average $\ln(1 + \text{violent H-M events})$",
            "P(BJP retained at next election)",
            min(5, max(2, post["tenure_logc"].notna().sum() // 4)),
        ),
    }

    for fname, args in panels.items():
        out = OUT_DIR / fname
        df, x, y, xl, yl, nb = args
        make_panel(df, x, y, xl, yl, out, nbins=nb)
        h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
        n_real = df[[x, y]].dropna().shape[0]
        print(f"  {fname}  N={n_real}  sha256_16={h}")


if __name__ == "__main__":
    main()
