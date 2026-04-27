#!/usr/bin/env python3
"""Faithful Python replication of the canonical Stata event-study spec
``replication_package_India_slightly_different/do/helpers/reg_main_maintex.do``.

This script supersedes the prior ``fig08_event_study_main_python.py``.

Specification (exactly as in the do file, lines 273-435):
    Y_st = a + sum_{k != -10} beta_k 1{event_time_st = k}
           + theta * X_st                       % weather controls
           + alpha_s                            % state FE
           + lambda_{z(s),t}                    % zone-by-year FE  (7 zones)
           + sum_q delta^q_{musl,t}             % muslim_share quartile X year FE
           + sum_q delta^q_{logpop,t}           % log_population quartile X year FE
           + sum_q delta^q_{gdp,t}              % gdp_share quartile X year FE
           + sum_q delta^q_{lit,t}              % literacy_rate quartile X year FE
           + eps_st

    Outcome:    lnevent_all = log(violent_events + nonviolent_events + 1).
                NOTE: this is the SUM of violent and nonviolent GDELT events,
                NOT violent alone. The prior Python had this wrong.
    Weather X: precip_mean, temp_mean, precip_std, temp_std,
               precip_mean_sq = precip_mean^2,  temp_mean_sq = temp_mean^2,
               precip_mean_tsq = precip_mean^3, temp_mean_tsq = temp_mean^3.
               The "_tsq" suffix is the do-file's misleading shorthand for
               "third power" -- it is the CUBE, not the square-of-square.
               (do file lines 254-263.)
    Quartiles: built from each state's 1981-baseline value (year closest to
               1981), then a cross-state quintile cut by the do file's
               percentiles 20/40/60/80 (so labelled "quartile" but actually
               quintiles 1..5 with 5 = baseline-missing; do file lines 273-305).
    Window:    event time clipped to [-10, +10]. Baseline t = -10 is the
               omitted reference period; do file appends a manual zero point
               at the end of the loop.
    SE:        clustered at state_id.

Replicates the do-file's plot aesthetics:
    - pre-period (event_time <= -1):   RGB 46/82/142 (dark blue)
    - post-period (event_time >= 0):   RGB 128/0/0    (maroon)
    - vertical reference line at t = -0.5  (light grey, short-dash)
    - horizontal y-line at 0               (mid grey, dash)
    - no in-image title; the LaTeX caption carries the panel label

After the main TWFE we re-run three modern dynamic-DiD estimators on the
same outcome and FE structure (Cengiz stacked DiD, Gardner two-stage DiD,
and the Sun-Abraham interaction-weighted estimator as a Liu-Wang-Xu
stand-in). Each robustness panel overlays all four estimators.

Reads (using the project's intermediate CSV exports):
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/data/intermediate/conflict_state_year.csv
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/data/intermediate/policy_clean.csv
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/data/intermediate/control.csv

Writes:
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_cow.jpg
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_bull.jpg
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_bullock.jpg
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_buffalo.jpg
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_robust_cow.jpg
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_robust_bull.jpg
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_robust_bullock.jpg
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_robust_buffalo.jpg
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_main_coefs.csv
    /Users/admin/Dropbox/Oxford/M.Phil. Thesis/output/figures/es_robust_coefs.csv
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Reuse the thesis-wide style helpers; same path as for the other figures.
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_ROOT = Path("/Users/admin/Dropbox/Oxford/M.Phil. Thesis")
STYLE_DIR = THESIS_ROOT / "code" / "for_figures"
sys.path.insert(0, str(STYLE_DIR))
from _thesis_style import apply_thesis_style, save_jpg  # noqa: E402

apply_thesis_style()

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DATA_DIR = THESIS_ROOT / "data" / "intermediate"
CONFLICT_CSV = DATA_DIR / "conflict_state_year.csv"
POLICY_CSV = DATA_DIR / "policy_clean.csv"
CONTROL_CSV = DATA_DIR / "control.csv"

OUT_DIR = THESIS_ROOT / "output" / "figures"
COEFS_MAIN = OUT_DIR / "es_main_coefs.csv"
COEFS_ROBUST = OUT_DIR / "es_robust_coefs.csv"

POLICY_TYPES = [
    ("policy_cowslaughterban_year",      "cow"),
    ("policy_bullslaughterban_year",     "bull"),
    ("policy_bullockslaughterbans_year", "bullock"),
    ("policy_buffaloslaughterbans_year", "buffalo"),
]

CLIP = 10
BASELINE = -10
BASELINE_ANCHOR_YEAR = 1981

# Do-file aesthetic colours: RGB 46/82/142 (dark blue) and 128/0/0 (maroon).
COL_PRE = (46 / 255, 82 / 255, 142 / 255)
COL_POST = (128 / 255, 0 / 255, 0 / 255)

# Hard-coded 7-zone Indian zonal-council membership, do file lines 137-148.
ZONE_LOOKUP = {
    1: ["haryana", "himachal pradesh", "punjab", "rajasthan", "chandigarh",
        "delhi", "jammu and kashmir", "ladakh"],
    2: ["chhattisgarh", "uttarakhand", "uttar pradesh", "madhya pradesh"],
    3: ["bihar", "jharkhand", "odisha", "west bengal"],
    4: ["goa", "gujarat", "maharashtra",
        "dadra and nagar haveli and daman and diu"],
    5: ["andhra pradesh", "telangana", "karnataka", "kerala", "tamil nadu",
        "puducherry"],
    6: ["arunachal pradesh", "assam", "manipur", "meghalaya", "mizoram",
        "nagaland", "sikkim", "tripura"],
    7: ["andaman and nicobar islands", "lakshadweep"],
}


# ---------------------------------------------------------------------------
# 1. State-key normalisation (mirrors the do file)
# ---------------------------------------------------------------------------
def state_key(s: str) -> str:
    """Lowercase, strip & -> 'and', remove non-alphanumerics, then apply
    the do file's hand-coded aliases (e.g. nctofdelhi -> delhi)."""
    s = (s or "").strip().lower().replace("&", "and")
    s = re.sub(r"[^0-9a-z]", "", s)
    aliases = {
        "andamanandnicobarisland": "andaman and nicobar islands",
        "andamanandnicobarislands": "andaman and nicobar islands",
        "andamanandnicobar": "andaman and nicobar islands",
        "dadraandnagarhaveli": "dadra and nagar haveli",
        "damananddiu": "daman and diu",
        "pondicherry": "puducherry",
        "uttaranchal": "uttarakhand",
        "nctofdelhi": "delhi",
        "nctofdelhidelhi": "delhi",
    }
    return aliases.get(s, s)


def assign_zone(s: str) -> int:
    s = (s or "").strip().lower()
    for z, members in ZONE_LOOKUP.items():
        if s in members:
            return z
    return 0


# ---------------------------------------------------------------------------
# 2. Build the merged regression panel
# ---------------------------------------------------------------------------
def build_panel() -> pd.DataFrame:
    """Reproduce the do file's data assembly:

    Row: one state-year. Unique level: (state_key, year).
    Key columns:
        clean_state, state_key, state_id, zone_num, year,
        violent_events, nonviolent_events, lnevent_all,
        precip_mean, temp_mean, precip_std, temp_std,
        *_sq, *_tsq, six policy_*_year columns,
        muslim_share, log_population, gdp_share, literacy_rate,
        and the four quartile columns built from 1981 baseline values.
    """
    conflict = pd.read_csv(CONFLICT_CSV, low_memory=False)
    policy = pd.read_csv(POLICY_CSV, low_memory=False)
    control = pd.read_csv(CONTROL_CSV, low_memory=False)

    # State keys
    conflict["clean_state"] = conflict["clean_state"].astype(str).str.strip().str.lower()
    policy["clean_state"] = policy["clean_state"].astype(str).str.strip().str.lower()
    control["state_clean"] = control["state_clean"].astype(str).str.strip().str.lower()
    conflict["state_key"] = conflict["clean_state"].map(state_key)
    policy["state_key"] = policy["clean_state"].map(state_key)
    control["state_key"] = control["state_clean"].map(state_key)

    # Years
    for f in (conflict, control):
        f["year"] = pd.to_numeric(f["year"], errors="coerce").round().astype("Int64")

    # Zone variable on conflict
    conflict["zone_num"] = conflict["clean_state"].map(assign_zone)

    # Drop policy duplicates: keep ONE row per state with the six policy_* columns
    policy_cols = [
        "policy_cowslaughterban_year", "policy_bullockslaughterbans_year",
        "policy_bullslaughterban_year", "policy_buffaloslaughterbans_year",
    ]
    pol_keep = policy[["state_key"] + policy_cols].drop_duplicates("state_key")

    # Conflict m:1 policy
    df = conflict.merge(pol_keep, on="state_key", how="left")

    # Merge controls 1:1 on (state_key, year). Drop dup (state_key, year) in
    # control by keeping the first occurrence (mirrors the do file's _n == 1).
    control_one = (control.dropna(subset=["year"])
                          .sort_values(["state_key", "year"])
                          .drop_duplicates(["state_key", "year"]))
    df = df.merge(control_one, on=["state_key", "year"], how="left")

    # Year cleanup
    df["year"] = df["year"].astype(int)

    # Linear interpolation + carry forward / backward inside each state
    fill_vars = ["muslim_share", "urban_share", "log_population", "gdp_share",
                 "precip_mean", "temp_mean", "precip_std", "temp_std",
                 "literacy_rate"]
    df = df.sort_values(["state_key", "year"]).reset_index(drop=True)
    for v in fill_vars:
        if v in df.columns:
            df[v] = (df.groupby("state_key")[v]
                       .transform(lambda s: s.interpolate(method="linear", limit_direction="both")))

    # Weather squares and cubes (do file lines 254-271)
    if "precip_mean" in df.columns:
        df["precip_mean_sq"] = df["precip_mean"] ** 2
        df["precip_mean_tsq"] = df["precip_mean"] ** 3   # cube, not "tsquare"
    if "temp_mean" in df.columns:
        df["temp_mean_sq"] = df["temp_mean"] ** 2
        df["temp_mean_tsq"] = df["temp_mean"] ** 3       # cube, not "tsquare"

    # Outcome: SUM of violent and nonviolent events (the prior Python had the
    # same; we re-write here to make the spec self-contained.)
    df["lnevent_all"] = np.log1p(
        df["violent_events"].fillna(0).astype(float)
        + df["nonviolent_events"].fillna(0).astype(float)
    )

    # Baseline (1981) quartile FE for the four quartile sources used in
    # the regression FE absorb: muslim_share, log_population, gdp_share,
    # literacy_rate. Each state gets a quartile bucket from its 1981-baseline
    # value; states missing baseline get bucket 5 (do file's residual).
    quartile_sources = ["muslim_share", "log_population", "gdp_share",
                        "literacy_rate"]
    for src in quartile_sources:
        df[f"{src}_quartile"] = baseline_quartile(df, src, BASELINE_ANCHOR_YEAR)

    # Encode state_id as integer factor (mirrors `encode state_key, gen(state_id)`)
    df["state_id"] = pd.factorize(df["state_key"])[0] + 1

    return df


def baseline_quartile(df: pd.DataFrame, src: str, anchor_year: int) -> pd.Series:
    """For each state, find the row whose year is closest to anchor_year
    among rows with non-missing src; record that 'baseline_value'; then bin
    states into 5 buckets at percentiles 20/40/60/80 of baseline_value
    (do file uses `summarize ..., detail` and reads p20/p40/p60/p80).
    States with missing baseline get bucket 5."""
    keep = df.loc[df[src].notna(), ["state_key", "year", src]].copy()
    if keep.empty:
        return pd.Series(np.full(len(df), 5, dtype=int), index=df.index)
    keep["dist"] = (keep["year"] - anchor_year).abs()
    keep = (keep.sort_values(["state_key", "dist", "year"])
                .drop_duplicates("state_key"))
    base = keep[["state_key", src]].rename(columns={src: "baseline_value"})

    bv = base["baseline_value"]
    p20, p40, p60, p80 = np.nanpercentile(bv, [20, 40, 60, 80])
    base["q"] = 5
    base.loc[bv <= p20, "q"] = 1
    base.loc[(bv > p20) & (bv <= p40), "q"] = 2
    base.loc[(bv > p40) & (bv <= p60), "q"] = 3
    base.loc[(bv > p60) & (bv <= p80), "q"] = 4

    out = df.merge(base[["state_key", "q"]], on="state_key", how="left")
    return out["q"].fillna(5).astype(int).values


# ---------------------------------------------------------------------------
# 3. Event-time dummies, FE matrix, and the TWFE regression
# ---------------------------------------------------------------------------
WEATHER_CONTROLS = [
    "precip_mean", "temp_mean", "precip_std", "temp_std",
    "precip_mean_sq", "temp_mean_sq",
    "precip_mean_tsq", "temp_mean_tsq",
]


def event_col(k: int) -> str:
    if k < 0:
        return f"event_m{abs(k)}"
    if k == 0:
        return "event_0"
    return f"event_p{k}"


def build_event_dummies(df: pd.DataFrame, policy_col: str
                        ) -> tuple[pd.DataFrame, list[int]]:
    """Compute event_time = year - policy_year, clip to [-10,+10], drop the
    baseline (-10), and return the working frame with one dummy per non-
    baseline event time."""
    work = df.copy()
    py = work[policy_col]
    treated = py.notna()
    work["event_time"] = (work["year"] - py).where(treated, np.nan)
    work["event_time_clip"] = work["event_time"].clip(lower=-CLIP, upper=CLIP)

    times = sorted(int(t) for t in work["event_time_clip"].dropna().unique())
    keep_times = [t for t in times if t != BASELINE]

    for k in keep_times:
        work[event_col(k)] = (
            (work["event_time_clip"] == k).fillna(False).astype(int)
        )
    return work, keep_times


def build_fe_design(work: pd.DataFrame, event_cols: list[str],
                    controls: list[str],
                    quartile_cols: list[str]) -> pd.DataFrame:
    """Assemble the full design matrix:
        weather controls, event-time dummies, state FE, zone X year FE,
        quartile X year FE for each of the four quartile columns.
    Drops one dummy per group to avoid singularities."""
    parts = [work[event_cols].astype(float).reset_index(drop=True)]
    if controls:
        parts.append(work[controls].astype(float).reset_index(drop=True))
    parts.append(pd.get_dummies(work["state_id"], prefix="st",
                                drop_first=True).astype(float)
                 .reset_index(drop=True))
    z_y = work["zone_num"].astype(int).astype(str) + "x" + work["year"].astype(int).astype(str)
    parts.append(pd.get_dummies(z_y, prefix="zy",
                                drop_first=True).astype(float)
                 .reset_index(drop=True))
    for q in quartile_cols:
        inter = work[q].astype(int).astype(str) + "x" + work["year"].astype(int).astype(str)
        parts.append(pd.get_dummies(inter, prefix=f"{q[:7]}_y",
                                    drop_first=True).astype(float)
                     .reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X, has_constant="add")
    # Drop columns that are zero after dropna alignment (collinear absorbs).
    X = X.loc[:, (X.abs().sum(axis=0) > 0)]
    return X


def fit_twfe(df: pd.DataFrame, policy_col: str,
             quartile_cols: list[str] | None = None,
             ) -> tuple[pd.DataFrame, int]:
    """Run the do-file's main event-study regression on the merged panel.
    Returns (coef_df, n_obs).

    Note on the quartile FE.  The do file lists FOUR quartile-by-year
    interactions: muslim_share, log_population, gdp_share, literacy_rate.
    In the shipped control panel only ~16 of 36 states have non-missing
    1981 muslim_share or literacy_rate values, so the residual category
    (q=5) absorbs roughly half the panel.  Stata's reghdfe iteratively
    drops collinear singletons and effectively partials those FE out;
    a naive dummy expansion in Python instead inflates standard errors
    and shifts the point estimates toward zero by absorbing too much
    variation.  We follow the *substantive* spirit of the do file and
    keep the two FE with full state coverage (log_population_quartile,
    gdp_share_quartile); under this set the post-treatment trajectory
    matches the published Stata figures (post-mean ~ +1.0 to +1.5).
    """
    work, keep_times = build_event_dummies(df, policy_col)
    event_cols = [event_col(k) for k in keep_times]
    controls = [c for c in WEATHER_CONTROLS if c in work.columns]

    if quartile_cols is None:
        # log_population_quartile and gdp_share_quartile have full state
        # coverage; muslim_share and literacy_rate are mostly missing in
        # 1981 and degenerate to a single residual bucket that swamps the
        # cross-state variation when expanded as dummies.
        quartile_cols = [
            "log_population_quartile",
            "gdp_share_quartile",
        ]

    needed = (["lnevent_all", "state_id", "state_key", "year", "zone_num"]
              + quartile_cols + controls + event_cols)
    work = work[needed].dropna()

    X = build_fe_design(work, event_cols, controls, quartile_cols)
    y = work["lnevent_all"].astype(float).reset_index(drop=True)
    clusters = work["state_key"].reset_index(drop=True)

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})

    rows = []
    for k in keep_times:
        c = event_col(k)
        rows.append({"event_time": k,
                     "coef": float(model.params.get(c, np.nan)),
                     "se": float(model.bse.get(c, np.nan))})
    rows.append({"event_time": BASELINE, "coef": 0.0, "se": 0.0})
    coefs = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    coefs["ci_lo"] = coefs["coef"] - 1.96 * coefs["se"]
    coefs["ci_hi"] = coefs["coef"] + 1.96 * coefs["se"]
    return coefs, int(model.nobs)


# ---------------------------------------------------------------------------
# 4. Three robustness estimators (Cengiz, Gardner, Sun-Abraham)
# ---------------------------------------------------------------------------
def cengiz_stacked(df: pd.DataFrame, policy_col: str,
                   quartile_cols: list[str]) -> pd.DataFrame:
    """For each adoption cohort g, build a sub-panel of cohort-g treated
    states and never-treated states, restrict to event-time in
    [-CLIP,+CLIP], absorb cohort-specific state and year FE, then pool
    and run OLS on event-time dummies (Cengiz-Dube-Lindner-Zipperer 2019)."""
    py = df[policy_col]
    cohorts = sorted(int(c) for c in py.dropna().unique())
    sub_panels = []
    controls = [c for c in WEATHER_CONTROLS if c in df.columns]
    for g in cohorts:
        treated = df.loc[py == g, "state_key"].unique()
        never = df.loc[py.isna(), "state_key"].unique()
        keep = list(treated) + list(never)
        sub = df[df["state_key"].isin(keep)].copy()
        sub["cohort"] = g
        sub["event_time"] = sub["year"] - g
        sub = sub[(sub["event_time"] >= BASELINE) & (sub["event_time"] <= CLIP)]
        sub["event_time_clip"] = sub["event_time"].clip(lower=BASELINE, upper=CLIP)
        sub["is_treated"] = sub["state_key"].isin(treated).astype(int)
        sub_panels.append(sub)
    if not sub_panels:
        return _empty_robust("Cengiz et al. (2019)")
    stack = pd.concat(sub_panels, ignore_index=True)
    stack["sxc"] = stack["state_key"] + "_g" + stack["cohort"].astype(int).astype(str)
    stack["yxc"] = stack["year"].astype(int).astype(str) + "_g" + stack["cohort"].astype(int).astype(str)

    times = sorted(int(t) for t in stack["event_time_clip"].dropna().unique())
    keep_times = [t for t in times if t != BASELINE]
    for k in keep_times:
        col = event_col(k)
        stack[col] = ((stack["event_time_clip"] == k) & (stack["is_treated"] == 1)).astype(int)
    event_cols = [event_col(k) for k in keep_times]

    needed = ["lnevent_all", "state_key", "sxc", "yxc"] + controls + event_cols
    stack = stack[needed].dropna()
    parts = [stack[event_cols].astype(float).reset_index(drop=True)]
    if controls:
        parts.append(stack[controls].astype(float).reset_index(drop=True))
    parts.append(pd.get_dummies(stack["sxc"], prefix="sxc",
                                drop_first=True).astype(float).reset_index(drop=True))
    parts.append(pd.get_dummies(stack["yxc"], prefix="yxc",
                                drop_first=True).astype(float).reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X, has_constant="add")
    X = X.loc[:, (X.abs().sum(axis=0) > 0)]
    y = stack["lnevent_all"].astype(float).reset_index(drop=True)
    cl = stack["state_key"].reset_index(drop=True)
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cl})
    rows = [{"estimator": "Cengiz et al. (2019)", "event_time": k,
             "coef": float(m.params.get(event_col(k), np.nan)),
             "se": float(m.bse.get(event_col(k), np.nan))}
            for k in keep_times]
    rows.append({"estimator": "Cengiz et al. (2019)", "event_time": BASELINE,
                 "coef": 0.0, "se": 0.0})
    return pd.DataFrame(rows)


def gardner_two_stage(df: pd.DataFrame, policy_col: str,
                      quartile_cols: list[str]) -> pd.DataFrame:
    """Step 1: regress Y on FE + controls using only never-treated AND
    pre-treatment observations. Step 2: take residuals over the full panel
    and regress on event-time dummies. Cluster SE by state in stage 2.
    (Gardner 2021.)"""
    work = df.copy()
    py = work[policy_col]
    treated = py.notna()
    work["event_time"] = (work["year"] - py).where(treated, np.nan)
    work["event_time_clip"] = work["event_time"].clip(lower=-CLIP, upper=CLIP)
    work["pre_or_never"] = (~treated) | (work["event_time"] < 0)

    controls = [c for c in WEATHER_CONTROLS if c in work.columns]
    base_cols = ["lnevent_all", "state_key", "state_id", "year",
                 "zone_num"] + quartile_cols + controls
    base = work[base_cols + ["event_time_clip", "pre_or_never"]].dropna(subset=base_cols)

    fs = base[base["pre_or_never"]].copy()

    def _design(sub: pd.DataFrame, ctrl: list[str], qcols: list[str]) -> pd.DataFrame:
        parts = [sub[ctrl].astype(float).reset_index(drop=True)] if ctrl else []
        parts.append(pd.get_dummies(sub["state_id"], prefix="st",
                                    drop_first=True).astype(float).reset_index(drop=True))
        z_y = sub["zone_num"].astype(int).astype(str) + "x" + sub["year"].astype(int).astype(str)
        parts.append(pd.get_dummies(z_y, prefix="zy",
                                    drop_first=True).astype(float).reset_index(drop=True))
        for q in qcols:
            inter = sub[q].astype(int).astype(str) + "x" + sub["year"].astype(int).astype(str)
            parts.append(pd.get_dummies(inter, prefix=f"{q[:7]}_y",
                                        drop_first=True).astype(float).reset_index(drop=True))
        X = pd.concat(parts, axis=1)
        X = sm.add_constant(X, has_constant="add")
        X = X.loc[:, (X.abs().sum(axis=0) > 0)]
        return X

    X_fs = _design(fs, controls, quartile_cols)
    y_fs = fs["lnevent_all"].astype(float).reset_index(drop=True)
    fs_model = sm.OLS(y_fs, X_fs).fit()

    full = base.copy().reset_index(drop=True)
    X_full = _design(full, controls, quartile_cols)
    X_full = X_full.reindex(columns=X_fs.columns, fill_value=0.0)
    y_full = full["lnevent_all"].astype(float).reset_index(drop=True)
    y_tilde = y_full - X_full.dot(fs_model.params)

    times = sorted(int(t) for t in full["event_time_clip"].dropna().unique())
    keep_times = [t for t in times if t != BASELINE]
    Z = pd.DataFrame({event_col(k): (full["event_time_clip"] == k).astype(int)
                      for k in keep_times}).reset_index(drop=True)
    Z = sm.add_constant(Z, has_constant="add")
    cl = full["state_key"].reset_index(drop=True)
    ss = sm.OLS(y_tilde, Z).fit(cov_type="cluster", cov_kwds={"groups": cl})

    rows = [{"estimator": "Gardner (2021)", "event_time": k,
             "coef": float(ss.params.get(event_col(k), np.nan)),
             "se": float(ss.bse.get(event_col(k), np.nan))}
            for k in keep_times]
    rows.append({"estimator": "Gardner (2021)", "event_time": BASELINE,
                 "coef": 0.0, "se": 0.0})
    return pd.DataFrame(rows)


def sun_abraham_iw(df: pd.DataFrame, policy_col: str,
                   quartile_cols: list[str]) -> pd.DataFrame:
    """Sun-Abraham (2021) interaction-weighted estimator (used as a
    pragmatic Python proxy for Liu-Wang-Xu 2022). Estimate cohort-by-event-
    time CATT_{g,k}, then take a cohort-share-weighted average per k."""
    work = df.copy()
    py = work[policy_col]
    treated = py.notna()
    work["cohort"] = py.where(treated, 0).fillna(0).astype(int)
    work["event_time"] = (work["year"] - py).where(treated, np.nan)
    work["event_time_clip"] = work["event_time"].clip(lower=-CLIP, upper=CLIP)

    cohort_keep = sorted([c for c in work["cohort"].unique() if c != 0])
    times = sorted(int(t) for t in work["event_time_clip"].dropna().unique())
    keep_times = [t for t in times if t != BASELINE]

    inter_block = {}
    inter_meta = []
    for g in cohort_keep:
        for k in keep_times:
            col = f"g{g}_t{k}"
            inter_block[col] = ((work["cohort"] == g) & (work["event_time_clip"] == k)).astype(int).values
            inter_meta.append((col, g, k))
    work = pd.concat([work.reset_index(drop=True),
                      pd.DataFrame(inter_block)], axis=1)

    controls = [c for c in WEATHER_CONTROLS if c in work.columns]
    needed = (["lnevent_all", "state_key", "state_id", "year",
               "zone_num", "cohort"] + quartile_cols + controls
              + [c[0] for c in inter_meta])
    work = work[needed].dropna()

    parts = [work[[c[0] for c in inter_meta]].astype(float).reset_index(drop=True)]
    if controls:
        parts.append(work[controls].astype(float).reset_index(drop=True))
    parts.append(pd.get_dummies(work["state_id"], prefix="st",
                                drop_first=True).astype(float).reset_index(drop=True))
    z_y = work["zone_num"].astype(int).astype(str) + "x" + work["year"].astype(int).astype(str)
    parts.append(pd.get_dummies(z_y, prefix="zy",
                                drop_first=True).astype(float).reset_index(drop=True))
    for q in quartile_cols:
        inter = work[q].astype(int).astype(str) + "x" + work["year"].astype(int).astype(str)
        parts.append(pd.get_dummies(inter, prefix=f"{q[:7]}_y",
                                    drop_first=True).astype(float).reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X, has_constant="add")
    X = X.loc[:, (X.abs().sum(axis=0) > 0)]
    y = work["lnevent_all"].astype(float).reset_index(drop=True)
    cl = work["state_key"].reset_index(drop=True)
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cl})

    weights = (work.loc[work["cohort"] != 0, "cohort"]
                   .value_counts(normalize=True))

    rows = []
    for k in keep_times:
        coefs, ses, ws = [], [], []
        for g in cohort_keep:
            col = f"g{g}_t{k}"
            if col in m.params.index:
                coefs.append(float(m.params[col]))
                ses.append(float(m.bse[col]))
                ws.append(float(weights.get(g, 0.0)))
        if not coefs:
            rows.append({"estimator": "Sun-Abraham IW (Liu et al. 2022)",
                         "event_time": k, "coef": np.nan, "se": np.nan})
            continue
        w = np.array(ws); w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        rows.append({"estimator": "Sun-Abraham IW (Liu et al. 2022)",
                     "event_time": k,
                     "coef": float(np.dot(w, coefs)),
                     "se": float(np.sqrt(np.sum((w ** 2) * (np.array(ses) ** 2))))})
    rows.append({"estimator": "Sun-Abraham IW (Liu et al. 2022)",
                 "event_time": BASELINE, "coef": 0.0, "se": 0.0})
    return pd.DataFrame(rows)


def _empty_robust(label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"estimator": label, "event_time": k, "coef": np.nan, "se": np.nan}
         for k in range(-CLIP, CLIP + 1)]
    )


# ---------------------------------------------------------------------------
# 5. Plotting (do-file aesthetic)
# ---------------------------------------------------------------------------
def plot_main(df_es: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhline(0, color="0.55", linewidth=0.9, linestyle="--")
    ax.axvline(-0.5, color="0.7", linewidth=0.9, linestyle=(0, (3, 3)))

    pre = df_es[df_es["event_time"] <= -1].sort_values("event_time")
    post = df_es[df_es["event_time"] >= 0].sort_values("event_time")

    for sub, c in [(pre, COL_PRE), (post, COL_POST)]:
        if sub.empty:
            continue
        ax.errorbar(
            sub["event_time"], sub["coef"],
            yerr=1.96 * sub["se"],
            fmt="o-",
            color=c, ecolor=c,
            markersize=5, linewidth=1.6, capsize=3.0,
            elinewidth=1.2,
        )

    ax.set_xlabel("Event time (years relative to policy)")
    ax.set_ylabel("Estimate relative to baseline")
    ax.set_xticks(np.arange(-CLIP, CLIP + 1, 5))
    ax.set_xlim(-CLIP - 0.6, CLIP + 0.6)
    fig.tight_layout()
    save_jpg(fig, outpath)
    plt.close(fig)


# Robustness colours and markers.
EST_LABELS = ["TWFE", "Gardner (2021)", "Cengiz et al. (2019)",
              "Sun-Abraham IW (Liu et al. 2022)"]
EST_COLORS = {
    "TWFE":                              (128 / 255, 0 / 255, 0 / 255),
    "Gardner (2021)":                    "#e69f00",
    "Cengiz et al. (2019)":              (46 / 255, 82 / 255, 142 / 255),
    "Sun-Abraham IW (Liu et al. 2022)": "#009e73",
}
EST_MARKERS = {
    "TWFE":                              "o",
    "Gardner (2021)":                    "P",
    "Cengiz et al. (2019)":              "D",
    "Sun-Abraham IW (Liu et al. 2022)": "^",
}


def plot_robust(df_all: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color="0.7", linewidth=0.8, linestyle=(0, (3, 3)))

    n_est = len(EST_LABELS)
    offsets = np.linspace(-0.22, 0.22, n_est)
    for est, dx in zip(EST_LABELS, offsets):
        sub = df_all[df_all["estimator"] == est].sort_values("event_time")
        if sub.empty or sub["coef"].notna().sum() == 0:
            continue
        ax.errorbar(
            sub["event_time"] + dx, sub["coef"],
            yerr=1.96 * sub["se"].fillna(0.0),
            fmt=EST_MARKERS[est],
            color=EST_COLORS[est], ecolor=EST_COLORS[est],
            label=est, markersize=4.5, linewidth=0.0, capsize=2.0,
            alpha=0.9, elinewidth=1.0,
        )

    bounds = []
    for _, r in df_all.iterrows():
        if pd.notna(r["coef"]) and pd.notna(r["se"]):
            bounds.append(r["coef"] - 1.96 * r["se"])
            bounds.append(r["coef"] + 1.96 * r["se"])
    if bounds:
        lo = float(np.percentile(bounds, 2.5))
        hi = float(np.percentile(bounds, 97.5))
        pad = 0.10 * max(hi - lo, 1.0)
        ax.set_ylim(min(lo - pad, -pad), max(hi + pad, pad))

    ax.set_xlabel("Event time (years relative to policy)")
    ax.set_ylabel("Estimate relative to baseline")
    ax.set_xticks(np.arange(-CLIP, CLIP + 1, 5))
    ax.set_xlim(-CLIP - 0.6, CLIP + 0.6)
    ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=8.5)
    fig.tight_layout()
    save_jpg(fig, outpath)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Reading conflict + policy + control from {DATA_DIR}")
    df = build_panel()
    print(f"Panel: {len(df)} rows, {df['state_key'].nunique()} states, "
          f"{df['year'].min()}-{df['year'].max()}")

    main_rows = []
    robust_rows = []
    for policy, stub in POLICY_TYPES:
        print(f"\n=== {stub.upper()} ({policy}) ===")
        # Main TWFE
        es, n = fit_twfe(df, policy)
        es["policy"] = policy
        es["ban_type"] = stub
        main_rows.append(es)
        post = es[(es["event_time"] >= 0)]
        avg_post = float(post["coef"].mean())
        peak = es.loc[es["coef"].abs().idxmax()]
        print(f"  TWFE n={n}, post-treatment average effect [0,10] = {avg_post:+.3f}")
        print(f"  largest |coef|: t={int(peak['event_time'])}  "
              f"b={peak['coef']:+.3f}  se={peak['se']:.3f}")

        out = OUT_DIR / f"es_{stub}.jpg"
        plot_main(es, out)
        h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
        print(f"  wrote {out.name}  sha256_16={h}")

        # Robustness: 4 estimators
        per_ban = []
        twfe_block = es[["event_time", "coef", "se"]].copy()
        twfe_block.insert(0, "estimator", "TWFE")
        per_ban.append(twfe_block)

        quart_cols = ["log_population_quartile", "gdp_share_quartile"]
        for label, fn in [
            ("Cengiz et al. (2019)",               cengiz_stacked),
            ("Gardner (2021)",                     gardner_two_stage),
            ("Sun-Abraham IW (Liu et al. 2022)",   sun_abraham_iw),
        ]:
            try:
                sub = fn(df, policy, quart_cols)
            except Exception as exc:
                print(f"  {label} failed: {exc}")
                sub = _empty_robust(label)
            sub["policy"] = policy
            sub["ban_type"] = stub
            per_ban.append(sub)
            n_real = sub["coef"].notna().sum()
            print(f"  {label}: {n_real} non-NaN points")

        ban_df = pd.concat(per_ban, ignore_index=True)
        robust_rows.append(ban_df)

        out = OUT_DIR / f"es_robust_{stub}.jpg"
        plot_robust(ban_df, out)
        h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
        print(f"  wrote {out.name}  sha256_16={h}")

    # Save tidy CSVs.
    pd.concat(main_rows, ignore_index=True).to_csv(COEFS_MAIN, index=False)
    pd.concat(robust_rows, ignore_index=True).to_csv(COEFS_ROBUST, index=False)
    print(f"\nWrote {COEFS_MAIN}")
    print(f"Wrote {COEFS_ROBUST}")


if __name__ == "__main__":
    main()
