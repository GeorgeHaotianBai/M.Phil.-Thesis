from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def solve_majority_effort_star(*, S: float, E_m: float, N_M: int, Lambda: float) -> float:
    if E_m <= 0.0 or S <= 0.0:
        return 0.0

    def equilibrium_condition(E_M: float) -> float:
        return E_M / N_M - (Lambda * S * E_m) / (S * E_M + E_m) ** 2

    lo = 0.0
    hi = 1.0
    while equilibrium_condition(hi) < 0.0:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("Failed to bracket the unique equilibrium root.")

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if equilibrium_condition(mid) < 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.6,
        }
    )


def style_capacity_regions(ax: plt.Axes, max_ratio: float) -> None:
    ax.axvspan(0.0, 1.0, color="0.90", alpha=0.7, zorder=0)
    ax.axvspan(1.0, max_ratio, color="0.96", alpha=1.0, zorder=0)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".pdf"))
    fig.savefig(base_path.with_suffix(".jpg"), dpi=300)
    plt.close(fig)


def main() -> None:
    # Parameters used in the paper's comparative-static illustration.
    N_M = 100
    Lambda = 1.0
    E_m = 5.0

    S_bar = 2.0 * E_m / np.sqrt(N_M * Lambda)
    capacity_ratio_grid = np.linspace(0.0, 6.0, 500)
    S_grid = capacity_ratio_grid * S_bar

    E_star = np.array(
        [solve_majority_effort_star(S=float(S), E_m=E_m, N_M=N_M, Lambda=Lambda) for S in S_grid]
    )
    Z_star = S_grid * E_star
    E_peak = 0.5 * np.sqrt(N_M * Lambda)

    illustrative_ratios = {
        "Dove": 0.30,
        "Intermediate Hawk": 0.80,
        "Strong Hawk": 4.00,
    }
    illustrative_S = {label: ratio * S_bar for label, ratio in illustrative_ratios.items()}
    illustrative_E = {
        label: solve_majority_effort_star(S=S_value, E_m=E_m, N_M=N_M, Lambda=Lambda)
        for label, S_value in illustrative_S.items()
    }
    illustrative_Z = {label: illustrative_S[label] * illustrative_E[label] for label in illustrative_S}

    set_plot_style()
    out_dir = Path(__file__).resolve().parent

    fig_left, ax_left = plt.subplots(figsize=(5.1, 4.3))
    style_capacity_regions(ax_left, float(capacity_ratio_grid.max()))
    ax_left.plot(capacity_ratio_grid, E_star, color="black", linewidth=2.1)
    ax_left.scatter([1.0], [E_peak], color="black", s=28, zorder=5)
    ax_left.annotate(
        r"Peak at $S_k=\overline{S}(E_m)$",
        xy=(1.0, E_peak),
        xytext=(1.55, E_peak + 0.35),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
    )

    for label, ratio in illustrative_ratios.items():
        ax_left.scatter([ratio], [illustrative_E[label]], color="black", s=26, zorder=5)

    ax_left.annotate(
        "Dove",
        xy=(illustrative_ratios["Dove"], illustrative_E["Dove"]),
        xytext=(0.57, illustrative_E["Dove"] - 0.85),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
    )
    ax_left.annotate(
        "Intermediate Hawk",
        xy=(illustrative_ratios["Intermediate Hawk"], illustrative_E["Intermediate Hawk"]),
        xytext=(1.55, illustrative_E["Intermediate Hawk"] - 0.78),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
    )
    ax_left.annotate(
        "Strong Hawk",
        xy=(illustrative_ratios["Strong Hawk"], illustrative_E["Strong Hawk"]),
        xytext=(4.35, illustrative_E["Strong Hawk"] + 0.18),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
    )

    ax_left.text(0.17, 0.93, "Grass-roots activation", transform=ax_left.transAxes, fontsize=10)
    ax_left.text(0.63, 0.93, "State substitution", transform=ax_left.transAxes, fontsize=10)
    ax_left.set_xlabel(r"Relative state capacity $S_k / \overline{S}(E_m)$", fontsize=11.5)
    ax_left.set_ylabel(r"Majority mobilization $E_M^*(S_k,E_m)$", fontsize=11.5)
    ax_left.set_xlim(0.0, float(capacity_ratio_grid.max()))
    ax_left.set_ylim(0.0, float(E_star.max()) * 1.20)
    save_figure(fig_left, out_dir / "fig_Sk_statics_mobilization")

    fig_right, ax_right = plt.subplots(figsize=(5.1, 4.3))
    style_capacity_regions(ax_right, float(capacity_ratio_grid.max()))
    ax_right.plot(capacity_ratio_grid, Z_star, color="black", linewidth=2.1)
    for label, ratio in illustrative_ratios.items():
        ax_right.scatter([ratio], [illustrative_Z[label]], color="black", s=26, zorder=5)

    ax_right.annotate(
        r"$Z_k^*=S_kE_M^*$ keeps rising",
        xy=(3.1, float(np.interp(3.1, capacity_ratio_grid, Z_star))),
        xytext=(1.55, float(np.interp(1.55, capacity_ratio_grid, Z_star)) + 4.2),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
    )
    ax_right.annotate(
        "Strong Hawk",
        xy=(illustrative_ratios["Strong Hawk"], illustrative_Z["Strong Hawk"]),
        xytext=(4.28, illustrative_Z["Strong Hawk"] - 1.5),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
    )
    ax_right.set_xlabel(r"Relative state capacity $S_k / \overline{S}(E_m)$", fontsize=11.5)
    ax_right.set_ylabel(r"Effective enforcement $Z_k^*=S_kE_M^*(S_k,E_m)$", fontsize=11.5)
    ax_right.set_xlim(0.0, float(capacity_ratio_grid.max()))
    ax_right.set_ylim(0.0, float(Z_star.max()) * 1.12)
    save_figure(fig_right, out_dir / "fig_Sk_statics_enforcement")


if __name__ == "__main__":
    main()
