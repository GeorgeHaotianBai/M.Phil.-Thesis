from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def solve_majority_effort_star(*, S: float, E_m: float, N_M: int, Lambda: float) -> float:
    if E_m <= 0:
        return 0.0
    if S <= 0:
        return 0.0

    def f(E_M: float) -> float:
        return E_M / N_M - (Lambda * S * E_m) / (S * E_M + E_m) ** 2

    lo = 0.0
    hi = 1.0
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("Failed to bracket the unique equilibrium root.")

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    # Parameters for Figure 2.
    N_M = 100
    Lambda = 1.0

    S_A = 1.0  # Hawk
    S_B = 0.4  # Dove

    E_m_grid = np.linspace(0.0, 12.0, 400)
    E_star_A = np.array(
        [solve_majority_effort_star(S=S_A, E_m=float(Em), N_M=N_M, Lambda=Lambda) for Em in E_m_grid]
    )
    E_star_B = np.array(
        [solve_majority_effort_star(S=S_B, E_m=float(Em), N_M=N_M, Lambda=Lambda) for Em in E_m_grid]
    )

    # Peak points from Lemma: E_m^peak(S)= S * sqrt(N_M * Lambda)/2.
    E_M_peak = 0.5 * np.sqrt(N_M * Lambda)
    E_m_peak_A = 0.5 * S_A * np.sqrt(N_M * Lambda)
    E_m_peak_B = 0.5 * S_B * np.sqrt(N_M * Lambda)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(E_m_grid, E_star_A, color="black", linewidth=2.0, label=r"Hawk ($S_A$ high)")
    ax.plot(
        E_m_grid,
        E_star_B,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label=r"Dove ($S_B$ low)",
    )

    ax.scatter([E_m_peak_A, E_m_peak_B], [E_M_peak, E_M_peak], color="black", s=30, zorder=6)
    ax.axvline(E_m_peak_A, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.axvline(E_m_peak_B, color="black", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.annotate(
        r"$E_m^{\mathrm{peak}}(S_A)$",
        xy=(E_m_peak_A, E_M_peak),
        xytext=(E_m_peak_A + 0.3, E_M_peak - 1.0),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
    )
    ax.annotate(
        r"$E_m^{\mathrm{peak}}(S_B)$",
        xy=(E_m_peak_B, E_M_peak),
        xytext=(E_m_peak_B + 0.3, E_M_peak + 0.55),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
    )

    # Arrow illustrating higher d -> higher E_m(d) -> higher E_M^* (on the complements region).
    Em0 = 0.6
    Em1 = 4.0
    EM0 = solve_majority_effort_star(S=S_A, E_m=Em0, N_M=N_M, Lambda=Lambda)
    EM1 = solve_majority_effort_star(S=S_A, E_m=Em1, N_M=N_M, Lambda=Lambda)
    ax.annotate(
        r"higher $d$",
        xy=(Em1, EM1),
        xytext=(Em0, EM0),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        fontsize=10,
        ha="left",
        va="bottom",
    )

    ax.set_xlabel(r"Minority mobilization $E_m$", fontsize=12)
    ax.set_ylabel(r"Majority enforcement $E_M^*(S_k,E_m)$", fontsize=12)
    ax.set_title(r"Majority Reaction Functions in the Group Contest", fontsize=12)
    ax.set_xlim(0.0, float(E_m_grid.max()))
    ax.set_ylim(0.0, max(6.0, float(max(E_star_A.max(), E_star_B.max())) * 1.06))
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    out_path = Path(__file__).resolve().parent / "fig_reaction_functions.jpg"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

