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
    # Baseline parameters for Figure 1.
    N_M = 100
    Lambda = 1.0
    E_m = 5.0

    S_grid = np.linspace(0.05, 1.0, 240)
    E_star = np.array(
        [solve_majority_effort_star(S=float(S), E_m=E_m, N_M=N_M, Lambda=Lambda) for S in S_grid]
    )

    # Illustrative "Dove" and "Hawk" technologies.
    S_B = 0.2
    S_A = 1.0
    E_B = solve_majority_effort_star(S=S_B, E_m=E_m, N_M=N_M, Lambda=Lambda)
    E_A = solve_majority_effort_star(S=S_A, E_m=E_m, N_M=N_M, Lambda=Lambda)

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

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(S_grid, E_star, color="black", linewidth=2.0)

    ax.scatter([S_B, S_A], [E_B, E_A], color="black", s=30, zorder=5)
    ax.annotate(
        "Dove ($S_B$)",
        xy=(S_B, E_B),
        xytext=(S_B + 0.06, E_B - 0.75),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
    )
    ax.annotate(
        "Hawk ($S_A$)",
        xy=(S_A, E_A),
        xytext=(S_A - 0.28, E_A - 0.7),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
        fontsize=10,
        ha="left",
    )

    ax.set_xlabel(r"$S_k$ (security competence)", fontsize=12)
    ax.set_ylabel(r"$E_M^*(S_k,E_m)$", fontsize=12)
    ax.set_title(r"Equilibrium Majority Mobilization vs. State Capacity", fontsize=12)
    ax.set_xlim(float(S_grid.min()), float(S_grid.max()))
    ax.set_ylim(0.0, max(6.0, float(E_star.max()) * 1.05))

    out_path = Path(__file__).resolve().parent / "fig_Sk_statics.jpg"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

