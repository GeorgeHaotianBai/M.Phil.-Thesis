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
    # Parameters for Figure 3.
    N_M = 100
    Lambda = 1.0
    gamma = 5.0
    sigma = 1.0

    S_A = 1.0  # Hawk
    S_B = 0.2  # Dove

    d_grid = np.linspace(0.0, 2.0, 300)
    E_m_grid = gamma * d_grid**sigma

    E_star_A = np.array(
        [solve_majority_effort_star(S=S_A, E_m=float(Em), N_M=N_M, Lambda=Lambda) for Em in E_m_grid]
    )
    E_star_B = np.array(
        [solve_majority_effort_star(S=S_B, E_m=float(Em), N_M=N_M, Lambda=Lambda) for Em in E_m_grid]
    )

    RD_A = E_star_A + E_m_grid
    RD_B = E_star_B + E_m_grid

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

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(d_grid, RD_A, color="black", linewidth=2.0, label=r"Hawk ($S_A$ high)")
    ax.plot(d_grid, RD_B, color="black", linewidth=2.0, linestyle="--", label=r"Dove ($S_B$ low)")

    ax.set_xlabel(r"Symbolic policy intensity $d$", fontsize=12)
    ax.set_ylabel(r"Rent dissipation $RD(d,S_k)$", fontsize=12)
    ax.set_title(r"Rent Dissipation Induced by Symbolic Policy", fontsize=12)
    ax.set_xlim(0.0, float(d_grid.max()))
    ax.set_ylim(0.0, float(max(RD_A.max(), RD_B.max()) * 1.05))
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    out_path = Path(__file__).resolve().parent / "fig_rent_dissipation.jpg"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

