"""Generate the analytic three-panel mechanism figure used in the manuscript.

Every panel is computed from a closed-form normal linear saddle field.  The
figure is an exact geometric illustration, not an empirical benchmark.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = ROOT / "output" / "pdf" / "fig_motivation_geometry.pdf"
OUT_PNG = ROOT / "output" / "pdf" / "fig_motivation_geometry.png"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7.8,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.0,
        "legend.fontsize": 6.4,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def bilinear_field(z):
    """Field for the paper's max--min convention L(x,y)=-xy."""
    return np.array([z[1], -z[0]])


def bilinear_iterates(z0, step, iterations, method):
    """Exact GDA, EGM, or PPM iterates for F(z)=J_0 z."""
    jacobian = np.array([[0.0, 1.0], [-1.0, 0.0]])
    z = z0.copy()
    trajectory = [z.copy()]
    for _ in range(iterations):
        if method == "GDA":
            z = z - step * bilinear_field(z)
        elif method == "EGM":
            extrapolated = z - step * bilinear_field(z)
            z = z - step * bilinear_field(extrapolated)
        elif method == "PPM":
            z = np.linalg.solve(
                np.eye(2) + step * jacobian,
                z,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        trajectory.append(z.copy())
    return np.asarray(trajectory)


def add_direction_arrow(ax, trajectory, color):
    """Add one direction arrow without obscuring the discrete markers."""
    arrow_index = max(1, len(trajectory) // 2)
    ax.annotate(
        "",
        xy=trajectory[arrow_index],
        xytext=trajectory[arrow_index - 1],
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 1.0,
            "mutation_scale": 7,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )


def panel_bilinear_trajectories(ax):
    grid = np.linspace(-2.05, 2.05, 280)
    x_grid, y_grid = np.meshgrid(grid, grid)
    phi = 0.5 * (x_grid**2 + y_grid**2)
    levels = np.linspace(0.0, float(phi.max()), 18)
    ax.contourf(
        x_grid,
        y_grid,
        phi,
        levels=levels,
        cmap="Blues",
        alpha=0.78,
    )
    ax.contour(
        x_grid,
        y_grid,
        phi,
        levels=levels[1::2],
        colors="k",
        linewidths=0.23,
        alpha=0.38,
    )

    z0 = np.array([0.75, 0.30])
    step = 0.25
    iterations = 28
    styles = {
        "GDA": ("#d62728", "-", "o"),
        "PPM": ("#2ca02c", "--", "s"),
        "EGM": ("#1f77b4", "-.", "^"),
    }
    final_norms = {}
    for method, (color, linestyle, marker) in styles.items():
        trajectory = bilinear_iterates(z0, step, iterations, method)
        final_norms[method] = float(np.linalg.norm(trajectory[-1]))
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=1.35,
            marker=marker,
            markersize=2.2,
            markevery=4,
            label=method,
            zorder=4,
        )
        add_direction_arrow(ax, trajectory, color)

    # These closed-form sentinels make the panel fail loudly if a sign or
    # implementation change reverses the stated stability mechanism.
    assert final_norms["GDA"] > float(np.linalg.norm(z0))
    assert final_norms["PPM"] < float(np.linalg.norm(z0))
    assert final_norms["EGM"] < float(np.linalg.norm(z0))

    ax.scatter(
        [z0[0]],
        [z0[1]],
        color="black",
        s=16,
        zorder=7,
    )
    ax.annotate(
        r"$\mathbf{z}_0$",
        xy=z0,
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=6.7,
    )
    ax.plot(
        0.0,
        0.0,
        marker="*",
        color="gold",
        markersize=9,
        markeredgecolor="k",
        markeredgewidth=0.55,
        zorder=7,
    )
    ax.set(
        xlim=(-2.05, 2.05),
        ylim=(-2.05, 2.05),
        xlabel=r"$x$",
        ylabel=r"$y$",
        title=r"(a) Bilinear trajectories, $L=-xy$",
    )
    ax.legend(loc="lower left", framealpha=0.92, handlelength=1.6)
    ax.set_box_aspect(1)


def panel_pure_rotation(ax):
    grid = np.linspace(-1.55, 1.55, 260)
    x_grid, y_grid = np.meshgrid(grid, grid)
    phi = 0.5 * (x_grid**2 + y_grid**2)
    levels = np.linspace(0.0, float(phi.max()), 17)
    contour = ax.contourf(
        x_grid, y_grid, phi, levels=levels, cmap="Blues", alpha=0.82
    )
    ax.contour(
        x_grid,
        y_grid,
        phi,
        levels=levels[1::2],
        colors="k",
        linewidths=0.25,
        alpha=0.42,
    )

    arrow_length = 0.25
    for radius in (0.48, 0.93, 1.30):
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            z = radius * np.array([np.cos(angle), np.sin(angle)])
            field = np.array([z[1], -z[0]])
            curvature = -z
            minus_field = -field / np.linalg.norm(field)
            plus_curvature = curvature / np.linalg.norm(curvature)
            ax.arrow(
                z[0],
                z[1],
                arrow_length * minus_field[0],
                arrow_length * minus_field[1],
                head_width=0.045,
                head_length=0.065,
                color="#d62728",
                linewidth=0.9,
                length_includes_head=True,
                zorder=4,
            )
            ax.arrow(
                z[0],
                z[1],
                arrow_length * plus_curvature[0],
                arrow_length * plus_curvature[1],
                head_width=0.045,
                head_length=0.065,
                color="#1f4e79",
                linewidth=0.9,
                length_includes_head=True,
                zorder=4,
            )
            ax.plot(z[0], z[1], "ko", markersize=1.8, zorder=5)

    ax.plot(
        0.0,
        0.0,
        marker="*",
        color="gold",
        markersize=11,
        markeredgecolor="k",
        markeredgewidth=0.6,
        zorder=6,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#d62728",
                lw=1.6,
                label=r"$-\mathbf{F}\perp-\nabla\phi$",
            ),
            Line2D(
                [0],
                [0],
                color="#1f4e79",
                lw=1.6,
                label=r"$+\mathbf{G}=-\nabla\phi$",
            ),
        ],
        loc="upper right",
        framealpha=0.92,
        handlelength=1.7,
    )
    ax.set(
        xlim=(-1.55, 1.55),
        ylim=(-1.55, 1.55),
        xlabel=r"$x$",
        ylabel=r"$y$",
        title=r"(b) Directions on $\phi=\frac{1}{2}\|\mathbf{F}\|^2$",
    )
    ax.set_box_aspect(1)
    return contour


def normal_field_coefficients(mu, sigma, z):
    j0 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    matrix = mu * np.eye(2) + sigma * j0
    field = matrix @ z
    curvature = matrix @ field
    hessian = matrix.T @ matrix
    grad_phi = hessian @ z
    e_coef = float(grad_phi @ field)
    d_coef = float(-grad_phi @ curvature)
    c_coef = float(field @ hessian @ field)
    b_coef = float(field @ hessian @ curvature)
    a_coef = float(curvature @ hessian @ curvature)
    return e_coef, d_coef, c_coef, b_coef, a_coef


def panel_exact_qp(ax):
    mu, sigma = 0.35, 0.70
    z = np.array([1.0, -0.65])
    e_coef, d_coef, c_coef, b_coef, a_coef = normal_field_coefficients(
        mu, sigma, z
    )
    hessian_2d = np.array([[c_coef, -b_coef], [-b_coef, a_coef]])
    beta_star, gamma_star = np.linalg.solve(
        hessian_2d, np.array([e_coef, d_coef])
    )

    def q_model(beta, gamma):
        return (
            -beta * e_coef
            - gamma * d_coef
            + 0.5 * c_coef * beta**2
            - b_coef * beta * gamma
            + 0.5 * a_coef * gamma**2
        )

    beta_max = 1.35 * beta_star
    gamma_max = 1.25 * gamma_star
    beta = np.linspace(0.0, beta_max, 260)
    gamma = np.linspace(0.0, gamma_max, 260)
    beta_grid, gamma_grid = np.meshgrid(beta, gamma)
    q_values = q_model(beta_grid, gamma_grid)
    levels = np.linspace(float(q_values.min()), 0.0, 18)
    contour = ax.contourf(
        beta_grid,
        gamma_grid,
        q_values,
        levels=levels,
        cmap="RdYlBu_r",
        alpha=0.88,
    )
    ax.contour(
        beta_grid,
        gamma_grid,
        q_values,
        levels=levels,
        colors="k",
        linewidths=0.22,
        alpha=0.32,
    )

    beta_field = max(e_coef / c_coef, 0.0)
    gamma_curvature = max(d_coef / a_coef, 0.0)
    s_grid = np.linspace(0.0, min(beta_max, np.sqrt(gamma_max)), 600)
    q_egm = q_model(s_grid, s_grid**2)
    s_best = float(s_grid[int(np.argmin(q_egm))])
    q_star = float(q_model(beta_star, gamma_star))
    q_field = float(q_model(beta_field, 0.0))
    q_curvature = float(q_model(0.0, gamma_curvature))
    q_fixed = float(q_model(s_best, s_best**2))
    assert np.all(np.linalg.eigvalsh(hessian_2d) > 0.0)
    assert beta_star > 0.0 and gamma_star > 0.0
    assert q_star < min(q_field, q_curvature, q_fixed) - 1e-10

    ax.plot(
        [0.0, beta_max],
        [0.0, 0.0],
        color="#1f77b4",
        lw=1.5,
        label=r"pure-$\mathbf{F}$",
    )
    ax.plot(
        [0.0, 0.0],
        [0.0, gamma_max],
        color="#2ca02c",
        lw=1.5,
        label=r"pure-$\mathbf{G}$",
    )
    ax.plot(
        s_grid,
        s_grid**2,
        color="#9467bd",
        lw=1.5,
        ls="--",
        label=r"fixed $(s,s^2)$",
    )
    ax.scatter(
        [beta_field, 0.0, s_best],
        [0.0, gamma_curvature, s_best**2],
        c=["#1f77b4", "#2ca02c", "#9467bd"],
        s=29,
        edgecolor="k",
        linewidth=0.45,
        zorder=5,
    )
    ax.scatter(
        [beta_star],
        [gamma_star],
        marker="*",
        s=105,
        color="#d62728",
        edgecolor="k",
        linewidth=0.65,
        zorder=6,
        label=r"adaptive $(\beta^\star,\gamma^\star)$",
    )
    ax.set(
        xlim=(0.0, beta_max),
        ylim=(0.0, gamma_max),
        xlabel=r"field step $\beta$",
        ylabel=r"curvature step $\gamma$",
        title=r"(c) QP model of $\Delta\mathcal{V}$, $\sigma/\mu=2$",
    )
    ax.legend(loc="upper left", framealpha=0.92, handlelength=1.7)
    ax.set_box_aspect(1)
    return contour


def panel_alignment_threshold(ax):
    ratio = np.linspace(0.0, 2.5, 600)
    field_alignment = 1.0 / np.sqrt(1.0 + ratio**2)
    curvature_alignment = (ratio**2 - 1.0) / (ratio**2 + 1.0)

    ax.axhline(0.0, color="0.35", linewidth=0.7)
    ax.axvline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax.axvspan(1.0, 2.5, color="#2ca02c", alpha=0.10)
    ax.plot(
        ratio,
        field_alignment,
        color="#d62728",
        lw=1.7,
        label=r"$\cos(-\mathbf{F},-\nabla\phi)$",
    )
    ax.plot(
        ratio,
        curvature_alignment,
        color="#1f4e79",
        lw=1.7,
        label=r"$\cos(+\mathbf{G},-\nabla\phi)$",
    )
    ax.text(
        1.04,
        -0.92,
        r"$+\mathbf{G}$ is descending",
        color="#216b2b",
        fontsize=7.2,
        va="bottom",
    )
    ax.annotate(
        r"$|\sigma|=|\mu|$",
        xy=(1.0, 0.0),
        xytext=(1.18, 0.23),
        arrowprops={"arrowstyle": "->", "lw": 0.7},
        fontsize=7.2,
    )
    ax.set(
        xlim=(0.0, 2.5),
        ylim=(-1.05, 1.05),
        xlabel=r"rotation ratio $|\sigma|/|\mu|$",
        ylabel=r"cosine with $-\nabla\phi$",
        title=r"(c) Normal-field alignment threshold",
    )
    ax.legend(loc="center right", framealpha=0.92, handlelength=1.7)
    ax.grid(alpha=0.25)


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    # A full-width row keeps all three mechanism panels square and at the same
    # physical size.  The trajectory panel reproduces the information in the
    # legacy raster figure using exact vector graphics.
    fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.45))
    panel_bilinear_trajectories(axes[0])
    panel_pure_rotation(axes[1])
    panel_exact_qp(axes[2])
    fig.subplots_adjust(
        left=0.055,
        right=0.988,
        top=0.91,
        bottom=0.19,
        wspace=0.32,
    )
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.035, dpi=240)
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
