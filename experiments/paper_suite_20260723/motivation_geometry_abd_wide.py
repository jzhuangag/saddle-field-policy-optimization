"""Generate a wide three-panel geometric-motivation figure.

The script redraws the three panels in one shared Matplotlib canvas so that
their axes, typography, captions, and spacing are aligned in the vector PDF.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "pdf"
OUT_PDF = OUT_DIR / "fig_motivation_geometry_abd_wide.pdf"
OUT_PNG = OUT_DIR / "fig_motivation_geometry_abd_wide.png"

FONT_SIZE = 7.0
FIELD_COLOR = "#C73E1D"
CURVATURE_COLOR = "#1F5A94"
GDA_COLOR = "#C73E1D"
PPM_COLOR = "#2A8C4A"
EGM_COLOR = "#2D6FB7"
BALANCED_COLOR = "#1A1A1A"
FIXED_COLOR = "#7A4FA3"
WEAK_COLOR = "#D97706"
OVER_COLOR = "#008C8C"
J0 = np.array([[0.0, 1.0], [-1.0, 0.0]])

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": FONT_SIZE,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman",
        "axes.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def field(z):
    """Return the saddle field F(z)=J0 z for L(x,y)=-xy."""
    return J0 @ z


def curvature(z):
    """Return the curvature direction G(z)=J0 F(z)=-z."""
    return J0 @ field(z)


def classical_trajectory(z0, step, iterations, method):
    """Compute a GDA, PPM, or EGM trajectory."""
    z = z0.copy()
    path = [z.copy()]
    for _ in range(iterations):
        if method == "GDA":
            z = z - step * field(z)
        elif method == "PPM":
            z = np.linalg.solve(np.eye(2) + step * J0, z)
        elif method == "EGM":
            extrapolated = z - step * field(z)
            z = z - step * field(extrapolated)
        else:
            raise ValueError(f"Unknown method: {method}")
        path.append(z.copy())
    return np.asarray(path)


def two_direction_trajectory(z0, beta, gamma, iterations):
    """Iterate z_+=z-beta F(z)+gamma G(z)."""
    z = z0.copy()
    path = [z.copy()]
    for _ in range(iterations):
        z = z - beta * field(z) + gamma * curvature(z)
        path.append(z.copy())
    return np.asarray(path)


def add_energy_contours(ax, limit, levels):
    """Draw the shared field-energy background phi(z)=||F(z)||^2/2."""
    grid = np.linspace(-limit, limit, 300)
    x_grid, y_grid = np.meshgrid(grid, grid)
    phi = 0.5 * (x_grid**2 + y_grid**2)
    contour_levels = np.linspace(0.0, float(phi.max()), levels)
    ax.contourf(
        x_grid,
        y_grid,
        phi,
        levels=contour_levels,
        cmap="Blues",
        alpha=0.66,
    )
    ax.contour(
        x_grid,
        y_grid,
        phi,
        levels=contour_levels[1::2],
        colors="0.30",
        linewidths=0.28,
        alpha=0.42,
    )


def style_axis(ax, limit):
    """Apply identical square-axis styling to every panel."""
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.tick_params(width=0.55, length=2.3, pad=1.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)


def add_stationary_point(ax):
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=56,
        color="#F2C14E",
        edgecolor="black",
        linewidth=0.5,
        zorder=8,
    )


def add_path_arrow(ax, path, color, index, width=0.9):
    """Place one arrowhead on a trajectory without using curve markers."""
    ax.annotate(
        "",
        xy=path[index],
        xytext=path[index - 1],
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": width,
            "mutation_scale": 7.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=7,
    )


def add_panel_caption(ax, text):
    ax.text(
        0.5,
        -0.205,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        clip_on=False,
    )


def compact_legend(ax, handles, location, columns=1, font_size=FONT_SIZE):
    return ax.legend(
        handles=handles,
        loc=location,
        ncol=columns,
        fontsize=font_size,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.86,
        handlelength=1.55,
        handletextpad=0.38,
        borderpad=0.22,
        borderaxespad=0.25,
        labelspacing=0.18,
        columnspacing=0.55,
    )


def draw_panel_a(ax):
    """Draw line-only GDA, PPM, and EGM trajectories."""
    limit = 2.05
    z0 = np.array([0.75, 0.30])
    step = 0.25
    styles = {
        "GDA": (GDA_COLOR, "-"),
        "PPM": (PPM_COLOR, "--"),
        "EGM": (EGM_COLOR, "-."),
    }
    add_energy_contours(ax, limit, levels=15)
    final_norms = {}
    for method, (color, linestyle) in styles.items():
        path = classical_trajectory(z0, step, 28, method)
        final_norms[method] = float(np.linalg.norm(path[-1]))
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=1.10,
            marker=None,
            zorder=5,
        )
        add_path_arrow(ax, path, color, len(path) // 2)
    assert final_norms["GDA"] > np.linalg.norm(z0)
    assert final_norms["PPM"] < np.linalg.norm(z0)
    assert final_norms["EGM"] < np.linalg.norm(z0)

    ax.scatter([z0[0]], [z0[1]], color="black", s=9, zorder=8)
    ax.annotate(r"$z_0$", xy=z0, xytext=(3, 3), textcoords="offset points")
    add_stationary_point(ax)
    style_axis(ax, limit)
    handles = [
        Line2D([0], [0], color=GDA_COLOR, linestyle="-", linewidth=1.10, marker=None, label="GDA"),
        Line2D([0], [0], color=PPM_COLOR, linestyle="--", linewidth=1.10, marker=None, label="PPM"),
        Line2D([0], [0], color=EGM_COLOR, linestyle="-.", linewidth=1.10, marker=None, label="EGM"),
    ]
    compact_legend(ax, handles, location="lower left")
    add_panel_caption(ax, "(a) Bilinear trajectories")


def draw_direction(ax, z, vector, color, length):
    vector = vector / np.linalg.norm(vector)
    ax.quiver(
        z[0],
        z[1],
        length * vector[0],
        length * vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        width=0.010,
        headwidth=4.4,
        headlength=5.0,
        headaxislength=4.4,
        zorder=6,
    )


def draw_panel_b(ax):
    """Draw the rotational field and inward curvature directions."""
    limit = 1.55
    add_energy_contours(ax, limit, levels=17)
    for radius in (0.68, 1.18):
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            z = radius * np.array([np.cos(angle), np.sin(angle)])
            draw_direction(ax, z, -field(z), FIELD_COLOR, length=0.27)
            draw_direction(ax, z, curvature(z), CURVATURE_COLOR, length=0.27)
            ax.scatter([z[0]], [z[1]], s=6, color="black", zorder=7)
    add_stationary_point(ax)
    style_axis(ax, limit)
    handles = [
        Line2D(
            [0],
            [0],
            color=FIELD_COLOR,
            linewidth=1.10,
            marker=">",
            markersize=3.6,
            label=r"$-F$: rotational",
        ),
        Line2D(
            [0],
            [0],
            color=CURVATURE_COLOR,
            linewidth=1.10,
            marker=">",
            markersize=3.6,
            label=r"$+G$: inward",
        ),
    ]
    compact_legend(ax, handles, location="upper left")
    add_panel_caption(ax, r"(b) Roles of $-F$ and $+G$")


def draw_panel_c(ax):
    """Compare four step-size ratios in the two-direction update."""
    limit = 2.45
    z0 = np.array([0.90, 0.35])
    beta = 0.25
    configurations = [
        {
            "label": r"balanced: $\gamma/\beta=0.80$",
            "gamma": 0.20,
            "iterations": 18,
            "color": BALANCED_COLOR,
            "linestyle": "-",
        },
        {
            "label": r"fixed: $\gamma=\beta^2$",
            "gamma": beta**2,
            "iterations": 18,
            "color": FIXED_COLOR,
            "linestyle": "-.",
        },
        {
            "label": r"weak: $\gamma/\beta=0.04$",
            "gamma": 0.01,
            "iterations": 18,
            "color": WEAK_COLOR,
            "linestyle": "--",
        },
        {
            "label": r"over-corrected: $\gamma/\beta=7.60$",
            "gamma": 1.90,
            "iterations": 14,
            "color": OVER_COLOR,
            "linestyle": ":",
        },
    ]
    add_energy_contours(ax, limit, levels=13)
    handles = []
    for configuration in configurations:
        path = two_direction_trajectory(
            z0,
            beta,
            configuration["gamma"],
            configuration["iterations"],
        )
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=configuration["color"],
            linestyle=configuration["linestyle"],
            linewidth=1.10,
            marker=None,
            zorder=6,
        )
        add_path_arrow(
            ax,
            path,
            configuration["color"],
            min(10, len(path) - 1),
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=configuration["color"],
                linestyle=configuration["linestyle"],
                linewidth=1.10,
                marker=None,
                label=configuration["label"],
            )
        )
    ax.scatter([z0[0]], [z0[1]], color="black", s=9, zorder=8)
    ax.annotate(r"$z_0$", xy=z0, xytext=(3, 3), textcoords="offset points")
    add_stationary_point(ax)
    style_axis(ax, limit)
    compact_legend(ax, handles, location="upper left", columns=2, font_size=5.4)
    add_panel_caption(ax, "(c) Multiple two-direction combinations")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.985,
        bottom=0.185,
        wspace=0.31,
    )
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.025, dpi=360)
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
