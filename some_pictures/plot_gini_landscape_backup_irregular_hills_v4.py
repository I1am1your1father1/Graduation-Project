from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "backup_irregular_hills_v4"
SURFACE_ALPHA = 0.65
POSITIVE_CASE_RELIEF_SCALE = 1.5
SURFACE_GRID_SIZE = 72
CONTOUR_LEVEL_COUNT = 9
CONTOUR_LINEWIDTH = 2.15
SURFACE_CMAP = LinearSegmentedColormap.from_list(
    "irregular_hills_ygbp",
    [
        "#000724",
        "#156100",
        "#a39600",
    ],
)
CONTOUR_SAMPLE_POSITIONS = np.array([0.02, 0.10, 0.22, 0.36, 0.50, 0.64, 0.78, 0.90, 0.98])
PROJECTION_CONTOUR_COLORS = [
    tuple(np.clip(np.array(SURFACE_CMAP(position)[:3]) * 0.78, 0.0, 1.0)) + (1.0,)
    for position in CONTOUR_SAMPLE_POSITIONS
]


def rotated_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    theta: float,
) -> np.ndarray:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = x - cx
    dy = y - cy
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    return np.exp(-(xr**2 / sx + yr**2 / sy))


def smooth_trend(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Global descent trend toward the lower-right target corner."""

    dx = x - 0.92
    dy = y - 0.08

    trend = 0.88 * dx**2 + 0.68 * dy**2 + 0.22 * dx * dy
    trend -= 0.20 * np.exp(-((x - 0.80) ** 2 / 0.022 + (y - 0.18) ** 2 / 0.010))
    trend += 0.05 * np.exp(-((x - 0.16) ** 2 / 0.020 + (y - 0.82) ** 2 / 0.030))
    return trend


def rough_component(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Irregular hill-valley field with varying heights, depths, and orientations."""

    xw = x + 0.03 * np.sin(1.4 * np.pi * y + 0.2) - 0.02 * np.cos(1.1 * np.pi * x - 0.3)
    yw = (
        y
        + 0.025 * np.sin(1.1 * np.pi * x - 0.4)
        + 0.015 * np.cos(1.7 * np.pi * y + 0.5)
    )

    features = [
        (0.20, 0.13, 0.80, 0.010, 0.030, 0.35),
        (-0.18, 0.24, 0.66, 0.018, 0.010, -0.65),
        (0.12, 0.31, 0.83, 0.014, 0.020, 0.10),
        (-0.14, 0.37, 0.56, 0.012, 0.017, 0.55),
        (0.16, 0.45, 0.70, 0.020, 0.014, -0.35),
        (-0.11, 0.52, 0.48, 0.014, 0.012, 0.90),
        (0.18, 0.58, 0.62, 0.010, 0.024, -0.75),
        (-0.16, 0.66, 0.34, 0.015, 0.010, 0.20),
        (0.10, 0.72, 0.74, 0.016, 0.020, 0.60),
        (-0.13, 0.79, 0.52, 0.012, 0.016, -0.25),
        (0.08, 0.84, 0.31, 0.009, 0.012, 0.40),
        (-0.09, 0.61, 0.17, 0.014, 0.010, -0.55),
    ]

    terrain = np.zeros_like(xw)
    for amp, cx, cy, sx, sy, theta in features:
        terrain += amp * rotated_gaussian(xw, yw, cx, cy, sx, sy, theta)

    terrain += 0.06 * rotated_gaussian(xw, yw, 0.43, 0.31, 0.040, 0.012, -0.20)
    terrain -= 0.07 * rotated_gaussian(xw, yw, 0.76, 0.20, 0.030, 0.010, 0.15)
    terrain += 0.05 * rotated_gaussian(xw, yw, 0.30, 0.46, 0.016, 0.050, 0.95)

    return terrain


def gini_surrogate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A square-domain analogue of Gini: maximal in the interior, zero on the boundary."""

    return x * (1.0 - x) + y * (1.0 - y)


def rough_scale_from_gamma(gamma: float) -> float:
    base_scale = np.exp(0.95 * gamma)
    # Extra boost centered at gamma=0 so the original landscape shows clearer relief.
    original_boost = 1.0 + 2.6 * np.exp(-((gamma / 0.30) ** 2))
    return float(base_scale * original_boost)


def positive_gamma_convexification(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
) -> np.ndarray:
    positive_gamma = max(gamma, 0.0)
    if positive_gamma == 0.0:
        return np.zeros_like(x)

    # Lift the interior while keeping the boundary at zero so the positive-gamma
    # surface bulges upward without turning into a tilted plane.
    gini = gini_surrogate(x, y)
    bulge = 0.55 * gini + 0.72 * gini**2

    # Preserve the shared low-energy basin from the gamma=0 / gamma<0 cases so
    # the positive-gamma minimum region still aligns with the first two panels.
    basin = np.exp(-((x - 0.70) ** 2 / 0.020 + (y - 0.24) ** 2 / 0.010))
    basin_wide = np.exp(-((x - 0.70) ** 2 / 0.050 + (y - 0.24) ** 2 / 0.022))

    return positive_gamma * (bulge - 0.40 * basin - 0.20 * basin_wide)


def scheduled_energy(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    rough_scale = rough_scale_from_gamma(gamma)
    return (
        smooth_trend(x, y)
        + rough_scale * rough_component(x, y)
        + gamma * gini_surrogate(x, y)
        + positive_gamma_convexification(x, y, gamma)
    )


def scale_relief(surface: np.ndarray, relief_scale: float) -> np.ndarray:
    center = float(surface.mean())
    return center + relief_scale * (surface - center)


def sample_grid_indices(size: int, target_count: int) -> np.ndarray:
    if target_count >= size:
        return np.arange(size)
    return np.unique(np.round(np.linspace(0, size - 1, target_count)).astype(int))


def downsample_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    facecolors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_idx = sample_grid_indices(x.shape[0], SURFACE_GRID_SIZE)
    col_idx = sample_grid_indices(x.shape[1], SURFACE_GRID_SIZE)
    sample_idx = np.ix_(row_idx, col_idx)
    return x[sample_idx], y[sample_idx], z[sample_idx], facecolors[sample_idx]


def style_axes(ax: plt.Axes, z_floor: float, z_top: float) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(z_floor, z_top)
    ax.set_box_aspect((1.0, 1.0, 0.62))
    ax.view_init(elev=25, azim=-57)
    ax.grid(False)
    ax.set_axis_off()


def draw_case(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    shifted_z: np.ndarray,
    z_floor: float,
    z_top: float,
    contour_levels: np.ndarray,
    lightsource: LightSource,
) -> None:
    facecolors = np.array(
        lightsource.shade(
            shifted_z,
            cmap=SURFACE_CMAP,
            blend_mode="overlay",
            fraction=1.10,
        ),
        copy=True,
    )
    if facecolors.shape[-1] == 3:
        facecolors = np.dstack(
            [
                facecolors,
                np.full(facecolors.shape[:2], SURFACE_ALPHA, dtype=facecolors.dtype),
            ]
        )
    else:
        facecolors[..., 3] = SURFACE_ALPHA

    ax.contour(
        x,
        y,
        shifted_z,
        zdir="z",
        offset=z_floor,
        levels=contour_levels,
        colors=PROJECTION_CONTOUR_COLORS,
        linewidths=CONTOUR_LINEWIDTH,
        alpha=1.0,
        linestyles="solid",
    )

    surface_x, surface_y, surface_z, surface_facecolors = downsample_surface(
        x,
        y,
        shifted_z,
        facecolors,
    )
    surface = ax.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        facecolors=surface_facecolors,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        rcount=surface_x.shape[0],
        ccount=surface_x.shape[1],
        edgecolor="none",
    )
    surface.set_edgecolor("none")
    surface.set_linewidth(0.0)
    surface.set_antialiased(False)

    style_axes(ax, z_floor, z_top)


def make_figure() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = np.linspace(0.0, 1.0, 220)
    x, y = np.meshgrid(grid, grid)
    lightsource = LightSource(azdeg=315, altdeg=32)

    cases = [
        ("Original", 0.0, 1.0),
        (r"$\gamma < 0$", -1.35, 1.0),
        (r"$\gamma > 0$", 1.10, POSITIVE_CASE_RELIEF_SCALE),
    ]

    case_surfaces = [
        (title, gamma, scale_relief(scheduled_energy(x, y, gamma), relief_scale))
        for title, gamma, relief_scale in cases
    ]
    global_min = min(surface.min() for _, _, surface in case_surfaces)
    global_max = max(surface.max() for _, _, surface in case_surfaces)
    global_span = global_max - global_min
    z_floor = -0.18 * global_span
    z_top = 1.02 * global_span
    contour_levels = np.linspace(
        0.04 * global_span,
        0.96 * global_span,
        CONTOUR_LEVEL_COUNT,
    )

    output_paths: list[Path] = []

    triptych = plt.figure(
        figsize=(15.2, 5.0), facecolor="white", constrained_layout=True
    )
    for index, (title, gamma, raw_z) in enumerate(case_surfaces, start=1):
        shifted_z = raw_z - global_min
        ax = triptych.add_subplot(1, 3, index, projection="3d")
        draw_case(ax, x, y, shifted_z, z_floor, z_top, contour_levels, lightsource)

        single = plt.figure(
            figsize=(5.2, 4.6), facecolor="white", constrained_layout=True
        )
        single_ax = single.add_subplot(1, 1, 1, projection="3d")
        draw_case(
            single_ax,
            x,
            y,
            shifted_z,
            z_floor,
            z_top,
            contour_levels,
            lightsource,
        )
        safe_title = title.replace("$", "").replace("\\", "").replace(" ", "_")
        safe_title = safe_title.replace("<", "lt").replace(">", "gt")
        single_path = OUTPUT_DIR / f"{safe_title}.png"
        single.savefig(single_path, dpi=280, bbox_inches="tight")
        plt.close(single)
        output_paths.append(single_path)

    triptych_path = OUTPUT_DIR / "gini_landscape_triptych.png"
    triptych.savefig(triptych_path, dpi=280, bbox_inches="tight")
    plt.close(triptych)
    output_paths.append(triptych_path)
    return output_paths


def main() -> None:
    for path in make_figure():
        print(path, flush=True)


if __name__ == "__main__":
    main()
