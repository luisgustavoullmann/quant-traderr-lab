"""
BrownianBridge_Static_Pipeline.py
====================================
Single 1920x1080 hero image: pinned-endpoint Brownian Bridge bundle
with the analytic variance envelope sigma^2 * t*(T-t)/T.

Layout (landscape):
    Top  : title strip
    Left : 3D scene -- 220 path bundle, yellow drift line, magenta
           +/- 2 sigma envelope curves, bright sphere markers at start
           and end points
    Right: 2D variance envelope panel, HUD with current parameters,
           equation strip, applications callout
"""

import os, sys, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
from BrownianBridge_Reel_Pipeline import (  # noqa: E402
    THEME, simulate, _hex_rgba, CONFIG as REEL_CONFIG,
)


CONFIG = {
    "WIDTH":  1920, "HEIGHT": 1080, "DPI": 100,
    "OUTPUT_FILE": os.path.join(BASE_DIR, "BrownianBridge_Static.png"),
}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def render_static(t_axis, paths, mean_line, std_env, out_path):
    n_p = paths.shape[1]
    a   = REEL_CONFIG["S_A"]
    b   = REEL_CONFIG["S_B"]
    sig = REEL_CONFIG["SIGMA"]

    fig = plt.figure(figsize=(CONFIG["WIDTH"]/CONFIG["DPI"],
                              CONFIG["HEIGHT"]/CONFIG["DPI"]),
                     dpi=CONFIG["DPI"], facecolor=THEME["BG"])

    # ---------- TITLE ----------
    fig.text(0.025, 0.945, "BROWNIAN BRIDGE",
             fontsize=22, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.025, 0.910,
             "diffusion pinned at both endpoints — the middle is pure information",
             fontsize=12, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.975, 0.945, f"σ = {sig:.0f}", ha="right",
             fontsize=11, color=THEME["MAGENTA"], family=THEME["FONT"])

    # ---------- 3D HERO (LEFT) ----------
    ax3d = fig.add_axes([-0.02, 0.05, 0.66, 0.85],
                        projection="3d", facecolor=THEME["BG"])

    rng = np.random.default_rng(7)
    y_spread = rng.uniform(-0.4, 0.4, n_p)

    for k in range(n_p):
        p = paths[:, k]
        dev = p[len(p) // 2] - mean_line[len(p) // 2]   # mid-time deviation
        if dev > std_env.max() * 0.4:
            color_hex = THEME["GREEN"]
        elif dev < -std_env.max() * 0.4:
            color_hex = THEME["RED"]
        else:
            color_hex = THEME["CYAN"]
        col = list(_hex_rgba(color_hex, 0.16))
        y = np.full_like(t_axis, y_spread[k])
        pts = np.array([t_axis, y, p]).T.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ax3d.add_collection3d(
            Line3DCollection(segs, colors=[col] * len(segs), linewidths=0.55))

    # Mean drift line
    ax3d.plot(t_axis, np.zeros_like(t_axis), mean_line,
              color=THEME["YELLOW"], linewidth=2.2, alpha=0.95)

    # Envelope curves
    upper = mean_line + 2.0 * std_env
    lower = mean_line - 2.0 * std_env
    ax3d.plot(t_axis, np.zeros_like(t_axis), upper,
              color=THEME["MAGENTA"], linewidth=1.0, alpha=0.65, linestyle="--")
    ax3d.plot(t_axis, np.zeros_like(t_axis), lower,
              color=THEME["MAGENTA"], linewidth=1.0, alpha=0.65, linestyle="--")

    # Endpoint markers
    for tt, ss in [(t_axis[0], a), (t_axis[-1], b)]:
        for s, alpha in [(540, 0.10), (260, 0.25), (130, 0.6), (55, 1.0)]:
            ax3d.scatter([tt], [0], [ss], s=s, color=THEME["YELLOW"],
                         alpha=alpha, edgecolors="none", depthshade=False)

    # Floating labels
    ax3d.text(t_axis[0] + 0.02, 0, a + 1.5,
              f"start  ${a:.0f}",
              color=THEME["YELLOW"], fontsize=11, fontweight="bold")
    ax3d.text(t_axis[-1] - 0.20, 0, b + 1.5,
              f"target  ${b:.0f}",
              color=THEME["YELLOW"], fontsize=11, fontweight="bold")
    ax3d.text(t_axis[len(t_axis) // 2] - 0.04, 0, upper.max() + 1.0,
              r"$\pm 2\sigma$ envelope",
              color=THEME["MAGENTA"], fontsize=10)

    p_max = max(upper.max(), b) + 4
    p_min = min(lower.min(), a) - 4
    ax3d.view_init(elev=15, azim=-58)
    ax3d.set_xlim(t_axis[0], t_axis[-1])
    ax3d.set_ylim(-0.8, 0.8)
    ax3d.set_zlim(p_min, p_max)
    ax3d.set_box_aspect((1.7, 0.5, 1.5))
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_alpha(0); pane.set_edgecolor((0, 0, 0, 0))
    ax3d.grid(False)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])

    # ---------- RIGHT: variance envelope ----------
    ax_v = fig.add_axes([0.66, 0.55, 0.32, 0.30], facecolor=THEME["PANEL"])
    ax_v.fill_between(t_axis, mean_line - 2 * std_env, mean_line + 2 * std_env,
                      color=THEME["MAGENTA"], alpha=0.20, edgecolor="none")
    ax_v.plot(t_axis, mean_line, color=THEME["YELLOW"], linewidth=1.6)
    ax_v.set_xlim(t_axis[0], t_axis[-1])
    ax_v.set_ylim(p_min, p_max)
    ax_v.set_yticks([])
    ax_v.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_v.set_xticklabels(["t=0", "T/4", "T/2", "3T/4", "T"])
    ax_v.tick_params(colors=THEME["TEXT_DIM"], labelsize=9)
    for s in ax_v.spines.values(): s.set_color("#1f1f1f"); s.set_linewidth(0.5)
    ax_v.set_title(r"variance envelope  $\sigma^{2}\, t(T-t)/T$",
                   color=THEME["TEXT_DIM"], fontsize=10, loc="left", pad=4)

    # ---------- RIGHT HUD ----------
    fig.text(0.67, 0.45, "max σ_t  (at t = T/2)",
             fontsize=11, color=THEME["TEXT_DIM"])
    fig.text(0.67, 0.395, f"{std_env.max():.2f}",
             fontsize=36, color=THEME["MAGENTA"], fontweight="bold")

    fig.text(0.67, 0.32, "endpoints", fontsize=10, color=THEME["TEXT_DIM"])
    fig.text(0.67, 0.272,
             f"${a:.0f}  →  ${b:.0f}",
             fontsize=22, color=THEME["YELLOW"], fontweight="bold")

    fig.text(0.67, 0.21,
             r"$X_{t}=a+(b-a)\,t/T + \sigma\,B_{t}$",
             fontsize=14, color=THEME["TEXT"])
    fig.text(0.67, 0.16,
             r"$\mathrm{Var}(X_{t})=\sigma^{2}\, t(T-t)/T$",
             fontsize=14, color=THEME["MAGENTA"])

    fig.text(0.025, 0.040,
             "USES   conditional Monte Carlo · VWAP execution · option pin risk",
             fontsize=11, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.025, 0.017,
             "the only diffusion that knows where it ends.",
             fontsize=10, color=THEME["TEXT_DIM"], style="italic")
    fig.text(0.975, 0.020, "@quant.traderr",
             ha="right", fontsize=11, color=THEME["TEXT_DIM"],
             alpha=0.65, family=THEME["FONT"])

    fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Saved: {out_path}")


def main():
    t0 = time.time()
    log("=== BROWNIAN BRIDGE STATIC ===")
    t_axis, paths, mean_line, std_env = simulate()
    log(f"Paths terminal range: [{paths[-1].min():.2f}, {paths[-1].max():.2f}]")
    log(f"max sigma: {std_env.max():.2f}")
    render_static(t_axis, paths, mean_line, std_env, CONFIG["OUTPUT_FILE"])
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
