"""
Strategy_Overfitting_Landscape_Static_Pipeline.py
=================================================
STATIC hero-image pipeline for "The Overfitting Landscape".

Standalone: DATA -> 2-D backtest grid (lookback x z-threshold) of BTC
mean-reversion Sharpe -> render the full rugged terrain at a fixed camera, with
the global-max peak flagged as the overfit trap.

    python Strategy_Overfitting_Landscape_Static_Pipeline.py
    -> Overfitting_Landscape_Static.png  (5760x3240)
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TICKER": "BTC-USD", "PERIOD": "3y",
    "LB_MIN": 5, "LB_MAX": 60, "THR_MIN": 0.3, "THR_MAX": 3.0, "THR_N": 56,
    "W": 1920, "H": 1080, "DPI": 300, "ELEV": 34, "AZIM": -46,
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": "Arial",
}
CMAP = LinearSegmentedColormap.from_list(
    "terrain", ["#0033aa", "#0066ff", "#00f2ff", "#00ff8c", "#ffd400",
                "#ff9500", "#ff3050"], N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def fetch_prices():
    c = CONFIG
    log(f"[Data] Fetching {c['TICKER']}...")
    try:
        import yfinance as yf
        data = yf.download(c["TICKER"], period=c["PERIOD"], interval="1d",
                           progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(c["TICKER"], axis=1, level=1)
        return data["Close"].values.flatten()
    except Exception as e:
        log(f"[Data] yfinance failed ({e}); synthetic fallback")
        rng = np.random.default_rng(3)
        return 30000 * np.exp(np.cumsum(rng.normal(0, 0.03, 760)))


def sharpe_grid(prices):
    c = CONFIG
    ret = np.diff(np.log(prices))
    lbs = np.arange(c["LB_MIN"], c["LB_MAX"] + 1)
    thrs = np.linspace(c["THR_MIN"], c["THR_MAX"], c["THR_N"])
    Z = np.zeros((len(lbs), len(thrs)))
    s = pd.Series(prices)
    for i, lb in enumerate(lbs):
        z = ((s - s.rolling(lb).mean()) / (s.rolling(lb).std() + 1e-9)).values[1:]
        for j, thr in enumerate(thrs):
            pos = np.where(z < -thr, 1.0, np.where(z > thr, -1.0, 0.0))
            pnl = (pos[:-1] * ret[1:])
            pnl = pnl[~np.isnan(pnl)]
            sd = pnl.std()
            Z[i, j] = (pnl.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0
    return lbs, thrs, Z


def render_static(out):
    c = CONFIG
    prices = fetch_prices()
    lbs, thrs, Z = sharpe_grid(prices)
    P1, P2 = np.meshgrid(lbs, thrs, indexing="ij")
    pi, pj = np.unravel_index(np.argmax(Z), Z.shape)
    zmin, zmax = float(min(Z.min(), 0)), float(Z.max())
    log(f"grid {Z.shape}  Sharpe [{Z.min():.2f}, {Z.max():.2f}]  peak at lb={lbs[pi]}, thr={thrs[pj]:.2f}")

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"],
                     facecolor=THEME["BG"])
    fig.text(0.5, 0.945, "THE OVERFITTING LANDSCAPE", ha="center", fontsize=27,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.905,
             "in-sample Sharpe over every parameter pair  ·  BTC mean-reversion",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([0.02, -0.02, 1.0, 0.94], projection="3d",
                      facecolor=THEME["BG"])
    norm = (Z - zmin) / (zmax - zmin + 1e-9)
    ax.plot_surface(P1, P2, Z, facecolors=CMAP(norm), rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, zorder=1)
    ax.plot_wireframe(P1, P2, Z, color=(1, 1, 1, 0.05), linewidth=0.3,
                      rstride=2, cstride=2, zorder=2)
    ax.scatter([lbs[pi]], [thrs[pj]], [Z[pi, pj]], s=80, color="white",
               edgecolors=THEME["RED"], linewidths=1.6, zorder=12)
    ax.plot([lbs[pi], lbs[pi]], [thrs[pj], thrs[pj]], [zmin, Z[pi, pj]],
            color=THEME["RED"], linewidth=1.2, alpha=0.7, zorder=11)

    ax.set_xlim(P1.min(), P1.max()); ax.set_ylim(P2.min(), P2.max())
    ax.set_zlim(zmin, zmax * 1.05)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1, 1, 0.55))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0.3, 0.3, 0.3, 0.25))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("lookback", color=THEME["TEXT_DIM"], fontsize=11, labelpad=-8)
    ax.set_ylabel("z-threshold", color=THEME["TEXT_DIM"], fontsize=11, labelpad=-8)
    ax.set_zlabel("Sharpe", color=THEME["TEXT_DIM"], fontsize=11, labelpad=-8)

    fig.text(0.5, 0.075, "the tallest peak is almost always luck", ha="center",
             fontsize=14, color=THEME["RED"], fontweight="bold", style="italic",
             family=THEME["FONT"])
    fig.text(0.5, 0.045,
             f"best in-sample Sharpe = {Z[pi, pj]:.2f}  ·  robust edges live on plateaus, not spikes",
             ha="center", fontsize=11, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.985, 0.018, "@quant.dhawan", ha="right", va="bottom",
             fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Overfitting_Landscape_Static.png"))
