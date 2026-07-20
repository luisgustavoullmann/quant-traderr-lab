"""
Market_Vector_Field_Static_Pipeline.py
=======================================
STATIC hero-image pipeline for "The Market's Wind Map".

Standalone: DATA -> rolling VAR(1) drift fields -> pick the most ROTATIONAL
window (largest imaginary eigenvalue = most swirl) -> render ONE high-res frame
with a fixed camera and fully-drawn streamlines (no comet phase).

    python Market_Vector_Field_Static_Pipeline.py
    -> Market_Vector_Field_Static.png  (5760x3240)
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TICKERS": ["SPY", "GLD", "BTC-USD"], "PERIOD": "3y", "WINDOW": 60,
    "FIELD_GAIN": 6.0, "CUBE": 2.0, "GRID_N": 5,
    "N_STREAM": 16, "STREAM_STEPS": 220, "STREAM_DT": 0.05,
    "W": 1920, "H": 1080, "DPI": 300, "ELEV": 24, "AZIM": -48,
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": "Arial",
}
CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#0066ff", "#00f2ff", "#00ff8c", "#ffd400", "#ff9500", "#ff3050"],
    N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def fetch_returns():
    c = CONFIG
    log(f"[Data] Fetching {c['TICKERS']}...")
    try:
        import yfinance as yf
        raw = yf.download(c["TICKERS"], period=c["PERIOD"], interval="1d",
                          progress=False)["Close"]
        raw = raw[c["TICKERS"]].dropna()
        return np.log(raw / raw.shift(1)).dropna().values, raw.index
    except Exception as e:
        log(f"[Data] yfinance failed ({e}); synthetic fallback")
        rng = np.random.default_rng(5)
        n = 760
        x = np.zeros((n, 3))
        for i in range(1, n):
            ph = np.sin(i / 90.0)
            A = np.array([[0.90, 0.05, 0.02], [-0.03, 0.92, 0.01],
                          [0.10 * ph, 0.02, 0.88]])
            x[i] = A @ x[i - 1] + rng.normal(0, 0.4, 3)
        return x * 0.01, pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")


def fit_fields(rets):
    c = CONFIG
    win = c["WINDOW"]
    Ms, swirl, idx = [], [], []
    for i in range(win, len(rets)):
        z = rets[i - win:i]
        z = (z - z.mean(0)) / (z.std(0) + 1e-9)
        A = np.linalg.lstsq(z[:-1], z[1:], rcond=None)[0].T
        M = A - np.eye(3)
        Ms.append(M)
        swirl.append(np.max(np.abs(np.linalg.eigvals(M).imag)))
        idx.append(i)
    return Ms, np.array(swirl), np.array(idx)


def streamline(M, seed):
    c = CONFIG
    pts, x = [seed.copy()], seed.copy()
    for _ in range(c["STREAM_STEPS"]):
        x = x + c["FIELD_GAIN"] * (M @ x) * c["STREAM_DT"]
        if np.max(np.abs(x)) > c["CUBE"] * 1.15:
            break
        pts.append(x.copy())
    return np.array(pts)


def grid_points():
    c = CONFIG
    g = np.linspace(-c["CUBE"] * 0.8, c["CUBE"] * 0.8, c["GRID_N"])
    P = np.array([[x, y, z] for x in g for y in g for z in g])
    return P[np.linalg.norm(P, axis=1) > 0.3]


def seeds():
    n = CONFIG["N_STREAM"]
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    r = CONFIG["CUBE"] * 0.95
    return np.stack([r * np.cos(th) * np.sin(phi), r * np.sin(th) * np.sin(phi),
                     r * np.cos(phi)], axis=1)


def render_static(out):
    c = CONFIG
    rets, dates = fetch_returns()
    Ms, swirl, order_idx = fit_fields(rets)
    wi = int(np.argmax(swirl))                         # most rotational field
    M = Ms[wi]
    log(f"windows: {len(Ms)}  max swirl at {pd.to_datetime(dates[order_idx[wi]]).strftime('%b %Y')}")

    P, S = grid_points(), seeds()

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"],
                     facecolor=THEME["BG"])
    fig.text(0.5, 0.945, "THE MARKET'S WIND MAP", ha="center", fontsize=27,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.905,
             "drift field  v(x) = (A − I)x  of SPY · GLD · BTC  ·  rolling VAR(1)",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([-0.04, -0.10, 1.08, 1.12], projection="3d",
                      facecolor=THEME["BG"])

    V = (M @ P.T).T
    speed = np.linalg.norm(V, axis=1); smax = speed.max() + 1e-9
    Vn = V / (speed[:, None] + 1e-9) * (c["CUBE"] * 0.28)
    for p, v, sp in zip(P, Vn, speed):
        ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color=CMAP(sp / smax),
                  alpha=0.30, linewidth=0.9, arrow_length_ratio=0.35)
    for s in S:
        path = streamline(M, s)
        if len(path) < 3:
            continue
        segs = np.concatenate([path[:-1, None], path[1:, None]], axis=1)
        sp = np.linalg.norm((M @ path[:-1].T).T, axis=1)
        cols = CMAP(np.clip(sp / smax, 0, 1)); cols[:, -1] = 0.5
        ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=0.9))
    ax.scatter([0], [0], [0], s=110, color=THEME["CYAN"],
               edgecolors="white", linewidths=0.9, zorder=12)

    lim = c["CUBE"] * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1, 1, 1))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    cur = pd.to_datetime(dates[order_idx[wi]]).strftime("%b %Y")
    fig.text(0.5, 0.095, "ROTATING", ha="center", fontsize=15,
             color=THEME["YELLOW"], fontweight="bold", family=THEME["FONT"])
    fig.text(0.5, 0.062, f"cyan dot = equilibrium   ·   {cur}", ha="center",
             fontsize=11, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.985, 0.018, "@quant.dhawan", ha="right", va="bottom",
             fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Market_Vector_Field_Static.png"))
