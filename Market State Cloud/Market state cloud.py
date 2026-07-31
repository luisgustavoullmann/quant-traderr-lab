"""
Market_State_Cloud_Static_Pipeline.py
======================================
STATIC hero-image pipeline for "The Market in State Space".

Standalone: DATA -> (momentum, volatility, trend) state vectors colored by the
forward 5-day return, threaded by time-lines -> render the full cloud at a fixed
camera. Colors are shuffled: that shuffle IS the efficient market.

    python Market_State_Cloud_Static_Pipeline.py
    -> Market_State_Cloud_Static.png  (5760x3240)
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
    "TICKER": "BTC-USD", "PERIOD": "4y",
    "MOM_S": 10, "VOL_W": 20, "MOM_L": 50, "FWD": 5,
    "W": 1920, "H": 1080, "DPI": 300, "ELEV": 22, "AZIM": -54,
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": "Arial",
}
CMAP = LinearSegmentedColormap.from_list(
    "fwd", ["#0066ff", "#00c2ff", "#7f8fa6", "#ffb000", "#ff3050"], N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def fetch_close():
    c = CONFIG
    log(f"[Data] Fetching {c['TICKER']}...")
    try:
        import yfinance as yf
        data = yf.download(c["TICKER"], period=c["PERIOD"], interval="1d",
                           progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(c["TICKER"], axis=1, level=1)
        return data["Close"].dropna()
    except Exception as e:
        log(f"[Data] yfinance failed ({e}); synthetic fallback")
        rng = np.random.default_rng(6)
        n = 1000
        px = 30000 * np.exp(np.cumsum(rng.normal(0.0004, 0.03, n)))
        return pd.Series(px, index=pd.date_range(end=pd.Timestamp.today(),
                                                 periods=n, freq="D"))


def build_states(px):
    c = CONFIG
    lp = np.log(px)
    df = pd.DataFrame({
        "x": lp - lp.shift(c["MOM_S"]),
        "y": lp.diff().rolling(c["VOL_W"]).std(),
        "z": lp - lp.shift(c["MOM_L"]),
        "f": lp.shift(-c["FWD"]) - lp,
    }).dropna()

    def zc(a): return (a - a.mean()) / (a.std() + 1e-9)
    X = np.stack([zc(df["x"]).values, zc(df["y"]).values, zc(df["z"]).values], 1)
    fwd = df["f"].values
    q = np.percentile(np.abs(fwd), 90) + 1e-9
    return X, np.clip(fwd / (2 * q) + 0.5, 0, 1), fwd


def render_static(out):
    c = CONFIG
    px = fetch_close()
    X, fnorm, fwd = build_states(px)
    lim = float(np.percentile(np.abs(X), 97))
    log(f"states: {len(X)}  fwd-return range: [{fwd.min()*100:.1f}%, {fwd.max()*100:.1f}%]")

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"],
                     facecolor=THEME["BG"])
    fig.text(0.5, 0.945, "THE MARKET IN STATE SPACE", ha="center", fontsize=27,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.905,
             "each day = (momentum, volatility, trend)  ·  color = next 5-day return",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([-0.02, -0.06, 1.04, 1.02], projection="3d",
                      facecolor=THEME["BG"])
    pts = X.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = CMAP(fnorm[:-1]); cols[:, -1] = 0.20
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=0.5))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=30, c=CMAP(fnorm),
               depthshade=False, edgecolors="none", alpha=0.9, zorder=5)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1, 1, 1))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0.3, 0.3, 0.3, 0.2))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, 0.072, "if color clustered, there'd be an edge.  mostly it's shuffled.",
             ha="center", fontsize=13, color=THEME["TEXT"], style="italic",
             family=THEME["FONT"])
    fig.text(0.5, 0.043, "blue = market fell next   ·   red = market rose next",
             ha="center", fontsize=11, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.985, 0.018, "@quant.dhawan", ha="right", va="bottom",
             fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Market_State_Cloud_Static.png"))
