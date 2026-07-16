"""
Return_Distribution_Bars_Static_Pipeline.py
============================================
STATIC hero-image pipeline for "When the Tails Wake Up".

Standalone: DATA -> monthly return histograms -> render the full 3-D bar matrix
(month x return-bucket x frequency) at a fixed camera, colored by tail-ness.

    python Return_Distribution_Bars_Static_Pipeline.py
    -> Return_Distribution_Bars_Static.png  (5760x3240)
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
    "TICKER": "BTC-USD", "PERIOD": "4y", "BUCKET_EDGE": 0.11, "N_BUCKETS": 15,
    "W": 1920, "H": 1080, "DPI": 300, "ELEV": 28, "AZIM": -62,
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": "Arial",
}
CMAP = LinearSegmentedColormap.from_list(
    "tail", ["#0066ff", "#00f2ff", "#00ff8c", "#ffd400", "#ff9500", "#ff3050"],
    N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def fetch_returns():
    c = CONFIG
    log(f"[Data] Fetching {c['TICKER']}...")
    try:
        import yfinance as yf
        data = yf.download(c["TICKER"], period=c["PERIOD"], interval="1d",
                           progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(c["TICKER"], axis=1, level=1)
        px = data["Close"].dropna()
        return np.log(px / px.shift(1)).dropna()
    except Exception as e:
        log(f"[Data] yfinance failed ({e}); synthetic fallback")
        rng = np.random.default_rng(4)
        n = 1000
        vol = 0.02 + 0.05 * (np.sin(np.arange(n) / 60) > 0.6)
        r = rng.standard_t(3, n) * vol
        return pd.Series(r, index=pd.date_range(end=pd.Timestamp.today(),
                                                periods=n, freq="D"))


def build_matrix(ret):
    c = CONFIG
    edges = np.linspace(-c["BUCKET_EDGE"], c["BUCKET_EDGE"], c["N_BUCKETS"] + 1)
    M, labels = [], []
    for period, r in ret.groupby(ret.index.to_period("M")):
        clipped = np.clip(r.values, edges[0] + 1e-6, edges[-1] - 1e-6)
        M.append(np.histogram(clipped, bins=edges)[0])
        labels.append(period.strftime("%b %y"))
    return np.array(M, dtype=float), labels


def render_static(out):
    c = CONFIG
    ret = fetch_returns()
    M, labels = build_matrix(ret)
    n_month, n_buck = M.shape
    mid = (n_buck - 1) / 2
    log(f"months: {n_month}  buckets: {n_buck}  max count: {M.max():.0f}")

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"],
                     facecolor=THEME["BG"])
    fig.text(0.5, 0.945, "WHEN THE TAILS WAKE UP", ha="center", fontsize=27,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.905,
             "monthly BTC daily-return histograms  ·  color = tail-ness",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([0.03, -0.02, 0.94, 0.94], projection="3d",
                      facecolor=THEME["BG"])
    xpos, ypos, zpos, dz, cols = [], [], [], [], []
    for i in range(n_month):
        for j in range(n_buck):
            if M[i, j] < 0.5:
                continue
            xpos.append(i); ypos.append(j); zpos.append(0); dz.append(M[i, j])
            cols.append(CMAP(abs(j - mid) / mid))
    ax.bar3d(xpos, ypos, zpos, dx=0.85, dy=0.85, dz=dz, color=cols,
             shade=True, zsort="max")

    ax.set_xlim(0, n_month); ax.set_ylim(0, n_buck); ax.set_zlim(0, M.max() * 1.05)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1.6, 1.0, 0.6))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0.3, 0.3, 0.3, 0.2))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("month  →", color=THEME["TEXT_DIM"], fontsize=11, labelpad=-8)
    ax.set_ylabel("daily return", color=THEME["TEXT_DIM"], fontsize=11, labelpad=-8)
    ax.set_zlabel("frequency", color=THEME["TEXT_DIM"], fontsize=11, labelpad=-8)

    fig.text(0.5, 0.075, "calm months: a tight blue spike.  crashes: the red tails rise.",
             ha="center", fontsize=13, color=THEME["TEXT"], style="italic",
             family=THEME["FONT"])
    fig.text(0.5, 0.045, f"{labels[0]} — {labels[-1]}", ha="center", fontsize=11,
             color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.985, 0.018, "@quant.dhawan", ha="right", va="bottom",
             fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Return_Distribution_Bars_Static.png"))
