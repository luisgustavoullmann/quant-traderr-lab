"""
Fat_Tail_Isosurface_Static_Pipeline.py
=======================================
STATIC hero-image pipeline for "The Shape of Fat Tails".

Standalone: DATA -> whiten -> 3-D KDE -> ray-cast the density level-set ->
ONE high-res frame at a fixed angle chosen to put the biggest tail spike in
profile against the Gaussian reference sphere. Denser ray grid than the reel.

    python Fat_Tail_Isosurface_Static_Pipeline.py
    -> Fat_Tail_Isosurface_Static.png  (3240x5760)
"""
import os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde, chi2
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TICKERS": ["SPY", "GLD", "BTC-USD"], "PERIOD": "5y", "MASS": 0.90,
    "N_THETA": 160, "N_PHI": 80,           # denser than the reel
    "R_MIN": 0.30, "R_MAX": 7.0, "N_R": 60,
    "SMOOTH": (1.7, 1.7),
    "W": 1080, "H": 1920, "DPI": 300, "ELEV": 16,
}
THEME = {"BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
         "ORANGE": "#ff9500", "RED": "#ff3050", "FONT": "Arial"}
CMAP = LinearSegmentedColormap.from_list(
    "tail", ["#0055ff", "#00c2ff", "#00ff8c", "#ffd400", "#ff9500", "#ff3050"],
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
        return np.log(raw / raw.shift(1)).dropna().values
    except Exception as e:
        log(f"[Data] yfinance failed ({e}); synthetic fallback")
        rng = np.random.default_rng(17)
        n = 1200
        g = rng.standard_t(3, n)[:, None] * np.array([0.9, 0.4, 1.6])
        idio = rng.standard_t(4, (n, 3)) * np.array([0.6, 0.8, 1.7])
        return (g + idio) * 0.01


def whiten(X):
    mu = X.mean(0)
    S = np.cov(X - mu, rowvar=False)
    vals, vecs = np.linalg.eigh(S)
    W = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, 1e-16, None))) @ vecs.T
    return (X - mu) @ W


def ray_cast_levelset(Z):
    c = CONFIG
    kde = gaussian_kde(Z.T)
    r0 = float(np.sqrt(chi2.ppf(c["MASS"], df=3)))
    L = float((2 * np.pi) ** -1.5 * np.exp(-0.5 * r0 ** 2))

    th = np.linspace(0, 2 * np.pi, c["N_THETA"], endpoint=False)
    ph = np.linspace(0.02, np.pi - 0.02, c["N_PHI"])
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    U = np.stack([np.sin(PH) * np.cos(TH), np.sin(PH) * np.sin(TH),
                  np.cos(PH)], axis=-1)
    radii = np.linspace(c["R_MIN"], c["R_MAX"], c["N_R"])
    pts = (U[..., None, :] * radii[:, None]).reshape(-1, 3)
    log(f"KDE eval: {pts.shape[0]:,} points...")
    dens = kde(pts.T).reshape(U.shape[0], U.shape[1], len(radii))

    inside = dens >= L
    R = np.full(U.shape[:2], c["R_MIN"])
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            k = np.nonzero(inside[i, j])[0]
            if len(k) == 0:
                continue
            kk = k[-1]
            if kk + 1 < len(radii):
                d0, d1 = dens[i, j, kk], dens[i, j, kk + 1]
                f = (d0 - L) / max(d0 - d1, 1e-30)
                R[i, j] = radii[kk] + f * (radii[kk + 1] - radii[kk])
            else:
                R[i, j] = radii[kk]
    return gaussian_filter(R, sigma=c["SMOOTH"], mode="wrap"), TH, PH, r0


def render_static(out):
    c = CONFIG
    Z = whiten(fetch_returns())
    R, TH, PH, r0 = ray_cast_levelset(Z)
    excess = float(R.max() / r0)
    log(f"r0={r0:.2f}  empirical r=[{R.min():.2f}, {R.max():.2f}]  "
        f"excess={excess:.2f}x")

    # aim the camera so the biggest spike sits in profile, not facing us
    i, j = np.unravel_index(np.argmax(R), R.shape)
    azim = np.degrees(TH[i, j]) - 90.0

    m = float(np.abs(R - r0).max()) + 1e-9
    norm = np.clip(0.5 + 0.5 * (R - r0) / m, 0, 1)
    X = R * np.sin(PH) * np.cos(TH)
    Y = R * np.sin(PH) * np.sin(TH)
    Z3 = R * np.cos(PH)

    su = np.linspace(0, 2 * np.pi, 80)
    sv = np.linspace(0, np.pi, 44)
    SX = r0 * np.outer(np.cos(su), np.sin(sv))
    SY = r0 * np.outer(np.sin(su), np.sin(sv))
    SZ = r0 * np.outer(np.ones_like(su), np.cos(sv))

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"],
                     facecolor=THEME["BG"])
    fig.text(0.5, 0.935, "THE SHAPE OF FAT TAILS", ha="center", fontsize=29,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.905, "3D density level-set  ·  SPY · GLD · BTC",
             ha="center", fontsize=15, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, 0.883, "vs the Gaussian sphere it should have been",
             ha="center", fontsize=13, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])

    ax = fig.add_axes([-0.22, 0.145, 1.44, 0.735], projection="3d",
                      facecolor=THEME["BG"])
    ax.plot_wireframe(SX, SY, SZ, color=(1, 1, 1, 0.28), linewidth=0.6,
                      rstride=3, cstride=2, zorder=1)
    ax.plot_surface(X, Y, Z3, facecolors=CMAP(norm), rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, zorder=2)

    lim = max(R.max(), r0) * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=c["ELEV"], azim=azim)
    ax.set_box_aspect((1, 1, 1))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, 0.176, f"tails reach {excess:.1f}× the Gaussian radius",
             ha="center", fontsize=17, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, 0.150, f"wireframe = {int(c['MASS']*100)}% Gaussian shell  ·  "
             f"green on it, red far beyond", ha="center", fontsize=12,
             color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, 0.098, "The bell curve", ha="center", fontsize=17,
             color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
    fig.text(0.5, 0.070, "is a sphere.", ha="center", fontsize=17,
             color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
    fig.text(0.5, 0.042, "Reality has spikes.", ha="center", fontsize=17,
             color=THEME["TEXT"], style="italic", family=THEME["FONT"])
    fig.text(0.5, 0.012, "@quant.dhawan", ha="center", fontsize=13,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Fat_Tail_Isosurface_Static.png"))
