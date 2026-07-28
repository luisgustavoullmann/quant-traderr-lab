"""
Research Project 2 (Carousel Rank #4): Stationarity Testing
============================================================
Prices wander with no fixed mean. Returns do not. Run an Augmented
Dickey-Fuller test on both and see the difference.

Then watch two INDEPENDENT random walks produce a "significant"
regression: the spurious regression trap that invalidates careless
research.

Requirements: pip install yfinance pandas numpy statsmodels matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

TICKER = "SPY"
START = "2015-01-01"
END = "2024-01-01"


def load_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return data["Close"].squeeze().dropna()


def run_adf(series, name):
    """H0: series has a unit root (non-stationary)."""
    stat, p_value, _, _, crit, _ = adfuller(series, autolag="AIC")
    verdict = "STATIONARY" if p_value < 0.05 else "NON-STATIONARY"
    print(f"\n{name}")
    print(f"  ADF statistic : {stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  5% crit value : {crit['5%']:.4f}")
    print(f"  Verdict       : {verdict}")
    return p_value


def spurious_regression_demo(n=500, seed=42):
    """Two INDEPENDENT random walks. Any relationship found is fake."""
    rng = np.random.default_rng(seed)
    walk_a = np.cumsum(rng.normal(0, 1, n))
    walk_b = np.cumsum(rng.normal(0, 1, n))

    # Regress one on the other -- they are unrelated by construction
    X = sm.add_constant(walk_b)
    model = sm.OLS(walk_a, X).fit()

    print("\n=== Spurious regression demo ===")
    print("Two INDEPENDENT random walks regressed on each other:")
    print(f"  R-squared : {model.rsquared:.3f}")
    print(f"  t-stat    : {model.tvalues[1]:.2f}")
    print(f"  p-value   : {model.pvalues[1]:.4f}")
    print("  These look significant, but the series are unrelated by")
    print("  construction. This is why you test stationarity first.")

    # Now regress their DIFFERENCES (stationary) -- the effect vanishes
    d_model = sm.OLS(np.diff(walk_a), sm.add_constant(np.diff(walk_b))).fit()
    print(f"\nAfter differencing (both now stationary):")
    print(f"  R-squared : {d_model.rsquared:.3f}")
    print(f"  p-value   : {d_model.pvalues[1]:.4f}  (no longer significant)")

    return walk_a, walk_b


def main():
    prices = load_prices(TICKER, START, END)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    print(f"=== ADF tests on {TICKER} ===")
    run_adf(prices, "Price level")
    run_adf(log_returns, "Log returns")
    print("\nPrices fail, returns pass. Model returns, not prices.")

    walk_a, walk_b = spurious_regression_demo()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(prices.index, prices.values)
    axes[0, 0].set_title(f"{TICKER} price: non-stationary")

    axes[0, 1].plot(log_returns.index, log_returns.values, lw=0.7)
    axes[0, 1].axhline(log_returns.mean(), color="orange", ls="--")
    axes[0, 1].set_title(f"{TICKER} log returns: stationary")

    axes[1, 0].plot(walk_a, label="Walk A")
    axes[1, 0].plot(walk_b, label="Walk B")
    axes[1, 0].set_title("Two INDEPENDENT random walks")
    axes[1, 0].legend()

    axes[1, 1].scatter(walk_b, walk_a, s=6, alpha=0.5)
    axes[1, 1].set_title("Spurious relationship (they are unrelated)")
    axes[1, 1].set_xlabel("Walk B")
    axes[1, 1].set_ylabel("Walk A")

    plt.tight_layout()
    plt.savefig("stationarity_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
