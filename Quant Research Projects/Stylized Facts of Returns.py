"""
Research Project 1 (Carousel Rank #5): Stylized Facts of Returns
=================================================================
Before modelling anything, measure what returns actually do. Compare
the return distribution to a fitted normal, build a Q-Q plot, and
check volatility clustering.

You will find fat tails and vol clustering, which is exactly why
models assuming normality underestimate risk.

Requirements: pip install yfinance pandas numpy scipy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

TICKER = "SPY"
START = "2010-01-01"
END = "2024-01-01"


def load_log_returns(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    prices = data["Close"].squeeze().dropna()
    return np.log(prices / prices.shift(1)).dropna()


def report_moments(returns):
    """Excess kurtosis > 0 means fatter tails than a normal."""
    print(f"Observations:      {len(returns):,}")
    print(f"Mean (daily):      {returns.mean():.5f}")
    print(f"Std dev (daily):   {returns.std():.5f}")
    print(f"Skew:              {stats.skew(returns):.3f}")
    print(f"Excess kurtosis:   {stats.kurtosis(returns):.3f}  (normal = 0)")

    # Jarque-Bera tests normality directly
    jb_stat, jb_p = stats.jarque_bera(returns)
    print(f"Jarque-Bera p:     {jb_p:.2e}  (p < 0.05 rejects normality)")


def count_tail_events(returns):
    """How many 'impossible' moves actually happened?"""
    z = (returns - returns.mean()) / returns.std()
    for n_sigma in (3, 4, 5):
        actual = (np.abs(z) > n_sigma).sum()
        # Expected count under a normal distribution
        expected = 2 * stats.norm.sf(n_sigma) * len(returns)
        print(f"  |move| > {n_sigma} sigma: {actual:4d} actual vs "
              f"{expected:6.2f} expected under normal")


def volatility_clustering(returns, max_lag=20):
    """Returns are ~uncorrelated, but their MAGNITUDE is persistent."""
    ret_ac = [returns.autocorr(lag=k) for k in range(1, max_lag + 1)]
    abs_ac = [returns.abs().autocorr(lag=k) for k in range(1, max_lag + 1)]
    print(f"Mean |autocorr| of returns   (lags 1-{max_lag}): {np.mean(np.abs(ret_ac)):.4f}")
    print(f"Mean  autocorr  of |returns| (lags 1-{max_lag}): {np.mean(abs_ac):.4f}")
    print("  Second value being much larger = volatility clustering.")
    return ret_ac, abs_ac


def main():
    returns = load_log_returns(TICKER, START, END)

    print(f"\n=== {TICKER} return moments ===")
    report_moments(returns)

    print(f"\n=== Tail events ===")
    count_tail_events(returns)

    print(f"\n=== Volatility clustering ===")
    ret_ac, abs_ac = volatility_clustering(returns)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    clip = np.percentile(np.abs(returns), 99.5)
    axes[0, 0].hist(returns, bins=np.linspace(-clip, clip, 80),
                    density=True, alpha=0.7, label="Actual")
    xs = np.linspace(-clip, clip, 400)
    axes[0, 0].plot(xs, stats.norm.pdf(xs, returns.mean(), returns.std()),
                    lw=2, label="Normal fit")
    axes[0, 0].set_title("Return distribution vs normal")
    axes[0, 0].legend()

    stats.probplot(returns, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q plot (curved ends = fat tails)")

    axes[1, 0].bar(range(1, len(ret_ac) + 1), ret_ac)
    axes[1, 0].set_title("Autocorrelation of returns (near zero)")
    axes[1, 0].set_xlabel("Lag")

    axes[1, 1].bar(range(1, len(abs_ac) + 1), abs_ac, color="darkorange")
    axes[1, 1].set_title("Autocorrelation of |returns| (persistent)")
    axes[1, 1].set_xlabel("Lag")

    plt.tight_layout()
    plt.savefig("stylized_facts_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
