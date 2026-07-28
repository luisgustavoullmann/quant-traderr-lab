"""
Research Project 3 (Carousel Rank #3): Factor Research
=======================================================
Take a candidate signal (here: 12-1 month momentum), rank every stock
by it each month, sort into quintiles, and measure mean forward
return per bucket.

A real factor produces a monotonic ladder and a positive Q5 minus Q1
spread, not a random scatter. Also computes the information
coefficient (IC): the rank correlation between signal and forward
return.

Requirements: pip install yfinance pandas numpy scipy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM",
    "JNJ", "XOM", "PG", "KO", "PEP", "WMT", "DIS", "CSCO", "INTC",
    "VZ", "PFE", "MRK", "BAC", "CVX", "T", "NKE", "MCD",
]
START = "2015-01-01"
END = "2024-01-01"
N_QUINTILES = 5


def load_monthly_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.resample("ME").last().dropna(how="all")


def build_momentum_signal(monthly):
    """12-1 momentum: 12-month return, skipping the most recent month
    (the skip avoids short-term reversal contaminating the signal)."""
    return (monthly.shift(1) / monthly.shift(12)) - 1


def forward_returns(monthly):
    return monthly.shift(-1) / monthly - 1


def quintile_study(signal, fwd_ret, n_q):
    """For each month, sort into quintiles and record mean fwd return."""
    rows = []
    ic_series = []

    for date in signal.index:
        sig = signal.loc[date].dropna()
        ret = fwd_ret.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < n_q * 2:
            continue
        sig, ret = sig[common], ret[common]

        # Information coefficient: rank correlation this month
        ic, _ = stats.spearmanr(sig, ret)
        ic_series.append(ic)

        ranks = sig.rank(pct=True)
        buckets = np.ceil(ranks * n_q).clip(1, n_q).astype(int)
        rows.append(ret.groupby(buckets).mean())

    panel = pd.DataFrame(rows)
    return panel, np.array(ic_series)


def main():
    monthly = load_monthly_prices(TICKERS, START, END)
    signal = build_momentum_signal(monthly)
    fwd = forward_returns(monthly)

    panel, ics = quintile_study(signal, fwd, N_QUINTILES)
    mean_by_q = panel.mean()
    se_by_q = panel.std() / np.sqrt(panel.count())

    print("=== 12-1 Momentum quintile study ===")
    print(f"Months analysed: {len(panel)}")
    for q in mean_by_q.index:
        print(f"  Q{q}: mean fwd return {mean_by_q[q]:+.4%} "
              f"(SE {se_by_q[q]:.4%})")

    spread = mean_by_q.iloc[-1] - mean_by_q.iloc[0]
    long_short = panel.iloc[:, -1] - panel.iloc[:, 0]
    t_stat = long_short.mean() / (long_short.std() / np.sqrt(len(long_short)))

    print(f"\nQ5 minus Q1 spread : {spread:+.4%} per month")
    print(f"Long-short t-stat  : {t_stat:.2f}  (|t| > 2 is the usual bar)")
    print(f"Mean IC            : {np.nanmean(ics):+.4f}")
    print(f"IC std dev         : {np.nanstd(ics):.4f}")

    print("\n--- How to read this ---")
    if abs(t_stat) < 2:
        print("This run does NOT clear the significance bar. That is a")
        print("legitimate finding, not a broken script. On a 25-stock")
        print("large-cap universe over a single decade you have very little")
        print("breadth, and large-cap US momentum was weak in this period.")
        print("A null result you can defend beats a fake one you cannot.")
        print("\nTry: a wider universe, a longer history, sector-neutral")
        print("ranking, or a different signal (value, low-vol, quality).")
    else:
        print("The spread clears |t| > 2 on this sample. Before believing it,")
        print("check monotonicity across quintiles and stability of the IC,")
        print("then re-test out of sample.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar([f"Q{q}" for q in mean_by_q.index], mean_by_q.values,
                yerr=se_by_q.values, capsize=5)
    axes[0].axhline(0, color="grey", lw=1)
    axes[0].set_title("Mean forward return by momentum quintile")
    axes[0].set_ylabel("Monthly return")

    axes[1].plot(np.cumsum(long_short.values))
    axes[1].set_title("Cumulative Q5 minus Q1 (long-short) return")
    axes[1].set_xlabel("Month")

    plt.tight_layout()
    plt.savefig("factor_research_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
