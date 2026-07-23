"""
Project 4 (Carousel Rank #2): Mean-Reversion Pairs Bot
=========================================================
Find two historically correlated stocks, test whether their spread is
cointegrated (not just correlated), then trade the z-score of that
spread instead of either stock's price. Long one, short the other:
market-neutral, statistical arbitrage style trading.

Requirements: pip install yfinance pandas numpy statsmodels matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import coint

TICKER_A = "KO"
TICKER_B = "PEP"
START = "2019-01-01"
END = "2024-01-01"
ZSCORE_WINDOW = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5


def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.dropna()


def test_cointegration(price_a, price_b):
    _, p_value, _ = coint(price_a, price_b)
    return p_value


def build_zscore_spread(price_a, price_b, window):
    hedge_ratio = np.polyfit(price_b, price_a, 1)[0]
    spread = price_a - hedge_ratio * price_b
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return spread, zscore, hedge_ratio


def generate_signals(zscore, entry_z, exit_z):
    position = pd.Series(0, index=zscore.index)
    current = 0
    for i, z in enumerate(zscore):
        if pd.isna(z):
            continue
        if current == 0 and z > entry_z:
            current = -1   # spread too high: short A, long B
        elif current == 0 and z < -entry_z:
            current = 1    # spread too low: long A, short B
        elif current != 0 and abs(z) < exit_z:
            current = 0
        position.iloc[i] = current
    return position


def main():
    prices = load_prices([TICKER_A, TICKER_B], START, END)
    p_value = test_cointegration(prices[TICKER_A], prices[TICKER_B])
    print(f"Cointegration test p-value for {TICKER_A}/{TICKER_B}: {p_value:.4f}")
    print("p < 0.05 suggests a genuinely mean-reverting spread, not just correlation.")

    spread, zscore, hedge_ratio = build_zscore_spread(
        prices[TICKER_A], prices[TICKER_B], ZSCORE_WINDOW
    )
    position = generate_signals(zscore, ENTRY_Z, EXIT_Z)

    spread_return = spread.diff().fillna(0)
    strategy_return = position.shift(1).fillna(0) * spread_return
    equity = strategy_return.cumsum()

    print(f"Hedge ratio ({TICKER_A} per {TICKER_B}): {hedge_ratio:.3f}")
    print(f"Cumulative spread P&L: {equity.iloc[-1]:.2f}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(zscore.index, zscore, color="steelblue")
    axes[0].axhline(ENTRY_Z, color="red", linestyle="--", lw=1)
    axes[0].axhline(-ENTRY_Z, color="green", linestyle="--", lw=1)
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_title(f"{TICKER_A}/{TICKER_B} spread z-score")

    axes[1].plot(position.index, position, color="darkorange")
    axes[1].set_title("Position (+1 long spread, -1 short spread)")

    axes[2].plot(equity.index, equity, color="seagreen")
    axes[2].set_title("Cumulative strategy P&L")

    plt.tight_layout()
    plt.savefig("pairs_trading_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
