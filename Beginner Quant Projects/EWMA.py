"""
Project 1 (Carousel Rank #5): EWMA Crossover
=============================================
Your first backtest. Pull historical prices, compute a fast and slow
EWMA (exponentially weighted moving average), and go long when the
fast one crosses above the slow one.

An EWMA weights recent prices more heavily than a plain SMA, so it
reacts faster to real trend changes and lags less.

Requirements: pip install yfinance pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

TICKER = "AAPL"
START = "2018-01-01"
END = "2024-01-01"
FAST_SPAN = 20
SLOW_SPAN = 50


def load_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return data["Close"].squeeze().dropna()


def backtest_ewma_crossover(prices, fast_span, slow_span):
    df = pd.DataFrame({"price": prices})
    df["fast"] = df["price"].ewm(span=fast_span, adjust=False).mean()
    df["slow"] = df["price"].ewm(span=slow_span, adjust=False).mean()

    # Signal: 1 = long, 0 = flat. Shift by 1 day so today's position is
    # based on YESTERDAY's crossover state -- this avoids lookahead bias,
    # the single most common mistake in a first backtest.
    df["signal"] = (df["fast"] > df["slow"]).astype(int).shift(1).fillna(0)

    df["daily_return"] = df["price"].pct_change().fillna(0)
    df["strategy_return"] = df["signal"] * df["daily_return"]

    df["buy_hold_equity"] = (1 + df["daily_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    return df


def main():
    prices = load_prices(TICKER, START, END)
    result = backtest_ewma_crossover(prices, FAST_SPAN, SLOW_SPAN)

    final_strategy = result["strategy_equity"].iloc[-1]
    final_bh = result["buy_hold_equity"].iloc[-1]
    print(f"{TICKER}: EWMA crossover grew $1 to ${final_strategy:.2f}")
    print(f"{TICKER}: buy-and-hold grew $1 to ${final_bh:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.index, result["buy_hold_equity"], label="Buy & Hold")
    ax.plot(result.index, result["strategy_equity"], label="EWMA Crossover")
    ax.set_title(f"{TICKER}: EWMA Crossover vs Buy & Hold")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ewma_crossover_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
