"""
Project 3 (Carousel Rank #3): Sharpe Ratio Dashboard
======================================================
Anyone can chase returns. Compute Sharpe ratio, Sortino ratio and max
drawdown for a handful of tickers, and rank them properly instead of
just eyeballing an equity curve.

Requirements: pip install yfinance pandas numpy
"""

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]
START = "2019-01-01"
END = "2024-01-01"
RISK_FREE_RATE = 0.02  # annualized


def load_daily_returns(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.pct_change().dropna()


def sharpe_ratio(daily_returns, rf_annual):
    rf_daily = rf_annual / 252
    excess = daily_returns - rf_daily
    return (excess.mean() / excess.std()) * np.sqrt(252)


def sortino_ratio(daily_returns, rf_annual):
    rf_daily = rf_annual / 252
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    downside_std = downside.std() if len(downside) > 0 else np.nan
    return (excess.mean() / downside_std) * np.sqrt(252)


def max_drawdown(daily_returns):
    equity = (1 + daily_returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def main():
    returns = load_daily_returns(TICKERS, START, END)

    rows = []
    for ticker in TICKERS:
        r = returns[ticker].dropna()
        rows.append({
            "Ticker": ticker,
            "Ann. Return": r.mean() * 252,
            "Ann. Vol": r.std() * np.sqrt(252),
            "Sharpe": sharpe_ratio(r, RISK_FREE_RATE),
            "Sortino": sortino_ratio(r, RISK_FREE_RATE),
            "Max Drawdown": max_drawdown(r),
        })

    dashboard = pd.DataFrame(rows).set_index("Ticker")
    dashboard = dashboard.sort_values("Sharpe", ascending=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(dashboard)
    dashboard.to_csv("sharpe_ratio_dashboard.csv")


if __name__ == "__main__":
    main()
