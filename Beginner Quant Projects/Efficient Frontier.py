"""
Project 5 (Carousel Rank #1): Efficient Frontier Optimizer
=============================================================
Simulate thousands of random portfolio weightings across a handful of
tickers, plot risk against return, and find the max-Sharpe portfolio
yourself instead of trusting a robo-advisor's black box.

Requirements: pip install yfinance pandas numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "JNJ", "XOM", "GLD"]
START = "2019-01-01"
END = "2024-01-01"
RISK_FREE_RATE = 0.02
N_PORTFOLIOS = 20_000


def load_annualized_stats(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    daily_returns = data.pct_change().dropna()
    mu = daily_returns.mean() * 252
    cov = daily_returns.cov() * 252
    return mu, cov


def simulate_portfolios(mu, cov, n_portfolios, rf):
    n_assets = len(mu)
    weights_record = np.zeros((n_portfolios, n_assets))
    results = np.zeros((n_portfolios, 3))  # return, vol, sharpe

    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        weights_record[i] = w

        port_return = np.dot(w, mu)
        port_vol = np.sqrt(w @ cov @ w)
        port_sharpe = (port_return - rf) / port_vol

        results[i] = [port_return, port_vol, port_sharpe]

    return results, weights_record


def main():
    mu, cov = load_annualized_stats(TICKERS, START, END)
    results, weights = simulate_portfolios(mu, cov, N_PORTFOLIOS, RISK_FREE_RATE)

    returns, vols, sharpes = results[:, 0], results[:, 1], results[:, 2]
    best_idx = np.argmax(sharpes)

    print("Max-Sharpe portfolio:")
    print(f"  Expected return: {returns[best_idx]:.2%}")
    print(f"  Volatility:      {vols[best_idx]:.2%}")
    print(f"  Sharpe ratio:    {sharpes[best_idx]:.2f}")
    print("  Weights:")
    for ticker, w in zip(TICKERS, weights[best_idx]):
        print(f"    {ticker}: {w:.1%}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(vols, returns, c=sharpes, cmap="viridis", s=4, alpha=0.5)
    ax.scatter(vols[best_idx], returns[best_idx], color="red", marker="*",
               s=300, edgecolors="black", label="Max Sharpe portfolio")
    fig.colorbar(sc, label="Sharpe ratio")
    ax.set_xlabel("Volatility (risk)")
    ax.set_ylabel("Expected return")
    ax.set_title(f"Efficient Frontier: {N_PORTFOLIOS:,} random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.savefig("efficient_frontier_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
