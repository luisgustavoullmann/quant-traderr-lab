"""
Project 2 (Carousel Rank #4): Monte Carlo GBM Simulator
========================================================
Simulate thousands of possible future price paths using Geometric
Brownian Motion (the same equation behind Black-Scholes and most
portfolio risk models), then plot percentile bands around them.

Requirements: pip install yfinance numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

TICKER = "AAPL"
LOOKBACK_YEARS = 3
HORIZON_DAYS = 252       # simulate 1 trading year ahead
N_SIMULATIONS = 10_000


def estimate_gbm_params(ticker, lookback_years):
    data = yf.download(ticker, period=f"{lookback_years}y", auto_adjust=True)
    prices = data["Close"].squeeze().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    mu = float(log_returns.mean()) * 252          # annualized drift
    sigma = float(log_returns.std()) * np.sqrt(252)  # annualized volatility
    s0 = float(prices.iloc[-1])
    return s0, mu, sigma


def simulate_gbm_paths(s0, mu, sigma, horizon_days, n_sims):
    dt = 1 / 252
    z = np.random.standard_normal((horizon_days, n_sims))
    daily_log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(daily_log_returns, axis=0)
    price_paths = s0 * np.exp(log_paths)
    return np.vstack([np.full(n_sims, s0), price_paths])


def main():
    s0, mu, sigma = estimate_gbm_params(TICKER, LOOKBACK_YEARS)
    print(f"{TICKER}: S0=${s0:.2f}  mu={mu:.2%}/yr  sigma={sigma:.2%}/yr")

    paths = simulate_gbm_paths(s0, mu, sigma, HORIZON_DAYS, N_SIMULATIONS)

    p5 = np.percentile(paths, 5, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p95 = np.percentile(paths, 95, axis=1)

    print(f"In {HORIZON_DAYS} trading days:")
    print(f"  5th percentile:  ${p5[-1]:.2f}")
    print(f"  Median:          ${p50[-1]:.2f}")
    print(f"  95th percentile: ${p95[-1]:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(paths[:, :200], color="steelblue", alpha=0.05)
    ax.plot(p50, color="orange", lw=2, label="Median path")
    ax.plot(p5, color="crimson", lw=1.5, linestyle="--", label="5th percentile")
    ax.plot(p95, color="green", lw=1.5, linestyle="--", label="95th percentile")
    ax.set_title(f"{TICKER}: {N_SIMULATIONS:,} Monte Carlo GBM Simulations")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Simulated price ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("monte_carlo_gbm_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
