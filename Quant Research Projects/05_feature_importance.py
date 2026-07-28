"""
Research Project 5 (Carousel Rank #1): Feature Importance
==========================================================
Fit a model on candidate predictors, then measure PERMUTATION
importance: shuffle each feature and see how much accuracy drops.

Critically, include a deliberately random control column. Any feature
that cannot beat that control is noise, no matter how good the story
sounds.

Requirements: pip install yfinance pandas numpy scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

TICKER = "SPY"
START = "2010-01-01"
END = "2024-01-01"
N_REPEATS = 10
SEED = 7


def build_features(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    close = data["Close"].squeeze().dropna()
    volume = data["Volume"].squeeze().dropna()

    df = pd.DataFrame(index=close.index)
    ret = close.pct_change()

    df["momentum_21d"] = close.pct_change(21)
    df["momentum_63d"] = close.pct_change(63)
    df["realized_vol_21d"] = ret.rolling(21).std()
    df["vol_ratio"] = ret.rolling(5).std() / ret.rolling(63).std()
    df["volume_zscore"] = ((volume - volume.rolling(63).mean())
                            / volume.rolling(63).std())
    df["dist_from_high"] = close / close.rolling(252).max() - 1

    # The control: pure noise. Anything below this is not a predictor.
    rng = np.random.default_rng(SEED)
    df["shuffled_control"] = rng.normal(size=len(df))

    # Label: does the next 5 days deliver a positive return?
    df["target"] = (close.shift(-5) / close - 1 > 0).astype(int)
    return df.dropna()


def main():
    df = build_features(TICKER, START, END)
    feature_cols = [c for c in df.columns if c != "target"]

    X = df[feature_cols].values
    y = df["target"].values

    # Chronological split -- never shuffle time series
    split = int(len(df) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=300, max_depth=5,
                                    random_state=SEED, n_jobs=6)
    model.fit(X_train, y_train)
    print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test accuracy : {model.score(X_test, y_test):.4f}")

    result = permutation_importance(model, X_test, y_test,
                                     n_repeats=N_REPEATS,
                                     random_state=SEED, n_jobs=6)

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False)

    control_score = imp.loc[imp["feature"] == "shuffled_control",
                             "importance"].iloc[0]

    print(f"\n=== Permutation importance (noise floor = {control_score:.5f}) ===")
    for _, row in imp.iterrows():
        verdict = "NOISE" if row["importance"] <= control_score else "signal"
        marker = "  <-- CONTROL" if row["feature"] == "shuffled_control" else ""
        print(f"  {row['feature']:20s} {row['importance']:+.5f} "
              f"+/- {row['std']:.5f}  [{verdict}]{marker}")

    survivors = imp[(imp["importance"] > control_score)
                    & (imp["feature"] != "shuffled_control")]
    print(f"\n{len(survivors)} of {len(feature_cols) - 1} features beat the "
          f"noise floor.")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["grey" if f == "shuffled_control" else "mediumpurple"
              for f in imp["feature"]]
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"],
            color=colors, capsize=4)
    ax.axvline(control_score, color="crimson", ls="--",
               label="Noise floor (shuffled control)")
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in accuracy")
    ax.set_title(f"{TICKER}: permutation importance vs shuffled control")
    ax.legend()
    plt.tight_layout()
    plt.savefig("feature_importance_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
