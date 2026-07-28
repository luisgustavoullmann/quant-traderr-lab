"""
Research Project 4 (Carousel Rank #2): Purged Walk-Forward CV
==============================================================
Standard k-fold cross-validation trains on the future to predict the
past, which quietly inflates every score in finance.

This builds walk-forward splits where training ALWAYS precedes
testing, with a purge gap so overlapping labels cannot leak across
the boundary, then shows the inflation by comparing against naive
k-fold on data with NO real signal.

Requirements: pip install numpy pandas scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

N_SAMPLES = 1200
LABEL_HORIZON = 60   # label looks 60 bars ahead -> 60 bars of overlap
LOOKBACK = 60        # features look 60 bars back -> autocorrelated
N_SPLITS = 5
N_TRIALS = 12        # average over many datasets: one run proves nothing
SEED = 7


def make_overlapping_dataset(n, lookback, horizon, seed):
    """Features and labels built from ONE pure-noise series.

    Features look BACKWARD over `lookback` bars, labels look FORWARD
    over `horizon` bars. Adjacent samples therefore share almost all
    of both windows, making them near-duplicates.

    There is no genuine predictive relationship: the feature window
    and the label window never overlap each other. So an honest CV
    must score ~0.50. k-fold scores higher only because shuffling
    puts a sample's near-duplicate neighbour in the training set."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n + lookback + horizon)

    X, y = [], []
    for i in range(lookback, lookback + n):
        past = noise[i - lookback:i]         # backward window
        future = noise[i:i + horizon]        # forward window (disjoint)
        X.append([past.sum(), past.mean(), past.std(),
                  past[-5:].sum(), past[-1], past.max(), past.min(),
                  np.median(past)])
        y.append(int(future.sum() > 0))
    return np.array(X), np.array(y)


def purged_walkforward_splits(n_samples, n_splits, purge):
    """Expanding-window splits with a purge gap between train and test."""
    test_size = n_samples // (n_splits + 1)
    for i in range(n_splits):
        train_end = test_size * (i + 1)
        test_start = train_end + purge
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx


def evaluate(X, y, splits, seed):
    scores = []
    for train_idx, test_idx in splits:
        model = RandomForestClassifier(n_estimators=100, random_state=seed,
                                        n_jobs=6)
        model.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
    return np.mean(scores)


def main():
    print("Dataset: pure noise. Features look back, labels look forward,")
    print("and the two windows never overlap, so there is NO real signal.")
    print("Honest accuracy should be ~0.50. Anything higher is leakage.")
    print(f"\nAveraging over {N_TRIALS} independent datasets, because a")
    print("single run of a noisy experiment proves nothing.\n")

    kfold, no_purge, purged = [], [], []

    for trial in range(N_TRIALS):
        seed = SEED + trial
        X, y = make_overlapping_dataset(N_SAMPLES, LOOKBACK, LABEL_HORIZON, seed)

        kfold.append(evaluate(X, y, list(
            KFold(n_splits=N_SPLITS, shuffle=True,
                  random_state=seed).split(X)), seed))
        no_purge.append(evaluate(X, y, list(
            purged_walkforward_splits(N_SAMPLES, N_SPLITS, purge=0)), seed))
        purged.append(evaluate(X, y, list(
            purged_walkforward_splits(N_SAMPLES, N_SPLITS,
                                       purge=LABEL_HORIZON)), seed))

    results = [
        ("Naive shuffled k-fold", kfold),
        ("Walk-forward (no purge)", no_purge),
        ("Purged walk-forward", purged),
    ]
    for label, scores in results:
        sem = np.std(scores) / np.sqrt(len(scores))
        print(f"{label:28s}: {np.mean(scores):.4f} +/- {sem:.4f} (SEM)")

    print("\nk-fold shuffles time, so a sample's near-duplicate neighbour")
    print("lands in the training set. It reports high accuracy on data")
    print("containing no signal at all. Walk-forward removes most of that.")
    print("\nNote: the purge mainly fixes a boundary leak affecting only the")
    print("first `horizon` test samples, so its effect is real but small")
    print("next to the k-fold catastrophe. Correctness, not a big number.")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["Naive\nk-fold", "Walk-forward\n(no purge)", "Purged\nwalk-forward"]
    means = [np.mean(s) for _, s in results]
    errs = [np.std(s) / np.sqrt(len(s)) for _, s in results]
    ax.bar(labels, means, yerr=errs, capsize=6,
           color=["crimson", "orange", "seagreen"])
    ax.axhline(0.5, color="black", ls="--", label="Honest baseline (50%)")
    ax.set_ylabel("Accuracy on pure-noise data")
    ax.set_title(f"CV scheme vs measured accuracy, {N_TRIALS} trials "
                  f"(higher = more leakage)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("purged_cv_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
