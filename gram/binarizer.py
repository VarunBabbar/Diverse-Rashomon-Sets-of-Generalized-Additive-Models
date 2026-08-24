#!/usr/bin/env python3
"""
GBDT-based binarization of tabular datasets for binary classification.

For each continuous feature, thresholds are extracted from the splits of a
gradient boosted stump ensemble and turned into ``feature<=t`` indicator
columns.  Already-binary (0/1) features are kept as-is.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# fit the tree using gradient boosted classifier
seed = 3


def fit_boosted_tree(X, y, n_est=10, lr=0.1, d=1):
    clf = GradientBoostingClassifier(loss='log_loss', learning_rate=lr, n_estimators=n_est, max_depth=d, random_state=seed)
    clf.fit(X, y)
    out = clf.score(X, y)
    return clf, out

# perform cut on the dataset


def cut(X, ts):
    df = X.copy()
    colnames = X.columns
    for j in range(len(ts)):
        for s in range(len(ts[j])):
            X[colnames[j]+'<='+str(ts[j][s])] = 1
            k = df[colnames[j]] > ts[j][s]
            X.loc[k, colnames[j]+'<='+str(ts[j][s])] = 0
        X = X.drop(colnames[j], axis=1)
    return X

# compute the thresholds


def get_thresholds(X, y, n_est, lr, d):
    # got a complaint here...
    y = np.ravel(y)
    # X is a dataframe
    clf, out = fit_boosted_tree(X, y, n_est, lr, d)
    thresholds = []

    for j in range(X.shape[1]):
        tj = np.array([])
        for i in range(len(clf.estimators_)):
            f = clf.estimators_[i, 0].tree_.feature
            t = clf.estimators_[i, 0].tree_.threshold
            tj = np.append(tj, t[f == j])
        tj = np.unique(tj)
        thresholds.append(tj.tolist())

    X_new = cut(X, thresholds)
    return X_new, thresholds, X_new.columns


def compute_thresholds(X, y, n_est, max_depth):
    # n_est, max_depth: GBDT parameters
    # set LR to 0.1
    lr = 0.1
    # GradientBoostingClassifier does not accept NaN; drop rows with missing values
    if isinstance(X, pd.DataFrame):
        valid = ~X.isna().any(axis=1)
    else:
        valid = ~np.isnan(np.asarray(X, dtype=float)).any(axis=1)
    if not valid.all():
        n_drop = int((~valid).sum())
        X = X.loc[valid] if isinstance(X, pd.DataFrame) else X[valid]
        if hasattr(y, "loc"):
            y = y.loc[valid]
        else:
            y = np.asarray(y)[np.asarray(valid)]
        if n_drop > 0:
            print(f"  Dropped {n_drop} rows with missing values for GBDT threshold fitting")
    start = time.perf_counter()
    X, thresholds, header = get_thresholds(X, y, n_est, lr, max_depth)
    guess_time = time.perf_counter()-start

    return X, thresholds, header, guess_time


def binarize_dataset(dataset, num_estimators):
    """Binarize a binary-classification dataset using GBDT thresholds.

    The last column of ``dataset`` is the (binary) label.  Features that are
    already binary are kept as-is; continuous features are replaced by
    ``feature<=t`` indicator columns.

    Returns:
        (binarized dataframe with label appended, thresholds per original
        column (None for already-binary columns), column header, GBDT time)
    """
    X, Y = pd.DataFrame(dataset.values[:, :-1], columns=dataset.columns[:-1]), pd.DataFrame(
        dataset.values[:, -1], columns=[dataset.columns[-1]])

    # Identify already-binary features (one-hot encoded)
    binary_cols = []
    continuous_cols = []
    for col in X.columns:
        unique_vals = X[col].dropna().unique()
        # Check if feature is already binary (only 0 and 1, or only 0, or only 1)
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)
    # Keep binary features as-is
    X_binary_keep = X[binary_cols].copy() if binary_cols else pd.DataFrame(index=X.index)

    # Binarize only continuous features
    if continuous_cols:
        X_continuous = X[continuous_cols]
        X_binary_cont, thresholds_cont, header_cont, threshold_guess_time = compute_thresholds(
            X_continuous, Y, n_est=num_estimators, max_depth=1)
        X_binary_cont = X_binary_cont[header_cont]
        # Combine binary and binarized continuous features
        X_binary = pd.concat([X_binary_keep.reset_index(drop=True), X_binary_cont.reset_index(drop=True)], axis=1)
        # Store thresholds as list matching original column order (None for binary cols)
        thresholds_full = []
        for col in X.columns:
            if col in binary_cols:
                thresholds_full.append(None)
            else:
                idx = continuous_cols.index(col)
                thresholds_full.append(thresholds_cont[idx])
        header = list(binary_cols) + list(header_cont)
    else:
        # All features are already binary
        X_binary = X_binary_keep
        thresholds_full = [None] * len(binary_cols)
        header = list(binary_cols)
        threshold_guess_time = 0.0

    return pd.concat([X_binary, Y], axis=1), thresholds_full, header, threshold_guess_time
