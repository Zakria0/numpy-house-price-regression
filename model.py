"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    Xp = np.array(X, copy=True)
    mu = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)

    Xp = np.where(np.isnan(Xp), mu, Xp)
    return Xp

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    IQR = q3 - q1
    lower, upper = q1 - k * IQR, q3 + k * IQR

    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    return np.concatenate([X, col.reshape(-1, 1)], axis=1)

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    l = np.unique(labels)
    one_hot_matrix = np.zeros((len(labels), len(l)))

    for i in range(len(labels)):
        one_hot_matrix[i][np.where(l == labels[i])[0]] = 1

    return one_hot_matrix.astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return(mean, np.where(std == 0, 1., std))

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    return np.concatenate([np.ones((len(X), 1)), X], axis=1)

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    return np.random.RandomState(seed).permutation(np.arange(n_samples))

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    n = len(indices)
    train_n = int(train_ratio * n)
    val_n = int(val_ratio * n)

    return indices[:train_n], indices[train_n: train_n + val_n], indices[train_n + val_n:]

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    A = X.T @ X
    b = X.T @ y
    theta = np.linalg.solve(A, b)

    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    e = y_true - y_pred
    return float(np.mean(np.abs(e)))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    mean = np.mean(y_true)
    SS_res = np.sum((y_true - y_pred) ** 2)
    SS_tot = np.sum((y_true - mean) ** 2)

    return 1 - SS_res / SS_tot if SS_tot else 0.

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    r = y_true - y_pred
    return {'mean': np.mean(r), 'std': np.std(r), 'median_abs': np.median(np.abs(r))}

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    Xp = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(Xp, iqr_k)
    return clip_columns(Xp, lower, upper)

# Step 20 - assemble_feature_matrix
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    ratio = make_ratio_feature(X_num[:, ratio_num_idx], X_num[:, ratio_den_idx])
    X_num = append_column(X_num, ratio)
    if cat_labels is not None:
        X_num = np.hstack([X_num, one_hot_encode(cat_labels)])
    
    return X_num

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    p = make_shuffled_indices(X.shape[0], seed)
    train_idx, val_idx, test_idx = partition_indices(p, train_ratio, val_ratio)
    
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)

    return {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test}

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    mean, std = fit_standardizer(splits['X_train'])
    std_splits = {
        'X_train': add_bias_column(apply_standardizer(splits['X_train'], mean, std)),
        'y_train': splits['y_train'],
        'X_val':   add_bias_column(apply_standardizer(splits['X_val'], mean, std)),
        'y_val':   splits['y_val'],
        'X_test':  add_bias_column(apply_standardizer(splits['X_test'], mean, std)),
        'y_test':  splits['y_test'],
    }

    return std_splits, mean, std

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

