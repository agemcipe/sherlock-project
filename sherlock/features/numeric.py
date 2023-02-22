# Extract Features from numeric data
import pandas as pd
from collections import OrderedDict
import numpy as np


def extract_numeric_features(col_values: list, features: OrderedDict):
    # values that can be converted to numeric
    pure_numeric_values = pd.to_numeric(col_values, errors="coerce")
    MAX_VALUE = 100_000_000  # TODO decide
    # statistics of numeric values
    pure_numeric_values = np.clip(
        pure_numeric_values[~pd.isna(pure_numeric_values)], -MAX_VALUE, MAX_VALUE
    )

    if len(pure_numeric_values) > 2:
        features["avg_numeric_values"] = np.mean(pure_numeric_values)
        features["std_numeric_values"] = np.std(pure_numeric_values)
        features["min_numeric_values"] = np.min(pure_numeric_values)
        features["max_numeric_values"] = np.max(pure_numeric_values)
        features["median_numeric_values"] = np.median(pure_numeric_values)
        features["sum_numeric_values"] = np.sum(pure_numeric_values)
        features["var_numeric_values"] = np.var(pure_numeric_values)
        features["iqr_numeric_values"] = np.subtract(
            *np.percentile(pure_numeric_values, [75, 25])
        )
    else:
        # TODO: impute with gloabal statistics
        features["avg_numeric_values"] = 0
        features["std_numeric_values"] = 0
        features["min_numeric_values"] = 0
        features["max_numeric_values"] = 0
        features["median_numeric_values"] = 0
        features["sum_numeric_values"] = 0
        features["var_numeric_values"] = 0
        features["iqr_numeric_values"] = 0

    for k in features.keys():
        if k.endswith("numeric_values"):
            if np.isnan(features[k]):
                features[k] = 0
            features[k] = np.clip(features[k], -MAX_VALUE, MAX_VALUE)
