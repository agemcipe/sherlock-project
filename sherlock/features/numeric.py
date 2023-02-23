# Extract Features from numeric data
import pandas as pd
from collections import OrderedDict
import numpy as np
import re
from typing import List


# %%
NUMBER_REGEX_WITH_UNITS = re.compile(
    r"([-+]?[0-9]*[.]?[0-9]+)(?:[eE][-+]?[0-9]+)?([ a-zA-Z]*)"
)


def match_numeric_regex(array):
    it = map(lambda x: (re.match(NUMBER_REGEX_WITH_UNITS, x)), array)

    return [tuple(map(lambda x: x.strip(), m.groups())) for m in it if m is not None]


# %%
test = ["1232.123 Test", "noMatch", "1232", "9999e-9"]
col_values = ["165285.0", "-247701.0", "79633.0", "0.0"]


# %%
def extract_numeric_features(col_values: List[str], features: OrderedDict):
    # values that can be converted to numeric

    # feature engineering

    # 1. statistics of numeric values
    # pure_numeric_values = pd.to_numeric(col_values, errors="coerce")

    _matches = match_numeric_regex(col_values)
    if _matches:
        pure_numeric_values = pd.to_numeric(
            [m[0] for m in _matches], errors="coerce", downcast="integer"
        )
        units = [m[1] for m in _matches]
    else:
        return
    # TODO deal with units
    # TODO deal with very large numbers
    # TODO deal with ordering of numbers

    MAX_VALUE = 100_000_000  # TODO decide
    pure_numeric_values = np.clip(
        pure_numeric_values[~pd.isna(pure_numeric_values)], -MAX_VALUE, MAX_VALUE
    )  # necessary for model loss to not be nan

    _lin_space = np.linspace(0, 100, 11)
    if len(pure_numeric_values) > 2:
        features["avg_numeric_values"] = np.mean(pure_numeric_values)
        features["std_numeric_values"] = np.std(pure_numeric_values)
        features["min_numeric_values"] = np.min(pure_numeric_values)
        features["max_numeric_values"] = np.max(pure_numeric_values)
        features["median_numeric_values"] = np.median(pure_numeric_values)
        features["sum_numeric_values"] = np.sum(pure_numeric_values)
        features["var_numeric_values"] = np.var(pure_numeric_values)
        # extend this to other percentiles
        for perc_v, perc in zip(
            np.percentile(pure_numeric_values, _lin_space), _lin_space.astype(int)
        ):
            features[f"percentile_{perc}_numeric_values"] = perc_v
    # else:
    #     # TODO: impute with gloabal statistics
    #     features["avg_numeric_values"] = np.nan
    #     features["std_numeric_values"] = np.nan
    #     features["min_numeric_values"] = np.nan
    #     features["max_numeric_values"] = np.nan
    #     features["median_numeric_values"] = np.nan
    #     features["sum_numeric_values"] = np.nan
    #     features["var_numeric_values"] = np.nan
    #     for , perc in zip(
    #         np.percentile(pure_numeric_values, _lin_space), _lin_space.astype(int)
    #     ):
    # 2. statistics of numeric values after removing characters

    for k in features.keys():
        if k.endswith("numeric_values"):
            features[k] = np.clip(features[k], -MAX_VALUE, MAX_VALUE)
