# %%
import pandas as pd
import numpy as np
import pathlib
from sherlock.functional import extract_features_to_csv


# %%
processed_data_dir = pathlib.Path(
    "/home/agemcipe/code/hpi_coursework/master_thesis/data/gittables/processed/full"
)
data_fp = processed_data_dir / "data.csv"

data = pd.read_csv(data_fp, nrows=1000)["values"].astype(str)

recalculate_feature_set = ["numeric"]

_name = "_".join(recalculate_feature_set)

_fp = processed_data_dir / "numeric_features_test.csv"

extract_features_to_csv(
    output_path=str(_fp),
    parquet_values=data,
    feature_set=recalculate_feature_set,
)
