
from sherlock.functional import extract_features_to_csv 
import pathlib
import pandas as pd


DATA_DIR = pathlib.Path("/home/agemcipe/code/hpi_coursework/master_thesis/data/gittables")
FP = DATA_DIR / "data.csv"
output_path = DATA_DIR / "test_output.parquet"
data = pd.read_csv(FP, nrows=10)["values"].astype(str)

extract_features_to_csv(output_path=str(output_path), parquet_values=data)