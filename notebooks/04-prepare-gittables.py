# %%
import pandas as pd
import numpy as np
from datetime import datetime
import pathlib
from tqdm.contrib.concurrent import process_map

# %% 
from sherlock.deploy.model import SherlockModel
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings
from sklearn.model_selection import train_test_split

# %%
prepare_feature_extraction()
initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()

# %%
DATA_DIR = pathlib.Path("/home/agemcipe/code/hpi_coursework/master_thesis/semanum/data/gittables")
ont_file = DATA_DIR / "dbpedia_semantic_types_filtered_1000.csv"
index_file = DATA_DIR / "mapping_column_name_semantic_type.csv"

# %%
ont_df = pd.read_csv(ont_file)
index_df = pd.read_csv(index_file)
# %%
index_df = index_df[index_df["dbpedia_semantic"].isin(ont_df["id"])]

# %%
_local_files = [
"allegro_con_spirito_tables_licensed" , 
"beats_per_minute_tables_licensed", 
"solar_constant_tables_licensed", 
"wartime_tables_licensed", 
"attrition_rate_tables_licensed", 
"radial_velocity_tables_licensed", 
"speed_of_light_tables_licensed", 
]

_index_df = index_df[index_df["file_name"].str.startswith(tuple(_local_files))]
# %% 

def _get_data_and_targets(row):
    i = row[0]
    if i % 100 == 0:
        print(i)

    row = row[1]
    _fp = DATA_DIR / "unzipped" / row["file_name"]
    _data = pd.read_parquet(_fp)[row["column_name"]]
    if not _data.isnull().all():
        return str(_data.values.tolist()), row["dbpedia_semantic"]

def get_data_and_targets(index_df: pd.DataFrame, n: int = 1000):
    """Read data from parquet files and return data and targets.

    Does so using multiprocessing.

    Parameters
    ----------
    index_df : pd.DataFrame
        _description_
    n : int, optional
        _description_, by default 1000
    """
    it = index_df.head(n).iterrows
    total = len(list(it()))
    print("Total:", total)

    res = process_map(
        _get_data_and_targets, it(), max_workers=4, total=total
    )

    res = [r for r in res if r is not None]
    data, targets = tuple(zip(*res))
    return list(data), list(targets)


data, targets = get_data_and_targets(_index_df.reset_index(), n=3000)
assert len(data) == len(targets)

print(len(data), len(targets))

raw_data = data
raw_targets = targets
    
# %% 
data = pd.Series(raw_data, name="values")
targets = pd.Series(raw_targets, name="labels")

_targets = targets.value_counts() > 15
_targets = _targets[_targets].index.tolist()
_targets = [
    "http://dbpedia.org/ontology/temperature",
    "http://dbpedia.org/ontology/speedLimit",
    "http://dbpedia.org/ontology/number",
    "http://dbpedia.org/ontology/boilingPoint",
]
idx = targets[targets.isin(_targets)].index
targets = targets[idx]
data = data[idx]

# %% 
feature_file_name = "../gittables_sample.csv"
extract_features_to_csv(output_path=feature_file_name, parquet_values=data)
# extract_features(output_filename=feature_file_name, data=data)
# extract_features_to_csv(output_path=feature_file_name, parquet_values=data)

# %% 
feature_vectors = pd.read_csv(feature_file_name, dtype=np.float32)

# %%
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, targets, test_size=0.1, random_state=41, stratify=targets)
X_train, X_validation, y_train, y_validation = train_test_split(feature_vectors, targets, test_size=0.2, random_state=41, stratify=targets)

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_validation)}")

columns_with_na_values = X_train.columns[X_train.isna().any()]
print("Columns with NA values:", columns_with_na_values)

# %%
# impute NA values with means
train_columns_means = pd.DataFrame(X_train.mean()).transpose()

# %%
X_train = X_train.fillna(train_columns_means.iloc[0])
X_validation = X_validation.fillna(train_columns_means.iloc[0]) # is this right using train mean?

# %%
model_id = 'gittables_sample'
start = datetime.now()
print(f'Started at {start}')

model = SherlockModel()
# Model will be stored with ID `model_id`
model.fit(X_train, y_train, X_validation, y_validation, model_id=model_id)

print('Trained and saved new model.')
print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

model.store_weights(model_id=model_id)
# %%
t = model.predict(X_test, model_id=model_id)
sum(t == y_test) / len(y_test)
# %%
