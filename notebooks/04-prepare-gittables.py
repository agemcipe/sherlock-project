# %%
import pandas as pd
import numpy as np
from datetime import datetime
import pathlib
from tqdm.contrib.concurrent import process_map
import tqdm

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
DATA_DIR = [
    pathlib.Path("/home/agemcipe/code/hpi_coursework/master_thesis/semanum/data/gittables"),
    pathlib.Path("/home/jonathan.haas/master_thesis/data/gittables"),
][1]
MODEL_ID = "gittables_full"
LEAST_TARGET_COUNT = 100 # Should not have an effect on the results for full dataset
BASE_FEATURES_FILE_NAME = "{model_id}_features{batch_id}.csv"

# %% 
def _get_data_and_targets(row):
    row = row[1]
    _fp = DATA_DIR / "unzipped" / row["file_name"]
    try:
        _data = pd.read_parquet(_fp, engine="fastparquet")[row["column_name"]]
        if not _data.isnull().all():
            return str(_data.values.tolist()), row["dbpedia_semantic"]
    except Exception as e:
        print("Exception while reading parquet file: ", _fp)
        print(e)
        return None

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
        _get_data_and_targets, it(), total=total, max_workers=16, chunksize=1000
    )

    res = [r for r in res if r is not None]
    data, targets = tuple(zip(*res))
    return list(data), list(targets)

def main():

    ont_file = DATA_DIR / "dbpedia_semantic_types_filtered_1000.csv"
    index_file = DATA_DIR / "mapping_column_name_semantic_type.csv"

    # %%
    ont_df = pd.read_csv(ont_file)
    index_df = pd.read_csv(index_file)
    # %%
    index_df = index_df[index_df["dbpedia_semantic"].isin(ont_df["id"])]

    # %%
    _local_dirs = [d.name for d in  (DATA_DIR / "unzipped").glob("*")]
    if _local_dirs:
        _index_df = index_df[index_df["file_name"].str.startswith(tuple(_local_dirs))]
    else:
        print("No local dirs found")
        raise FileNotFoundError()


    data_fp = DATA_DIR / f"{MODEL_ID}_data.csv"
    targets_fp = DATA_DIR / f"{MODEL_ID}_targets.csv"

    BASE_FEATURES_FILE_PATH = DATA_DIR / BASE_FEATURES_FILE_NAME.format(
        model_id=MODEL_ID, batch_id="")
    batch_size = 10000
    
    # TODO: make sure to load targets as well
    if not BASE_FEATURES_FILE_PATH.exists():
        if data_fp.exists() and targets_fp.exists():
            print("Loading data and targets from parquet files...")
            data = pd.read_csv(data_fp)["values"].astype(str)
            targets = pd.read_csv(targets_fp)
        else: 
            print("Loading data and targets from individual parquet files...")
            data, targets = get_data_and_targets(_index_df.reset_index(), n = 100_000_000)
            
            data = pd.Series(data, name="values")
            targets = pd.Series(targets, name="labels")

            targets_fil_count = targets.value_counts()[targets.value_counts() > LEAST_TARGET_COUNT].index

            idx = targets[targets.isin(targets_fil_count)].index
            targets = targets[idx]
            data = data[idx]


            data.to_csv(data_fp, index=False)
            targets.to_csv(targets_fp, index=False)

        assert len(data) == len(targets)

        print("Finished loading data and targets")
        print("Number of Rows", len(data))
    
        # batching is necessary because of memory constraints
        print(f"Extracting features in batches of {batch_size}")
        for i in range(0, len(data), batch_size):
            print("Extracting features for batch", i, "to", i+batch_size)
            _fp = DATA_DIR / BASE_FEATURES_FILE_NAME.format(model_id=MODEL_ID, batch_id=f"_{i}")
            data_batch = data[i:i+batch_size]
            extract_features_to_csv(output_path=str(_fp), parquet_values=data_batch)
        
        # concat all batches
        print("Concatenating batches")
        with open(BASE_FEATURES_FILE_PATH, "w") as outfile:
            for i in tqdm.tqdm(range(0, len(data), batch_size)):
                _fp = DATA_DIR / BASE_FEATURES_FILE_NAME.format(model_id=MODEL_ID, batch_id=f"_{i}")
                if i == 0: 
                    with open(_fp, "r") as infile:
                        outfile.write(infile.read())
                else:
                    with open(_fp, "r") as infile:
                        next(infile)
                        outfile.write(infile.read())
    
    # %% 
    targets = pd.read_csv(targets_fp)["labels"].values
    feature_vectors = pd.read_csv(str(BASE_FEATURES_FILE_PATH), dtype=np.float32)

    print("Length of feature vectors:", len(feature_vectors))
    print("Length of targets:", len(targets))
    
    # %%
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, targets, test_size=0.1, random_state=41, stratify=targets)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=41, stratify=y_train)

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_validation)}")
    print(f"Distinct labels in train: {len(np.unique(y_train))}")


    # %%
    columns_with_na_values = X_train.columns[X_train.isna().any()]
    print("Columns with NA values:", columns_with_na_values)
    # impute NA values with means
    train_columns_means = pd.DataFrame(X_train.mean()).transpose()

    print("Imputing NA values with means...")
    X_train = X_train.fillna(train_columns_means.iloc[0])
    X_validation = X_validation.fillna(train_columns_means.iloc[0]) # is this right using train mean?

    # %% 
    # store data as .parquet files
    print("Storing data as parquet files...")
    X_train.to_parquet(DATA_DIR / f'{MODEL_ID}_X_train.parquet', engine='pyarrow', compression='snappy')
    pd.DataFrame(y_train, columns=["label"]).to_parquet(DATA_DIR / f'{MODEL_ID}_y_train.parquet', engine='pyarrow', compression='snappy')

    X_validation.to_parquet(DATA_DIR / f'{MODEL_ID}_X_validation.parquet', engine='pyarrow', compression='snappy')
    pd.DataFrame(y_validation, columns=["label"]).to_parquet(DATA_DIR / f'{MODEL_ID}_y_validation.parquet', engine='pyarrow', compression='snappy')

    X_test.to_parquet(DATA_DIR / f'{MODEL_ID}_X_test.parquet', engine='pyarrow', compression='snappy')
    pd.DataFrame(y_test, columns=["label"]).to_parquet(DATA_DIR / f'{MODEL_ID}_y_test.parquet', engine='pyarrow', compression='snappy')


    # %%
    start = datetime.now()
    print(f'Started Model Training at {start}')

    model = SherlockModel()
    # Model will be stored with ID `model_id`
    model.fit(X_train, y_train, X_validation, y_validation, model_id=MODEL_ID)

    print('Trained and saved new model.')
    print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

    model.store_weights(model_id=MODEL_ID)
    # %%
    t = model.predict(X_test, model_id=MODEL_ID)
    print("Test Acc.", sum(t == y_test) / len(y_test))

# %%
if __name__ == "__main__":
    prepare_feature_extraction()
    initialise_word_embeddings()
    initialise_pretrained_model(400)
    initialise_nltk()
    main()