# %%
import pandas as pd
import numpy as np
from datetime import datetime
import pathlib
from tqdm.contrib.concurrent import process_map
import tqdm

# %%
from sherlock.functional import extract_features_to_csv, IMPLEMENTED_FEATURES
from sherlock.features.paragraph_vectors import (
    initialise_pretrained_model,
    initialise_nltk,
)
from sherlock.features.preprocessing import (
    prepare_feature_extraction,
)
from sherlock.features.word_embeddings import initialise_word_embeddings
from sklearn.model_selection import train_test_split

from sherlock.helpers import DATA_DIR


# %%
PROCESSED_DATA_ID = "full_numeric_features"


def get_processed_data_dir(
    base_path: pathlib.Path = DATA_DIR, processed_data_id: str = PROCESSED_DATA_ID
):
    PROCESSED_DATA_DIR = base_path / "processed" / processed_data_id
    if PROCESSED_DATA_DIR.exists():
        print(f"Warning. {PROCESSED_DATA_DIR} already exists. Overwriting.")
    else:
        PROCESSED_DATA_DIR.mkdir(parents=True)
    return PROCESSED_DATA_DIR


# %%
LEAST_TARGET_COUNT = 100  # Should not have an effect on the results for full dataset


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


def split_data(data, targets, test_size=0.1, random_state=41, store_fp=None):
    """Split data and targets into train and test sets.

    Parameters
    ----------
    data : list
        _description_
    targets : list
        _description_
    test_size : float, optional
        _description_, by default 0.2
    random_state : int, optional
        _description_, by default 42

    Returns
    -------
    tuple
        _description_
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=test_size, random_state=random_state, stratify=targets
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_validation)}")
    print(f"Distinct labels in train: {len(np.unique(y_train))}")

    columns_with_na_values = X_train.columns[X_train.isna().any()]
    print("Columns with NA values:", columns_with_na_values)
    # impute NA values with means
    train_columns_means = pd.DataFrame(X_train.mean()).transpose()

    print("Imputing NA values with means...")
    X_train = X_train.fillna(train_columns_means.iloc[0])
    X_validation = X_validation.fillna(
        train_columns_means.iloc[0]
    )  # is this right using train mean?

    # store data as .parquet files
    if store_fp is not None and pathlib.Path(store_fp).exists():
        print("Storing data as parquet files...")
        x_train_fp = store_fp / f"X_train.parquet"
        y_train_fp = store_fp / f"y_train.parquet"
        x_validation_fp = store_fp / f"X_validation.parquet"
        y_validation_fp = store_fp / f"y_validation.parquet"
        x_test_fp = store_fp / f"X_test.parquet"
        y_test_fp = store_fp / f"y_test.parquet"

        for data, fp in zip(
            [X_train, y_train, X_validation, y_validation, X_test, y_test],
            [
                x_train_fp,
                y_train_fp,
                x_validation_fp,
                y_validation_fp,
                x_test_fp,
                y_test_fp,
            ],
        ):
            if fp.exists():
                print("Warning", fp, "already exists. Deleting...")
                fp.unlink()
            if isinstance(data, pd.DataFrame):
                data.to_parquet(fp, engine="pyarrow", compression="snappy")
            else:
                pd.DataFrame(data, columns=["label"]).to_parquet(
                    fp, engine="pyarrow", compression="snappy"
                )

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def main(feature_set=IMPLEMENTED_FEATURES, recalculate_feature_set=[]):
    prepare_feature_extraction()
    initialise_word_embeddings()
    initialise_pretrained_model(400)
    initialise_nltk()

    ont_file = DATA_DIR / "dbpedia_semantic_types_filtered_1000.csv"
    index_file = DATA_DIR / "mapping_column_name_semantic_type.csv"
    processed_data_dir = get_processed_data_dir()

    resulting_files = [
        processed_data_dir / f"X_train.parquet",
        processed_data_dir / f"y_train.parquet",
        processed_data_dir / f"X_validation.parquet",
        processed_data_dir / f"y_validation.parquet",
        processed_data_dir / f"X_test.parquet",
        processed_data_dir / f"y_test.parquet",
    ]

    if not recalculate_feature_set and all([p.exists() for p in resulting_files]):
        print("Data already processed. Skipping...")
        return [
            pd.read_parquet(resulting_files[0]),
            pd.read_parquet(resulting_files[1])["label"].values,
            pd.read_parquet(resulting_files[2]),
            pd.read_parquet(resulting_files[3])["label"].values,
            pd.read_parquet(resulting_files[4]),
            pd.read_parquet(resulting_files[5])["label"].values,
        ]

    ont_df = pd.read_csv(ont_file)
    index_df = pd.read_csv(index_file)
    index_df = index_df[index_df["dbpedia_semantic"].isin(ont_df["id"])]

    _local_dirs = [d.name for d in (DATA_DIR / "unzipped").glob("*")]
    if _local_dirs:
        _index_df = index_df[index_df["file_name"].str.startswith(tuple(_local_dirs))]
    else:
        print("No local dirs found")
        raise FileNotFoundError()

    data_fp = processed_data_dir / f"data.csv"
    targets_fp = processed_data_dir / f"targets.csv"

    BASE_FEATURES_FILE_NAME = "features{batch_id}.csv"
    BASE_FEATURES_FILE_PATH = processed_data_dir / BASE_FEATURES_FILE_NAME.format(
        batch_id=""
    )
    batch_size = 10000

    if recalculate_feature_set:
        if BASE_FEATURES_FILE_PATH.exists() and data_fp.exists():
            print("Recalculating features", recalculate_feature_set)
            data = pd.read_csv(data_fp)["values"].astype(str)

            _name = "_".join(recalculate_feature_set)
            _fp = processed_data_dir / BASE_FEATURES_FILE_NAME.format(
                batch_id=f"_{_name}"
            )

            if _fp.exists():
                print("Warning", _fp, "already exists. Overwriting...")

            extract_features_to_csv(
                output_path=str(_fp),
                parquet_values=data,
                feature_set=recalculate_feature_set,
            )
            _len_data = len(data)
            del data

            print("Done Calculating new features")
            print("Merging new features with old features")
            new_feature_vectors = pd.read_csv(
                str(_fp), dtype=np.float32, skip_blank_lines=False
            )
            feature_vectors = pd.read_csv(
                str(BASE_FEATURES_FILE_PATH), dtype=np.float32
            )

            _new_cols = [
                c
                for c in new_feature_vectors.columns
                if c not in feature_vectors.columns
            ]
            print("New Columns", _new_cols)
            _overwrite_cols = [
                c for c in new_feature_vectors.columns if c in feature_vectors.columns
            ]
            print("Existing Columns (will be overwritten)", _overwrite_cols)

            assert _len_data == len(new_feature_vectors) == len(feature_vectors)
            feature_vectors[_overwrite_cols] = new_feature_vectors[_overwrite_cols]
            feature_vectors = pd.concat(
                [feature_vectors, new_feature_vectors[_new_cols]], axis=1
            )

            print("Writing new features to", BASE_FEATURES_FILE_PATH)
            feature_vectors.to_csv(
                str(BASE_FEATURES_FILE_PATH),
                index=False,
            )

    if not BASE_FEATURES_FILE_PATH.exists():
        if data_fp.exists() and targets_fp.exists():
            print("Loading data and targets from parquet files...")
            data = pd.read_csv(data_fp)["values"].astype(str)
            targets = pd.read_csv(targets_fp)
        else:
            print("Loading data and targets from individual parquet files...")
            data, targets = get_data_and_targets(_index_df.reset_index(), n=100_000_000)

            data = pd.Series(data, name="values")
            targets = pd.Series(targets, name="labels")

            targets_fil_count = targets.value_counts()[
                targets.value_counts() > LEAST_TARGET_COUNT
            ].index

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
            print("Extracting features for batch", i, "to", i + batch_size)
            _fp = processed_data_dir / BASE_FEATURES_FILE_NAME.format(batch_id=f"_{i}")
            data_batch = data[i : i + batch_size]
            extract_features_to_csv(
                output_path=str(_fp), parquet_values=data_batch, feature_set=feature_set
            )

        # concat all batches
        print("Concatenating batches")
        with open(BASE_FEATURES_FILE_PATH, "w") as outfile:
            for i in tqdm.tqdm(range(0, len(data), batch_size)):
                _fp = processed_data_dir / BASE_FEATURES_FILE_NAME.format(
                    batch_id=f"_{i}"
                )
                if i == 0:
                    with open(_fp, "r") as infile:
                        outfile.write(infile.read())
                else:
                    with open(_fp, "r") as infile:
                        next(infile)  # skip header
                        outfile.write(infile.read())
                # remove batch file
                _fp.unlink()

    targets = pd.read_csv(targets_fp)["labels"].values
    feature_vectors = pd.read_csv(str(BASE_FEATURES_FILE_PATH), dtype=np.float32)

    print("Length of feature vectors:", len(feature_vectors))
    print("Length of targets:", len(targets))

    X_train, y_train, X_validation, y_validation, X_test, y_test = split_data(
        feature_vectors, targets, store_fp=processed_data_dir
    )
    return X_train, y_train, X_validation, y_validation, X_test, y_test


# %%
if __name__ == "__main__":
    main()
