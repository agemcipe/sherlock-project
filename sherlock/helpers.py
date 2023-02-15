import os
import zipfile
import pathlib
import gdown
import mlflow



BASE_DIR_OPT = [
    pathlib.Path("/home/agemcipe/code/hpi_coursework/master_thesis/"),
    pathlib.Path("/home/jonathan.haas/master_thesis/"),
]
BASE_DIR = [p for p in BASE_DIR_OPT if p.exists()][0]
BASE_DATA_DIR = BASE_DIR / "data" / "gittables"
DATA_DIR = BASE_DATA_DIR / "gittables"
MODEL_FILES_DIR = pathlib.Path(__file__).parents.parents / "model_files"

    
def setup_mlflow(experiment_name):
    
    mlflow.set_tracking_uri(BASE_DIR / "outcomes" / "mlruns")

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
        print("Created experiment:", experiment_name)
    mlflow.set_experiment(experiment_name)

def get_mlflow_artifact_dir(experiment_name, run_id=None):
    MLFLOW_ARTIFACT_BASE_DIR = BASE_DIR / "outcomes" / "mlflow_artifacts"
    MLFLOW_ARTIFACT_DIR = MLFLOW_ARTIFACT_BASE_DIR / experiment_name 
    if run_id is not None:
        MLFLOW_ARTIFACT_DIR = MLFLOW_ARTIFACT_DIR / run_id

    MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return MLFLOW_ARTIFACT_DIR

def download_data():
    """Download raw and preprocessed data files.
    The data is downloaded from Google Drive and stored in the 'data/' directory.
    """
    data_dir = "../data/data/"
    zip_filepath = "../data/data.zip"
    print(f"Downloading the raw data into {data_dir}.")

    if not os.path.exists(data_dir):
        print("Downloading data directory.")
        gdown.download(
            url="https://drive.google.com/uc?id=1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU",
            output=zip_filepath,
        )

        with zipfile.ZipFile(zip_filepath, "r") as zf:
            zf.extractall("../data/")

    print("Data was downloaded.")

