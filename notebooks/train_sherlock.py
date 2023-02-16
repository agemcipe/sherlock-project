import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import shutil

from sherlock.deploy.model import SherlockModel

from sherlock.helpers import setup_mlflow, get_mlflow_artifact_dir

EXPERIMENT_NAME = "sherlock-base"
MODEL_ID = "sherlock-base" 

def main(model_id, experiment_name, X_train, y_train, X_validation, y_validation, X_test, y_test):
    setup_mlflow(experiment_name=experiment_name)
    with mlflow.start_run() as run:
        start = datetime.now()

        mlflow_artifact_dir = get_mlflow_artifact_dir(experiment_name, run.info.run_id)
        print(f'Artifacts will be stored at {mlflow_artifact_dir}')

        print(f'Started Model Training at {start}')

        model = SherlockModel()
        # Model will be stored with ID `model_id`
        model.fit(X_train, y_train, X_validation, y_validation, model_id=model_id, active_run = run)

        print('Trained and saved new model.')
        print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

        fp = model.store_weights(model_id=model_id)

        shutil.copyfile(
            fp, mlflow_artifact_dir / f"weights.h5"
            )
        mlflow.log_artifact(mlflow_artifact_dir / f"weights.h5")

        model_output_fp = mlflow_artifact_dir / f"model.json"
        # TODO: log this to mlflow
        with open(model_output_fp, "w") as f_model:
            f_model.write(model.model.to_json())
        mlflow.log_artifact(model_output_fp) 
        return model, X_test, y_test

if __name__ == "__main__":
    main()