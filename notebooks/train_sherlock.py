import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import shutil

from sherlock.deploy.model import SherlockModel

from sherlock.helpers import setup_mlflow, get_mlflow_artifact_dir

from analyze_gittables import run_analysis
EXPERIMENT_NAME = "sherlock-base"
MODEL_ID = "sherlock-no-age" 

def main(model_id, experiment_name, X_train, y_train, X_validation, y_validation, X_test, y_test):
    setup_mlflow(experiment_name=experiment_name)
    with mlflow.start_run() as run:
        start = datetime.now()

        mlflow_artifact_dir = get_mlflow_artifact_dir(experiment_name, run.info.run_id)
        print(f'Artifacts will be stored at {mlflow_artifact_dir}')

        print(f'Started Model Training at {start}')

        model = SherlockModel()
        # Model will be stored with ID `model_id`
        # let's filter for age type
        if model_id in ["sherlock-no-age", "sherlock-small"]:
            if model_id == "sherlock-no-age":
                print("Removing age type...")
                _y_train = pd.Series(y_train)
                allowed_classes = _y_train[_y_train != "http://dbpedia.org/ontology/age"].unique()
            elif model_id == "sherlock-small":
                print("Using small dataset...")
                allowed_classes = pd.Series(y_train).value_counts().index[:10]
            else:
                raise ValueError(f"Unknown model_id {model_id}")
            
            train_idx = pd.Series(y_train).isin(allowed_classes)
            val_idx = pd.Series(y_validation).isin(allowed_classes)
            test_idx = pd.Series(y_test).isin(allowed_classes)
            print(len(train_idx), len(val_idx), len(test_idx))
            print(len(X_train))
            X_train = X_train.loc[train_idx]
            y_train = y_train[train_idx]
            X_validation = X_validation.loc[val_idx]
            y_validation = y_validation[val_idx]
            X_test = X_test.loc[test_idx]
            y_test = y_test[test_idx]
        
            
        print("train_rows", X_train.shape[0])
        print("train_cols", X_train.shape[1])
        print("train_classes", len(np.unique(y_train)))
        print("validation_rows", X_validation.shape[0])
        print("validation_cols", X_validation.shape[1])
        print("validation_classes", len(np.unique(y_validation)))
        print("test_rows", X_test.shape[0])
        print("test_cols", X_test.shape[1])
        print("test_classes", len(np.unique(y_test)))
            # log parameters

        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("train_cols", X_train.shape[1])
        mlflow.log_param("train_classes", len(np.unique(y_train)))
        mlflow.log_param("validation_rows", X_validation.shape[0])
        mlflow.log_param("validation_cols", X_validation.shape[1])
        mlflow.log_param("validation_classes", len(np.unique(y_validation)))
        mlflow.log_param("test_rows", X_test.shape[0])
        mlflow.log_param("test_cols", X_test.shape[1])
        mlflow.log_param("test_classes", len(np.unique(y_test)))

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
        run_analysis(model_id, experiment_name, model = model, X_test = X_test, y_test = y_test) 
        return model, X_test, y_test

if __name__ == "__main__":
    main()