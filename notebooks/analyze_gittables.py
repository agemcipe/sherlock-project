# %% [markdown]
# # This notebook analyzes the results of the Model trained on the Gittables dataset
# - Load train, val, test datasets (should be preprocessed)
# - Evaluate and analyse the model predictions.

# %%
from datetime import datetime
import pathlib
import numpy as np
import pandas as pd
import mlflow

if mlflow.__version__ == '1.23.1':
    print("mlflow.__version__ is 1.23.1.is used...")
    from mlflow.tracking import MlflowClient
else:
    from mlflow.client import MlflowClient

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score 
from sherlock.deploy.model import SherlockModel

try:
    import matplotlib
    matplotlib.use("AGG")
    import matplotlib.pyplot as plt
    _plot_available = True
except:
    print("matplotlib with AGG backend not available. Plotting disabled.")
    _plot_available = False
    
from sherlock.helpers import get_mlflow_artifact_dir


# %%
def run_analysis(model_id, experiment_name, mlflow_client: MlflowClient = None, model=None, X_test=None, y_test=None):

    mlflow_artifact_dir = get_mlflow_artifact_dir(experiment_name, mlflow.active_run().info.run_id)

    if mlflow_client is None:
        mlflow_client = MlflowClient()

    _mlflow_active = mlflow.active_run() is not None
    if not _mlflow_active:
        print("No Mlflow run active.")
    else:
        print(f"Mlflow run active: {mlflow.active_run()}")

    if X_test is None:
        print("Loading test data...")
        X_test = pd.read_parquet( mlflow_artifact_dir / f'{model_id}_X_test.parquet')
    if y_test is None:
        print("Loading test labels...")
        y_test = pd.read_parquet( mlflow_artifact_dir / f'{model_id}_y_test.parquet').values.flatten()

    
    if model is None:
        start = datetime.now()
        print(f'Loading Model Started at {start}')

        model = SherlockModel()
        model.initialize_model_from_json(with_weights=True, model_id=model_id)

        print('Initialized model.')
        print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

    print("Predicting on test set...")
    y_pred = model.predict(X_test, model_id=model_id)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    print("Test accuracy:", test_acc)
    print("Test f1-Score:", test_f1)

    if _mlflow_active:
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred, average="weighted"))

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    # log classification report
    if _mlflow_active:
        mlflow.log_text(classification_report(y_test, y_pred, digits=3), "classification_report.txt")

    # load classes
    print("Creating Confusion Matrix")
    classes = np.load(f"../model_files/classes_{model_id}.npy", allow_pickle=True) # TODO: make this more robust
    classes_short = [s.replace("http://dbpedia.org/ontology/", "") for s in classes]
    cfm = confusion_matrix(y_test , y_pred, labels=classes, normalize='true')
    cfm_df = pd.DataFrame(cfm, index=classes_short, columns=classes_short)

    # log confusion matrix
    if _mlflow_active:
        print("Logging Confusion Matrix")
        cfm_fp = f"../outcomes/confusion_matrix_{model_id}.csv"
        cfm_df.to_csv(cfm_fp, index=True, header=True)
        
        mlflow.log_artifact(f"../outcomes/confusion_matrix_{model_id}.csv", "confusion_matrix.png")

    # plotting confusion matrix
    if _plot_available:
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title(f"Confusion Matrix for {model_id}")
        dist = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=list(classes_short), )
        dist.plot(ax=ax, xticks_rotation='vertical', include_values=False, cmap='Blues', values_format='d')
        _fp = mlflow_artifact_dir / f"confusion_matrix_{model_id}.png"
        fig.savefig(_fp, dpi=300)
        
        if _mlflow_active:
            mlflow.log_figure(fig, f"confusion_matrix.png")
        
