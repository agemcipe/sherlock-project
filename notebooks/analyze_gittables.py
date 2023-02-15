# %% [markdown]
# # This notebook analyzes the results of the Model trained on the Gittables dataset
# - Load train, val, test datasets (should be preprocessed)
# - Evaluate and analyse the model predictions.

# %%
from ast import literal_eval
from collections import Counter
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


# %% [markdown]
# ## Load datasets for training, validation, testing
# %% 

# TODO: Move to config file
MODEL_ID = "gittables_full"

BASE_DIR_OPT = [
    pathlib.Path("/home/agemcipe/code/hpi_coursework/master_thesis/"),
    pathlib.Path("/home/jonathan.haas/master_thesis/"),
]

BASE_DIR = [p for p in BASE_DIR_OPT if p.exists()][0]
print("BASE_DIR:", BASE_DIR)
assert BASE_DIR.exists()

BASE_DATA_DIR = BASE_DIR / "data"
print("DATA_DIR:", BASE_DATA_DIR)
assert BASE_DATA_DIR.exists()

DATA_DIR = BASE_DATA_DIR / "gittables"
assert DATA_DIR.exists()
# %% 
MLFLOW_EXPERIMENT_NAME = "gittables"
MLFLOW_ARTIFACT_BASE_DIR = BASE_DIR / "outcomes" / "mlflow_artifacts"
MLFLOW_ARTIFACT_DIR = MLFLOW_ARTIFACT_BASE_DIR / MLFLOW_EXPERIMENT_NAME 
print("MLFLOW_ARTIFACT_DIR:", MLFLOW_ARTIFACT_DIR)
MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# %%
def run_analysis(mlflow_client: MlflowClient = None, model_id = None, model=None, X_test=None, y_test=None):
    if mlflow_client is None:
        mlflow_client = MlflowClient()

    _mlflow_active = mlflow.active_run() is not None
    if not _mlflow_active:
        print("No Mlflow run active.")
    else:
        print(f"Mlflow run active: {mlflow.active_run()}")
    if model_id is None:
        model_id = MODEL_ID 

    if X_test is None:
        print("Loading test data...")
        X_test = pd.read_parquet( MLFLOW_ARTIFACT_DIR / f'{model_id}_X_test.parquet')
    if y_test is None:
        print("Loading test labels...")
        y_test = pd.read_parquet( MLFLOW_ARTIFACT_DIR / f'{model_id}_y_test.parquet').values.flatten()

    
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
        
        mlflow.log_artifact(f"../outcomes/confusion_matrix_{model_id}.png", "confusion_matrix.png")

    # plotting confusion matrix
    if _plot_available:
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title(f"Confusion Matrix for {model_id}")
        dist = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=list(classes_short), )
        dist.plot(ax=ax, xticks_rotation='vertical', include_values=False, cmap='Blues', values_format='d')
        _fp = MLFLOW_ARTIFACT_DIR / f"confusion_matrix_{model_id}.png"
        fig.savefig(_fp, dpi=300)
        
        if _mlflow_active:
            mlflow.log_figure(fig, f"confusion_matrix.png")
        
        
if __name__ == "__main__":
    run_analysis()
     
# %% 
#     report = classification_report(y_test, y_pred, output_dict=True)

#     class_scores = list(filter(lambda x: isinstance(x, tuple) and isinstance(x[1], dict) and 'f1-score' in x[1] and x[0] in classes, list(report.items())))

#     class_scores = sorted(class_scores, key=lambda item: item[1]['f1-score'], reverse=True)

# # %% [markdown]
# # ### Top 5 Types

# # %%
# print(f"\t\tf1-score\tprecision\trecall\t\tsupport")

# l = []
# for key, value in class_scores[0:5]:
#     l.append({**{"semantic_type": key}, **value})

# df = pd.DataFrame(l)
# df = df[["semantic_type", "precision", "recall", "f1-score", "support"]]
# df
# # %% [markdown]
# # ### Bottom 5 Types

# # %%
# print(f"\t\tf1-score\tprecision\trecall\t\tsupport")

# l = []
# for key, value in class_scores[-5:]:
#     l.append({**{"semantic_type": key}, **value})

# df = pd.DataFrame(l)
# df = df[["semantic_type", "precision", "recall", "f1-score", "support"]]
# %% [markdown]
# ### All Scores

# %%

# %% [markdown]

# %% [markdown]
# ## Review errors

# # %%
# size = len(y_test)
# mismatches = list()

# for idx, k1 in enumerate(y_test[:size]):
#     k2 = y_pred[idx]

#     if k1 != k2:
#         mismatches.append(k1)
        
#         # zoom in to specific errors. Use the index in the next step
#         if k1 in ('address'):
#             print(f'[{idx}] expected "{k1}" but predicted "{k2}"')
        
# f1 = f1_score(y_test[:size], y_pred[:size], average="weighted")
# print(f'Total mismatches: {len(mismatches)} (F1 score: {f1})')

# data = Counter(mismatches)
# data.most_common()   # Returns all unique items and their counts

# # %%
# test_samples = pd.read_parquet('../data/data/raw/test_values.parquet')

# # %%
# idx = 1001
# original = test_samples.iloc[idx]
# converted = original.apply(literal_eval).to_list()

# print(f'Predicted "{y_pred[idx]}", actual label "{y_test[idx]}". Actual values:\n{converted}')

# # %%


