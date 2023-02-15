# %% [markdown]
# # This notebook analyzes the results of the Model trained on the Gittables dataset
# - Load train, val, test datasets (should be preprocessed)
# - Evaluate and analyse the model predictions.

# %%
from ast import literal_eval
from collections import Counter
from datetime import datetime
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score 

from sherlock.deploy.model import SherlockModel

# %% [markdown]
# ## Load datasets for training, validation, testing
# %% 
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
start = datetime.now()
print(f'Started at {start}')

X_test = pd.read_parquet( MLFLOW_ARTIFACT_DIR / f'{MODEL_ID}_X_test.parquet')
y_test = pd.read_parquet( MLFLOW_ARTIFACT_DIR / f'{MODEL_ID}_y_test.parquet').values.flatten()

# y_test = np.array([x.lower() for x in y_test])

print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

# %% [markdown]
# ## Initialize the model with pretrained weights

# %% [markdown]
# ### Load Sherlock with pretrained weights

# %%
start = datetime.now()
print(f'Started at {start}')

model = SherlockModel()
model.initialize_model_from_json(with_weights=True, model_id=MODEL_ID);

print('Initialized model.')
print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

# %% [markdown]
# ### Make prediction

# %%
y_pred = model.predict(X_test, model_id=MODEL_ID)
test_acc = accuracy_score(y_test, y_pred)
# predicted_labels = np.array([x.lower() for x in predicted_labels])
# %% 
print(f'prediction count {len(y_pred)}, type = {type(y_pred)}')
# Should be fully deterministic too.
print("Test accuracy:", test_acc)
print("Test f1-Score:", f1_score(y_test, y_pred, average="weighted"))

# %%
# If using the original model, model_id should be replaced with "sherlock"
#model_id = "sherlock"
classes = np.load(f"../model_files/classes_{MODEL_ID}.npy", allow_pickle=True)

report = classification_report(y_test, y_pred, output_dict=True)

class_scores = list(filter(lambda x: isinstance(x, tuple) and isinstance(x[1], dict) and 'f1-score' in x[1] and x[0] in classes, list(report.items())))

class_scores = sorted(class_scores, key=lambda item: item[1]['f1-score'], reverse=True)

# %% [markdown]
# ### Top 5 Types

# %%
print(f"\t\tf1-score\tprecision\trecall\t\tsupport")

l = []
for key, value in class_scores[0:5]:
    l.append({**{"semantic_type": key}, **value})

df = pd.DataFrame(l)
df = df[["semantic_type", "precision", "recall", "f1-score", "support"]]
df
# %% [markdown]
# ### Bottom 5 Types

# %%
print(f"\t\tf1-score\tprecision\trecall\t\tsupport")

l = []
for key, value in class_scores[-5:]:
    l.append({**{"semantic_type": key}, **value})

df = pd.DataFrame(l)
df = df[["semantic_type", "precision", "recall", "f1-score", "support"]]
df 
# %% [markdown]
# ### All Scores

# %%
print(classification_report(y_test, y_pred, digits=3))

# %% [markdown]
# ## Review errors
fig, ax = plt.subplots(figsize=(20, 20))
classes_short = [s.replace("http://dbpedia.org/ontology/", "") for s in classes]
classes_short = classes
cfm = confusion_matrix(y_test , y_pred, labels=list(classes_short), normalize='true')
# _cfm = (cfm > 0).astype(int) 
_cfm = cfm
dist = ConfusionMatrixDisplay(confusion_matrix=_cfm, display_labels=list(classes_short), )
dist.plot(ax=ax, xticks_rotation='vertical', include_values=False, cmap='Blues', values_format='d')

# %% [markdown]
# ## Review errors

# %%
size = len(y_test)
mismatches = list()

for idx, k1 in enumerate(y_test[:size]):
    k2 = y_pred[idx]

    if k1 != k2:
        mismatches.append(k1)
        
        # zoom in to specific errors. Use the index in the next step
        if k1 in ('address'):
            print(f'[{idx}] expected "{k1}" but predicted "{k2}"')
        
f1 = f1_score(y_test[:size], y_pred[:size], average="weighted")
print(f'Total mismatches: {len(mismatches)} (F1 score: {f1})')

data = Counter(mismatches)
data.most_common()   # Returns all unique items and their counts

# %%
test_samples = pd.read_parquet('../data/data/raw/test_values.parquet')

# %%
idx = 1001
original = test_samples.iloc[idx]
converted = original.apply(literal_eval).to_list()

print(f'Predicted "{y_pred[idx]}", actual label "{y_test[idx]}". Actual values:\n{converted}')

# %%


