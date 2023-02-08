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

from sklearn.metrics import f1_score, classification_report

from sherlock.deploy.model import SherlockModel

# %% [markdown]
# ## Load datasets for training, validation, testing
# %% 
DATA_DIR = [
    pathlib.Path("/home/agemcipe/code/hpi_coursework/master_thesis/semanum/data/gittables"),
    pathlib.Path("/home/jonathan.haas/gittables"),
][1]
MODEL_ID = "gittables_full"

# %%
start = datetime.now()
print(f'Started at {start}')

# X_train = pd.read_parquet( DATA_DIR / f"{MODEL_ID}_X_train.parquet")
# y_train = pd.read_parquet( DATA_DIR / f"{MODEL_ID}_y_train.parquet").values.flatten() 
# y_train = np.array([x.lower() for x in y_train])

print(f'Load data (train) process took {datetime.now() - start} seconds.')

# print('Distinct types for columns in the Dataframe (should be all float32):')
# print(set(X_train.dtypes))

# %%
start = datetime.now()
print(f'Started at {start}')

X_validation = pd.read_parquet(DATA_DIR / f'{MODEL_ID}_X_validation.parquet')
y_validation = pd.read_parquet(DATA_DIR / f'{MODEL_ID}_y_validation.parquet').values.flatten()

# y_validation = np.array([x.lower() for x in y_validation])

print(f'Load data (validation) process took {datetime.now() - start} seconds.')

# %%
start = datetime.now()
print(f'Started at {start}')

X_test = pd.read_parquet(DATA_DIR / f'{MODEL_ID}_X_test.parquet')
y_test = pd.read_parquet(DATA_DIR / f'{MODEL_ID}_y_test.parquet').values.flatten()

# y_test = np.array([x.lower() for x in y_test])

print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

# %% [markdown]
# ## Initialize the model with pretrained weights

# %% [markdown]
# ### Option 1: load Sherlock with pretrained weights

# %%
start = datetime.now()
print(f'Started at {start}')

model = SherlockModel();
model.initialize_model_from_json(with_weights=True, model_id=MODEL_ID);

print('Initialized model.')
print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

# %% [markdown]
# ### Make prediction

# %%
predicted_labels = model.predict(X_test)
# predicted_labels = np.array([x.lower() for x in predicted_labels])

# %%
print(f'prediction count {len(predicted_labels)}, type = {type(predicted_labels)}')

size=len(y_test)

# Should be fully deterministic too.
f1_score(y_test[:size], predicted_labels[:size], average="weighted")

# %%
# If using the original model, model_id should be replaced with "sherlock"
#model_id = "sherlock"
classes = np.load(f"../model_files/classes_{MODEL_ID}.npy", allow_pickle=True)

report = classification_report(y_test, predicted_labels, output_dict=True)

class_scores = list(filter(lambda x: isinstance(x, tuple) and isinstance(x[1], dict) and 'f1-score' in x[1] and x[0] in classes, list(report.items())))

class_scores = sorted(class_scores, key=lambda item: item[1]['f1-score'], reverse=True)

# %% [markdown]
# ### Top 5 Types

# %%
print(f"\t\tf1-score\tprecision\trecall\t\tsupport")

for key, value in class_scores[0:5]:
    if len(key) >= 8:
        tabs = '\t' * 1
    else:
        tabs = '\t' * 2

    print(f"{key}{tabs}{value['f1-score']:.3f}\t\t{value['precision']:.3f}\t\t{value['recall']:.3f}\t\t{value['support']}")

# %% [markdown]
# ### Bottom 5 Types

# %%
print(f"\t\tf1-score\tprecision\trecall\t\tsupport")

for key, value in class_scores[len(class_scores)-5:len(class_scores)]:
    if len(key) >= 8:
        tabs = '\t' * 1
    else:
        tabs = '\t' * 2

    print(f"{key}{tabs}{value['f1-score']:.3f}\t\t{value['precision']:.3f}\t\t{value['recall']:.3f}\t\t{value['support']}")

# %% [markdown]
# ### All Scores

# %%
print(classification_report(y_test, predicted_labels, digits=3))

# %% [markdown]
# ## Review errors

# %%
size = len(y_test)
mismatches = list()

for idx, k1 in enumerate(y_test[:size]):
    k2 = predicted_labels[idx]

    if k1 != k2:
        mismatches.append(k1)
        
        # zoom in to specific errors. Use the index in the next step
        if k1 in ('address'):
            print(f'[{idx}] expected "{k1}" but predicted "{k2}"')
        
f1 = f1_score(y_test[:size], predicted_labels[:size], average="weighted")
print(f'Total mismatches: {len(mismatches)} (F1 score: {f1})')

data = Counter(mismatches)
data.most_common()   # Returns all unique items and their counts

# %%
test_samples = pd.read_parquet('../data/data/raw/test_values.parquet')

# %%
idx = 1001
original = test_samples.iloc[idx]
converted = original.apply(literal_eval).to_list()

print(f'Predicted "{predicted_labels[idx]}", actual label "{y_test[idx]}". Actual values:\n{converted}')

# %%


