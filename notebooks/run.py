
# run sherlock on gittabes
from prepare_gittables import main as prepare_gittables
from train_sherlock import main as train_sherlock

model_id = ["sherlock-full", "sherlock-no-age", "sherlock-small"][-1]

experiment_name = [
    "sherlock-base",
    "test"
][-1]
feature_sets = ["char", "word", "par", "rest"]
epochs = 1

X_train, y_train, X_validation, y_validation, X_test, y_test =  prepare_gittables()

model, X_test, y_test = train_sherlock(model_id, experiment_name, X_train, y_train, X_validation, y_validation, X_test, y_test, feature_sets, epochs)