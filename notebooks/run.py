
# run sherlock on gittabes
from prepare_gittables import main as prepare_gittables
from train_sherlock import main as train_sherlock

MODEL_ID = ["sherlock-full", "sherlock-no-age", "sherlock-small"][-1]

experiment_name = [
    "sherlock-base",
    "test"
][0]
feature_sets = ["char", "word", "par", "rest"]
epochs = 1

X_train, y_train, X_validation, y_validation, X_test, y_test =  prepare_gittables()

for feature in feature_sets:
    print(f"Running sherlock with {feature} features")
    model_id = MODEL_ID + "__" + feature
    model, X_test, y_test = train_sherlock(model_id, experiment_name, X_train, y_train, X_validation, y_validation, X_test, y_test, feature_sets, epochs)