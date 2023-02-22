# run sherlock on gittabes
from prepare_gittables import main as prepare_gittables
from train_sherlock import main as train_sherlock

MODEL_ID = ["sherlock-full", "sherlock-no-age", "sherlock-small"][0]

experiment_name = ["sherlock-base", "test"][0]
feature_sets = ["char", "word", "par", "rest", "numeric"]
epochs = 1

X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_gittables(
    feature_sets, recalculate_feature_set=["numeric"]
)

model_id = MODEL_ID + "__" + "all"
print(model_id)
model, X_test, y_test = train_sherlock(
    model_id,
    experiment_name,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    feature_sets,
    epochs,
)
