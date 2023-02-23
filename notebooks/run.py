# run sherlock on gittabes
from prepare_gittables import main as prepare_gittables
from train_sherlock import main as train_sherlock

MODEL_ID = ["sherlock-full", "sherlock-no-age", "sherlock-small"][0]

experiment_name = ["sherlock-base", "test"][0]
feature_sets = ["char", "word", "par", "rest", "numeric"]
epochs = 100

X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_gittables(
    feature_sets, recalculate_feature_set=[]
)
feature_set_old = ["char", "word", "par", "rest"]

for _model in ["sherlock-full"]:
    for feature_set in [feature_sets, ["numeric"]]:
        if feature_set == feature_sets:
            _name = "all"
        elif feature_set == feature_set_old:
            _name = "all-no-numeric"
        else:
            _name = "_".join(feature_set)

        model_id = _model + "__" + _name

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
            feature_set,
            epochs,
        )
