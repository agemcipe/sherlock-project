

# run sherlock on gittabes
from prepare_gittables import main as prepare_gittables
from train_sherlock import main as train_sherlock

model_id = "sherlock-small"
experiment_name = "sherlock-base"

X_train, y_train, X_validation, y_validation, X_test, y_test =  prepare_gittables()

model, X_test, y_test = train_sherlock(model_id, experiment_name, X_train, y_train, X_validation, y_validation, X_test, y_test)