import os

import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    concatenate,
)
from tensorflow.keras.models import Model, model_from_json

from sherlock.deploy import helpers
from sherlock.functional import IMPLEMENTED_FEATURES


class SherlockModel:
    def __init__(self):
        self.lamb = 0.0001
        self.do = 0.35
        self.lr = 0.0001

        self.model_files_directory = "../model_files/"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train,
        X_val: pd.DataFrame,
        y_val,
        model_id: str,
        active_run: mlflow.ActiveRun = None,
        feature_sets: list = IMPLEMENTED_FEATURES,
        epochs: int = 100,
    ):
        if active_run:
            mlflow.set_tag("model_id", model_id)
            mlflow.tensorflow.autolog()

        if model_id == "sherlock":
            raise ValueError(
                "`model_id` cannot be `sherlock` to avoid overwriting the original model weights."
            )
        num_classes = len(set(y_train))

        encoder = LabelEncoder()
        encoder.fit(y_train)

        feature_cols = helpers.categorize_features(feature_sets)

        features = {"train": {}, "val": {}}

        if "char" in feature_sets:
            features["train"]["char"] = X_train[feature_cols["char"]]
            features["val"]["char"] = X_val[feature_cols["char"]]
        if "word" in feature_sets:
            features["train"]["word"] = X_train[feature_cols["word"]]
            features["val"]["word"] = X_val[feature_cols["word"]]
        if "par" in feature_sets:
            features["train"]["par"] = X_train[feature_cols["par"]]
            features["val"]["par"] = X_val[feature_cols["par"]]
        if "rest" in feature_sets:
            print(feature_cols["rest"])
            features["train"]["rest"] = X_train[feature_cols["rest"]]
            features["val"]["rest"] = X_val[feature_cols["rest"]]
        if "numeric" in feature_sets:
            features["train"]["numeric"] = X_train[feature_cols["numeric"]]
            features["val"]["numeric"] = X_val[feature_cols["numeric"]]

        y_train_int = encoder.transform(y_train)
        y_val_int = encoder.transform(y_val)
        y_train_cat = tf.keras.utils.to_categorical(y_train_int)
        y_val_cat = tf.keras.utils.to_categorical(y_val_int)

        callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

        _models = {}
        if "char" in feature_sets:
            _models["char"] = self._build_char_submodel(
                features["train"]["char"].shape[1]
            )  # tuple
        if "word" in feature_sets:
            _models["word"] = self._build_word_submodel(
                features["train"]["word"].shape[1]
            )
        if "par" in feature_sets:
            _models["par"] = self._build_par_submodel(features["train"]["par"].shape[1])
        if "rest" in feature_sets:
            _models["rest"] = self._build_rest_submodel(
                features["train"]["rest"].shape[1]
            )
        if "numeric" in feature_sets:
            _models["numeric"] = self._build_word_submodel(
                features["train"]["numeric"].shape[1]
            )

        # Merge submodels and build main network

        # merged_model1 = concatenate([char_model, word_model, par_model, rest_model])
        if len(feature_sets) == 1:
            merged_model1 = _models[feature_sets[0]][1]
        else:
            merged_model1 = concatenate(
                [_models[feature][1] for feature in feature_sets]
            )

        merged_model_output = self._add_main_layers(merged_model1, num_classes)

        model = Model(
            [_models[feature][0] for feature in feature_sets],
            merged_model_output,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        if active_run:
            mlflow.log_param(
                "train_cols",
                sum(
                    len(X_train[feature_cols[feature]].columns)
                    for feature in feature_sets
                ),
            )
            mlflow.log_param(
                "val_cols",
                sum(
                    len(X_val[feature_cols[feature]].columns)
                    for feature in feature_sets
                ),
            )

        model.fit(
            [features["train"][feature].values for feature in feature_sets],
            y_train_cat,
            validation_data=(
                [features["val"][feature].values for feature in feature_sets],
                y_val_cat,
            ),
            callbacks=callbacks,
            epochs=epochs,
            batch_size=256,
        )

        self.model = model

        _ = helpers._get_categorical_label_encodings(y_train, y_val, model_id)

    def predict(
        self,
        X: pd.DataFrame,
        model_id: str = "sherlock",
        feature_sets=IMPLEMENTED_FEATURES,
    ) -> np.array:
        """Use sherlock model to generate predictions for X.

        Parameters
        ----------
        X
            Featurized dataframe to generate predictions for.
        model_id
            ID of the model used for generating predictions.

        Returns
        -------
        Array with predictions for X.
        """
        y_pred = self.predict_proba(X, model_id, feature_sets=feature_sets)
        y_pred_classes = helpers._proba_to_classes(y_pred, model_id)

        return y_pred_classes

    def predict_proba(
        self,
        X: pd.DataFrame,
        model_id: str = "sherlock",
        feature_sets=IMPLEMENTED_FEATURES,
    ) -> np.array:
        """Use sherlock model to generate predictions for X.

        Parameters
        ----------
        X
            Featurized data set to generate predictions for.
        model_id
            Identifier of a trained model to use for generating predictions.

        Returns
        -------
        Array with predictions for X.
        """
        feature_cols_dict = helpers.categorize_features(feature_sets=feature_sets)

        y_pred = self.model.predict(
            [X[feature_cols_dict[feature_set]] for feature_set in feature_sets]
        )

        return y_pred

    def initialize_model_from_json(
        self, with_weights: bool, model_id: str = "sherlock"
    ):
        """Load model architecture and populate with pretrained weights.

        Parameters
        ----------
        with_weights
            Whether to populate the model with trained weights.
        model_id
            The ID of the model file to build, defaults to `sherlock` for using the
            sherlock model with the original weights.
        """
        # callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

        model_filename = os.path.join(
            self.model_files_directory, f"{model_id}_model.json"
        )
        if not os.path.exists(model_filename):
            raise ValueError(
                f"""
                No model file associated with this ID: {model_id}, was found.
                The desired model should be specified and stored first before it can be used.
                """
            )

        file = open(model_filename, "r")
        model = model_from_json(file.read())
        file.close()

        if with_weights:
            weights_filename = os.path.join(
                self.model_files_directory, f"{model_id}_weights.h5"
            )
            if not os.path.exists(weights_filename):
                raise ValueError(
                    f"""
                    There are no weights associated with this model ID: {model_id}.
                    The desired model should be trained first before it can be initialized.
                    """
                )
            model.load_weights(weights_filename)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        self.model = model

    def store_weights(self, model_id: str):
        if model_id == "sherlock":
            raise ValueError(
                "Cannot save model weights with `sherlock` model ID. Choose an alternative."
            )

        weights_filename = os.path.join(
            self.model_files_directory, f"{model_id}_weights.h5"
        )

        self.model.save_weights(weights_filename)
        return weights_filename

    def _build_char_submodel(self, char_shape):
        n_weights = 300

        char_model_input = Input(shape=(char_shape,))
        char_model1 = BatchNormalization(axis=1)(char_model_input)
        char_model2 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(char_model1)
        char_model3 = Dropout(self.do)(char_model2)
        char_model4 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(char_model3)

        return char_model_input, char_model4

    def _build_word_submodel(self, word_shape):
        n_weights = 200

        word_model_input = Input(shape=(word_shape,))
        word_model1 = BatchNormalization(axis=1)(word_model_input)
        word_model2 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(word_model1)
        word_model3 = Dropout(self.do)(word_model2)
        word_model4 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(word_model3)

        return word_model_input, word_model4

    def _build_par_submodel(self, par_shape):
        n_weights = 400

        par_model_input = Input(shape=(par_shape,))
        par_model1 = BatchNormalization(axis=1)(par_model_input)
        par_model2 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(par_model1)
        par_model3 = Dropout(self.do)(par_model2)
        par_model4 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(par_model3)

        return par_model_input, par_model4

    def _build_rest_submodel(self, rest_shape):
        # Build submodel for remaining features
        rest_model_input = Input(shape=(rest_shape,))
        rest_model1 = BatchNormalization(axis=1)(rest_model_input)

        return rest_model_input, rest_model1

    def _add_main_layers(self, merged_model1, num_classes):
        n_weights = 500

        merged_model2 = BatchNormalization(axis=1)(merged_model1)
        merged_model3 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(merged_model2)
        merged_model4 = Dropout(self.do)(merged_model3)
        merged_model5 = Dense(
            n_weights,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(merged_model4)
        merged_model_output = Dense(
            num_classes,
            activation=tf.nn.softmax,
            kernel_regularizer=tf.keras.regularizers.l2(self.lamb),
        )(merged_model5)

        return merged_model_output
