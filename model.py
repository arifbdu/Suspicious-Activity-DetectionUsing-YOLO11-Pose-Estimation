# ============================================================
# model.py
#
# Reviewer #1 revision experiments
#
# Comment 2:
#   Evaluate whether feature selection affects performance.
#
# Comment 3:
#   Report model configurations and hyperparameters.
#   Add a nonlinear MLP classifier to the comparison.
#
# Comment 4:
#   Tune the KNN k value.
#   Compare standard KNN with distance-weighted KNN and
#   an Average-distance KNN variant.
#
# IMPORTANT:
#   Suspicious activity is mapped to class 1.
#   Therefore, precision, recall, F1-score, and AUC describe
#   suspicious-activity detection.
# ============================================================

from pathlib import Path
from functools import partial
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
)


# ============================================================
# Exact file paths
# ============================================================

BASE_DIR = Path(r"F:\1. RESEARCH\Ashiq Sir\class - 2")

DATASET_PATH = (
    BASE_DIR
    / "dataset_path"
    / "dataset_pathdataset.csv"
)

OUTPUT_DIR = BASE_DIR / "reviewer1_outputs"

# This copy can be loaded later by app.py.
ROOT_DEPLOYMENT_MODEL_PATH = BASE_DIR / "best_model.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility settings
# ============================================================

RANDOM_STATE = 42

# Test multiple k values requested by Reviewer #1.
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]

# Five-fold stratified cross-validation on the training set.
CV_FOLDS = 5

# Keep this as 1 for stable Windows execution with the custom
# Average-distance KNN classifier.
N_JOBS = 1

# Mutual-information selector with a fixed random seed.
# Using functools.partial makes the saved pipeline loadable
# later from app.py.
MUTUAL_INFORMATION_SCORER = partial(
    mutual_info_classif,
    random_state=RANDOM_STATE,
)


# ============================================================
# Average-distance KNN classifier
# ============================================================

class AverageDistanceKNNClassifier(ClassifierMixin, BaseEstimator):
    """
    Average-distance KNN variant.

    For each test sample:
    1. Calculate Euclidean distances to the training samples.
    2. For each class, identify its k nearest training samples.
    3. Calculate the average class-specific distance.
    4. Predict the class with the smaller average distance.

    This implementation provides an explicit Average-distance
    KNN baseline for Reviewer #1 comment 4.
    """

    _estimator_type = "classifier"

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = check_X_y(
            X,
            y,
            dtype=float,
        )

        if self.n_neighbors < 1:
            raise ValueError(
                "n_neighbors must be at least 1."
            )

        self.X_train_ = X
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)

        if len(self.classes_) != 2:
            raise ValueError(
                "This implementation requires exactly two classes."
            )

        return self

    def predict_proba(self, X):
        check_is_fitted(
            self,
            attributes=[
                "X_train_",
                "y_train_",
                "classes_",
            ],
        )

        X = check_array(
            X,
            dtype=float,
        )

        distances = pairwise_distances(
            X,
            self.X_train_,
            metric="euclidean",
        )

        average_class_distances = np.zeros(
            shape=(len(X), len(self.classes_)),
            dtype=float,
        )

        for class_index, class_label in enumerate(self.classes_):
            class_mask = self.y_train_ == class_label
            class_distances = distances[:, class_mask]

            available_samples = class_distances.shape[1]

            if available_samples == 0:
                raise ValueError(
                    f"No training samples found for class {class_label}."
                )

            effective_k = min(
                self.n_neighbors,
                available_samples,
            )

            nearest_class_distances = np.partition(
                class_distances,
                kth=effective_k - 1,
                axis=1,
            )[:, :effective_k]

            average_class_distances[:, class_index] = (
                np.mean(
                    nearest_class_distances,
                    axis=1,
                )
            )

        # Smaller average distance should produce a larger score.
        inverse_distance_scores = (
            1.0
            / (average_class_distances + 1e-12)
        )

        probabilities = (
            inverse_distance_scores
            / inverse_distance_scores.sum(
                axis=1,
                keepdims=True,
            )
        )

        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)

        predicted_indices = np.argmax(
            probabilities,
            axis=1,
        )

        return self.classes_[predicted_indices]


# ============================================================
# Pipeline construction functions
# ============================================================

def preprocessing_steps(
    selected_feature_count=None,
    total_feature_count=34,
    add_scaler=True,
):
    """
    Build preprocessing steps.

    Feature selection is performed inside each training fold.
    This avoids information leakage from the test set.
    """

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
    ]

    if (
        selected_feature_count is not None
        and selected_feature_count < total_feature_count
    ):
        steps.append(
            (
                "selector",
                SelectKBest(
                    score_func=MUTUAL_INFORMATION_SCORER,
                    k=selected_feature_count,
                ),
            )
        )

    if add_scaler:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return steps


def standard_knn_pipeline(
    k,
    selected_feature_count=None,
    total_feature_count=34,
):
    steps = preprocessing_steps(
        selected_feature_count=selected_feature_count,
        total_feature_count=total_feature_count,
        add_scaler=True,
    )

    steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=k,
                weights="uniform",
                metric="minkowski",
                p=2,
            ),
        )
    )

    return Pipeline(steps)


def distance_weighted_knn_pipeline(
    k,
    selected_feature_count=None,
    total_feature_count=34,
):
    steps = preprocessing_steps(
        selected_feature_count=selected_feature_count,
        total_feature_count=total_feature_count,
        add_scaler=True,
    )

    steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=k,
                weights="distance",
                metric="minkowski",
                p=2,
            ),
        )
    )

    return Pipeline(steps)


def average_distance_knn_pipeline(
    k,
    selected_feature_count=None,
    total_feature_count=34,
):
    steps = preprocessing_steps(
        selected_feature_count=selected_feature_count,
        total_feature_count=total_feature_count,
        add_scaler=True,
    )

    steps.append(
        (
            "classifier",
            AverageDistanceKNNClassifier(
                n_neighbors=k,
            ),
        )
    )

    return Pipeline(steps)


def comparison_models(
    best_standard_k,
    best_weighted_k,
    best_average_k,
    selected_feature_count,
    total_feature_count,
):
    """
    Build all classifiers for Reviewer #1 comment 3.
    """

    models = {
        f"Standard KNN (k={best_standard_k})":
            standard_knn_pipeline(
                k=best_standard_k,
                selected_feature_count=selected_feature_count,
                total_feature_count=total_feature_count,
            ),

        f"Distance-weighted KNN (k={best_weighted_k})":
            distance_weighted_knn_pipeline(
                k=best_weighted_k,
                selected_feature_count=selected_feature_count,
                total_feature_count=total_feature_count,
            ),

        f"Average-distance KNN (k={best_average_k})":
            average_distance_knn_pipeline(
                k=best_average_k,
                selected_feature_count=selected_feature_count,
                total_feature_count=total_feature_count,
            ),

        "Random Forest":
            Pipeline(
                preprocessing_steps(
                    selected_feature_count=selected_feature_count,
                    total_feature_count=total_feature_count,
                    add_scaler=False,
                )
                + [
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),

        "Logistic Regression":
            Pipeline(
                preprocessing_steps(
                    selected_feature_count=selected_feature_count,
                    total_feature_count=total_feature_count,
                    add_scaler=True,
                )
                + [
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=500,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),

        "SVM with RBF kernel":
            Pipeline(
                preprocessing_steps(
                    selected_feature_count=selected_feature_count,
                    total_feature_count=total_feature_count,
                    add_scaler=True,
                )
                + [
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            probability=True,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),

        "MLP":
            Pipeline(
                preprocessing_steps(
                    selected_feature_count=selected_feature_count,
                    total_feature_count=total_feature_count,
                    add_scaler=True,
                )
                + [
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            solver="adam",
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),

        "XGBoost":
            Pipeline(
                preprocessing_steps(
                    selected_feature_count=selected_feature_count,
                    total_feature_count=total_feature_count,
                    add_scaler=False,
                )
                + [
                    (
                        "classifier",
                        xgb.XGBClassifier(
                            n_estimators=50,
                            learning_rate=0.1,
                            max_depth=3,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
    }

    return models


# ============================================================
# Evaluation functions
# ============================================================

SCORING = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "AUC": "roc_auc",
}


def cross_validation_summary(
    model_name,
    model,
    X_train,
    y_train,
    cv,
):
    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        scoring=SCORING,
        cv=cv,
        n_jobs=N_JOBS,
        error_score="raise",
    )

    return {
        "Model": model_name,
        "CV_Accuracy_Mean": float(
            np.mean(scores["test_Accuracy"])
        ),
        "CV_Accuracy_SD": float(
            np.std(scores["test_Accuracy"])
        ),
        "CV_Precision_Mean": float(
            np.mean(scores["test_Precision"])
        ),
        "CV_Recall_Mean": float(
            np.mean(scores["test_Recall"])
        ),
        "CV_F1_Mean": float(
            np.mean(scores["test_F1"])
        ),
        "CV_AUC_Mean": float(
            np.mean(scores["test_AUC"])
        ),
    }


def held_out_test_summary(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    fitted_model = clone(model)

    fitted_model.fit(
        X_train,
        y_train,
    )

    predictions = fitted_model.predict(
        X_test
    )

    if hasattr(
        fitted_model,
        "predict_proba",
    ):
        scores = fitted_model.predict_proba(
            X_test
        )[:, 1]

    elif hasattr(
        fitted_model,
        "decision_function",
    ):
        scores = fitted_model.decision_function(
            X_test
        )

    else:
        scores = predictions

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    return {
        "Model": model_name,
        "Test_Accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "Test_Precision": float(
            precision_score(
                y_test,
                predictions,
            )
        ),
        "Test_Recall": float(
            recall_score(
                y_test,
                predictions,
            )
        ),
        "Test_F1": float(
            f1_score(
                y_test,
                predictions,
            )
        ),
        "Test_AUC": float(
            roc_auc_score(
                y_test,
                scores,
            )
        ),
        "True_Normal": int(matrix[0, 0]),
        "False_Suspicious": int(matrix[0, 1]),
        "False_Normal": int(matrix[1, 0]),
        "True_Suspicious": int(matrix[1, 1]),
    }


def choose_best_k(
    tuning_dataframe,
    model_name,
):
    rows = tuning_dataframe[
        tuning_dataframe["Model"] == model_name
    ].copy()

    rows = rows.sort_values(
        by=[
            "CV_F1_Mean",
            "CV_Accuracy_Mean",
            "k",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    return int(rows.iloc[0]["k"])


# ============================================================
# Output helper functions
# ============================================================

def save_knn_tuning_plot(
    tuning_dataframe,
):
    plt.figure(
        figsize=(9, 5),
    )

    for model_name, group in tuning_dataframe.groupby(
        "Model"
    ):
        group = group.sort_values(
            "k"
        )

        plt.plot(
            group["k"],
            group["CV_F1_Mean"],
            marker="o",
            label=model_name,
        )

    plt.xlabel(
        "Number of neighbors (k)"
    )

    plt.ylabel(
        "Mean five-fold cross-validation F1-score"
    )

    plt.title(
        "KNN Variant Comparison Across k Values"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "comment4_knn_variant_k_tuning_plot.png",
        dpi=300,
    )

    plt.close()


def save_feature_selection_plot(
    feature_selection_dataframe,
):
    plot_dataframe = (
        feature_selection_dataframe
        .sort_values(
            "Number_of_Features"
        )
    )

    plt.figure(
        figsize=(8, 5),
    )

    plt.plot(
        plot_dataframe["Number_of_Features"],
        plot_dataframe["CV_F1_Mean"],
        marker="o",
    )

    plt.xlabel(
        "Number of selected pose features"
    )

    plt.ylabel(
        "Mean five-fold cross-validation F1-score"
    )

    plt.title(
        "Effect of Pose Feature Selection on Standard KNN"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "comment2_feature_selection_plot.png",
        dpi=300,
    )

    plt.close()


def save_model_comparison_plot(
    model_comparison_dataframe,
):
    plot_dataframe = (
        model_comparison_dataframe
        .sort_values(
            "Test_F1",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 6),
    )

    plt.barh(
        plot_dataframe["Model"],
        plot_dataframe["Test_F1"],
    )

    plt.xlabel(
        "Held-out test F1-score"
    )

    plt.ylabel(
        "Model"
    )

    plt.title(
        "Model Comparison for Suspicious-Activity Detection"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "comment3_model_comparison_plot.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Main execution
# ============================================================

def main():
    print("=" * 70)
    print("Reviewer #1 experiments started")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "\nDataset file not found:\n"
            f"{DATASET_PATH}\n\n"
            "Confirm that the CSV file exists inside:\n"
            r"F:\1. RESEARCH\Ashiq Sir\class - 2\dataset_path"
        )

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    # Remove accidental CSV index columns if present.
    dataframe = dataframe.loc[
        :,
        ~dataframe.columns.astype(str).str.startswith(
            "Unnamed:"
        ),
    ]

    required_columns = {
        "label",
        "image_name",
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        raise ValueError(
            "The CSV file must contain 'label' and 'image_name' columns."
        )

    X = dataframe.drop(
        columns=[
            "label",
            "image_name",
        ]
    ).copy()

    # Convert all features to numeric coordinates.
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Correct positive-class mapping:
    # suspicious behavior must be class 1.
    y = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .map(
            {
                "Normal": 0,
                "Suspicious": 1,
            }
        )
    )

    if y.isna().any():
        invalid_labels = (
            dataframe.loc[
                y.isna(),
                "label",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected label values detected: "
            f"{invalid_labels}. "
            "Expected only 'Normal' and 'Suspicious'."
        )

    total_feature_count = X.shape[1]

    print(f"Dataset path: {DATASET_PATH}")
    print(f"Total samples: {len(dataframe)}")
    print(
        "Total pose features: "
        f"{total_feature_count}"
    )

    print("\nClass distribution:")
    print(
        dataframe["label"]
        .value_counts()
    )

    if total_feature_count != 34:
        print(
            "\nWARNING: Expected 34 pose features "
            "(17 keypoints × 2 coordinates), but found "
            f"{total_feature_count}. "
            "The script will continue with the available features."
        )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    dataset_summary = {
        "Dataset_Path": str(
            DATASET_PATH
        ),
        "Total_Samples": int(
            len(dataframe)
        ),
        "Normal_Samples": int(
            (y == 0).sum()
        ),
        "Suspicious_Samples": int(
            (y == 1).sum()
        ),
        "Total_Pose_Features": int(
            total_feature_count
        ),
        "Class_Mapping": {
            "Normal": 0,
            "Suspicious": 1,
        },
        "Split_Method": (
            "80/20 stratified frame-level split "
            "with random_state=42"
        ),
        "Cross_Validation": (
            "Five-fold stratified cross-validation "
            "on the training partition"
        ),
    }

    with open(
        OUTPUT_DIR
        / "dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dataset_summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # ========================================================
    # Reviewer #1 Comment 4
    # Tune each KNN variant independently
    # ========================================================

    print("\n" + "=" * 70)
    print("Comment 4: KNN variant tuning")
    print("=" * 70)

    tuning_rows = []

    for k in K_VALUES:
        variants = {
            "Standard KNN":
                standard_knn_pipeline(
                    k=k,
                    total_feature_count=total_feature_count,
                ),

            "Distance-weighted KNN":
                distance_weighted_knn_pipeline(
                    k=k,
                    total_feature_count=total_feature_count,
                ),

            "Average-distance KNN":
                average_distance_knn_pipeline(
                    k=k,
                    total_feature_count=total_feature_count,
                ),
        }

        for variant_name, variant_model in variants.items():
            row = cross_validation_summary(
                model_name=variant_name,
                model=variant_model,
                X_train=X_train,
                y_train=y_train,
                cv=cv,
            )

            row["k"] = k

            tuning_rows.append(
                row
            )

            print(
                f"{variant_name:24s} | "
                f"k={k:2d} | "
                f"CV Accuracy={row['CV_Accuracy_Mean']:.6f} | "
                f"CV F1={row['CV_F1_Mean']:.6f} | "
                f"CV AUC={row['CV_AUC_Mean']:.6f}"
            )

    tuning_dataframe = pd.DataFrame(
        tuning_rows
    )

    tuning_dataframe.to_csv(
        OUTPUT_DIR
        / "comment4_knn_variant_k_tuning_results.csv",
        index=False,
    )

    save_knn_tuning_plot(
        tuning_dataframe
    )

    best_standard_k = choose_best_k(
        tuning_dataframe,
        "Standard KNN",
    )

    best_weighted_k = choose_best_k(
        tuning_dataframe,
        "Distance-weighted KNN",
    )

    best_average_k = choose_best_k(
        tuning_dataframe,
        "Average-distance KNN",
    )

    print("\nOptimal k values selected using CV F1-score:")
    print(
        f" - Standard KNN: k={best_standard_k}"
    )
    print(
        " - Distance-weighted KNN: "
        f"k={best_weighted_k}"
    )
    print(
        " - Average-distance KNN: "
        f"k={best_average_k}"
    )

    # ========================================================
    # Reviewer #1 Comment 2
    # Feature-selection analysis
    # ========================================================

    print("\n" + "=" * 70)
    print("Comment 2: Feature-selection experiment")
    print("=" * 70)

    feature_counts = [
        10,
        15,
        20,
        25,
        30,
        total_feature_count,
    ]

    feature_counts = sorted(
        set(
            count
            for count in feature_counts
            if 1 <= count <= total_feature_count
        )
    )

    feature_selection_rows = []

    for feature_count in feature_counts:
        if feature_count == total_feature_count:
            selected_feature_count = None
        else:
            selected_feature_count = feature_count

        model = standard_knn_pipeline(
            k=best_standard_k,
            selected_feature_count=selected_feature_count,
            total_feature_count=total_feature_count,
        )

        row = cross_validation_summary(
            model_name=(
                f"Standard KNN with "
                f"{feature_count} features"
            ),
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )

        row["Number_of_Features"] = (
            feature_count
        )

        feature_selection_rows.append(
            row
        )

        print(
            f"Features={feature_count:2d} | "
            f"CV Accuracy={row['CV_Accuracy_Mean']:.6f} | "
            f"CV F1={row['CV_F1_Mean']:.6f} | "
            f"CV AUC={row['CV_AUC_Mean']:.6f}"
        )

    feature_selection_dataframe = pd.DataFrame(
        feature_selection_rows
    )

    feature_selection_dataframe.to_csv(
        OUTPUT_DIR
        / "comment2_feature_selection_results.csv",
        index=False,
    )

    save_feature_selection_plot(
        feature_selection_dataframe
    )

    sorted_feature_results = (
        feature_selection_dataframe
        .sort_values(
            by=[
                "CV_F1_Mean",
                "CV_Accuracy_Mean",
                "Number_of_Features",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
    )

    best_feature_count = int(
        sorted_feature_results.iloc[0][
            "Number_of_Features"
        ]
    )

    print(
        "\nBest number of features selected using "
        "CV F1-score: "
        f"{best_feature_count}"
    )

    if best_feature_count == total_feature_count:
        selected_feature_count_for_pipeline = None
    else:
        selected_feature_count_for_pipeline = (
            best_feature_count
        )

    # --------------------------------------------------------
    # Save selected feature names
    # --------------------------------------------------------

    temporary_feature_selection_model = (
        standard_knn_pipeline(
            k=best_standard_k,
            selected_feature_count=(
                selected_feature_count_for_pipeline
            ),
            total_feature_count=total_feature_count,
        )
    )

    temporary_feature_selection_model.fit(
        X_train,
        y_train,
    )

    if (
        "selector"
        in temporary_feature_selection_model.named_steps
    ):
        selector = (
            temporary_feature_selection_model
            .named_steps["selector"]
        )

        selected_feature_names = (
            X.columns[
                selector.get_support()
            ]
            .tolist()
        )

    else:
        selected_feature_names = (
            X.columns.tolist()
        )

    pd.DataFrame(
        {
            "Selected_Feature":
                selected_feature_names
        }
    ).to_csv(
        OUTPUT_DIR
        / "comment2_selected_feature_names.csv",
        index=False,
    )

    # ========================================================
    # Reviewer #1 Comment 3
    # Model comparison including MLP
    # ========================================================

    print("\n" + "=" * 70)
    print("Comment 3: Multi-model comparison including MLP")
    print("=" * 70)

    models = comparison_models(
        best_standard_k=best_standard_k,
        best_weighted_k=best_weighted_k,
        best_average_k=best_average_k,
        selected_feature_count=(
            selected_feature_count_for_pipeline
        ),
        total_feature_count=total_feature_count,
    )

    model_comparison_rows = []

    for model_name, model in models.items():
        cv_row = cross_validation_summary(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )

        test_row = held_out_test_summary(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        combined_row = {
            **cv_row,
            **test_row,
        }

        model_comparison_rows.append(
            combined_row
        )

        print(
            f"{model_name:34s} | "
            f"Test Accuracy={combined_row['Test_Accuracy']:.6f} | "
            f"Test Precision={combined_row['Test_Precision']:.6f} | "
            f"Test Recall={combined_row['Test_Recall']:.6f} | "
            f"Test F1={combined_row['Test_F1']:.6f} | "
            f"Test AUC={combined_row['Test_AUC']:.6f}"
        )

    model_comparison_dataframe = pd.DataFrame(
        model_comparison_rows
    )

    model_comparison_dataframe.to_csv(
        OUTPUT_DIR
        / "comment3_model_comparison_results.csv",
        index=False,
    )

    save_model_comparison_plot(
        model_comparison_dataframe
    )

    # --------------------------------------------------------
    # Explicit hyperparameter table
    # --------------------------------------------------------

    hyperparameters = [
        {
            "Model": "Standard KNN",
            "Hyperparameters": (
                f"n_neighbors={best_standard_k}; "
                "weights=uniform; "
                "metric=Euclidean distance; "
                f"selected_features={best_feature_count}"
            ),
        },
        {
            "Model": "Distance-weighted KNN",
            "Hyperparameters": (
                f"n_neighbors={best_weighted_k}; "
                "weights=distance; "
                "metric=Euclidean distance; "
                f"selected_features={best_feature_count}"
            ),
        },
        {
            "Model": "Average-distance KNN",
            "Hyperparameters": (
                f"n_neighbors={best_average_k}; "
                "prediction=minimize average class-specific "
                "Euclidean distance; "
                f"selected_features={best_feature_count}"
            ),
        },
        {
            "Model": "Random Forest",
            "Hyperparameters": (
                "n_estimators=200; "
                "random_state=42; "
                f"selected_features={best_feature_count}"
            ),
        },
        {
            "Model": "Logistic Regression",
            "Hyperparameters": (
                "max_iter=500; "
                "standard scaling; "
                "random_state=42; "
                f"selected_features={best_feature_count}"
            ),
        },
        {
            "Model": "SVM with RBF kernel",
            "Hyperparameters": (
                "kernel=rbf; "
                "C=1.0; "
                "gamma=scale; "
                "probability=True; "
                f"selected_features={best_feature_count}"
            ),
        },
        {
            "Model": "MLP",
            "Hyperparameters": (
                "hidden_layer_sizes=(64, 32); "
                "activation=relu; "
                "solver=adam; "
                "max_iter=2000; "
                "random_state=42; "
                f"selected_features={best_feature_count}"
            ),
        },
        {
            "Model": "XGBoost",
            "Hyperparameters": (
                "n_estimators=50; "
                "learning_rate=0.1; "
                "max_depth=3; "
                "objective=binary:logistic; "
                "tree_method=hist; "
                "random_state=42; "
                f"selected_features={best_feature_count}"
            ),
        },
    ]

    pd.DataFrame(
        hyperparameters
    ).to_csv(
        OUTPUT_DIR
        / "comment3_hyperparameter_table.csv",
        index=False,
    )

    # ========================================================
    # Final standard KNN model for deployment
    # ========================================================

    print("\n" + "=" * 70)
    print("Final standard KNN model for deployment")
    print("=" * 70)

    final_standard_knn = standard_knn_pipeline(
        k=best_standard_k,
        selected_feature_count=(
            selected_feature_count_for_pipeline
        ),
        total_feature_count=total_feature_count,
    )

    final_standard_knn.fit(
        X_train,
        y_train,
    )

    final_predictions = (
        final_standard_knn.predict(
            X_test
        )
    )

    final_probabilities = (
        final_standard_knn.predict_proba(
            X_test
        )[:, 1]
    )

    final_matrix = confusion_matrix(
        y_test,
        final_predictions,
    )

    # --------------------------------------------------------
    # Save final confusion matrix in the previous paper style
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(6.0, 5.0)
    )

    image = axis.imshow(
        final_matrix,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_title(
        "KNN - Confusion Matrix"
    )

    tick_marks = np.arange(
        len(["0", "1"])
    )

    axis.set_xticks(
        tick_marks
    )

    axis.set_yticks(
        tick_marks
    )

    axis.set_xticklabels(
        ["0", "1"]
    )

    axis.set_yticklabels(
        ["0", "1"]
    )

    axis.set_ylabel(
        "Actual"
    )

    axis.set_xlabel(
        "Predicted"
    )

    threshold = (
        final_matrix.max()
        / 2.0
    )

    for row_index, column_index in np.ndindex(
        final_matrix.shape
    ):
        axis.text(
            column_index,
            row_index,
            format(
                final_matrix[
                    row_index,
                    column_index,
                ],
                "d",
            ),
            horizontalalignment="center",
            verticalalignment="center",
            color=(
                "white"
                if final_matrix[
                    row_index,
                    column_index
                ] > threshold
                else "black"
            ),
        )

    figure.tight_layout()

    figure.savefig(
        BASE_DIR
        / "final_knn_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIR
        / "final_knn_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # Save final ROC curve in the previous paper style
    # --------------------------------------------------------

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        final_probabilities,
    )

    final_roc_auc = auc(
        false_positive_rate,
        true_positive_rate,
    )

    figure, axis = plt.subplots(
        figsize=(6.0, 5.0)
    )

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"KNN (AUC = {final_roc_auc:.4f})",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        "--",
        color="tab:orange",
    )

    axis.set_xlim(
        [-0.01, 1.01]
    )

    axis.set_ylim(
        [-0.01, 1.05]
    )

    axis.set_xlabel(
        "False Positive Rate"
    )

    axis.set_ylabel(
        "True Positive Rate"
    )

    axis.set_title(
        "ROC Curve - KNN"
    )

    axis.legend(
        loc="lower right"
    )

    figure.tight_layout()

    figure.savefig(
        BASE_DIR
        / "final_knn_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIR
        / "final_knn_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    final_metrics = {
        "Model": "Standard KNN",
        "Best_k": int(
            best_standard_k
        ),
        "Selected_Number_of_Features": int(
            best_feature_count
        ),
        "Test_Accuracy": float(
            accuracy_score(
                y_test,
                final_predictions,
            )
        ),
        "Test_Precision": float(
            precision_score(
                y_test,
                final_predictions,
            )
        ),
        "Test_Recall": float(
            recall_score(
                y_test,
                final_predictions,
            )
        ),
        "Test_F1": float(
            f1_score(
                y_test,
                final_predictions,
            )
        ),
        "Test_AUC": float(
            roc_auc_score(
                y_test,
                final_probabilities,
            )
        ),
        "Confusion_Matrix": {
            "True_Normal": int(
                final_matrix[0, 0]
            ),
            "False_Suspicious": int(
                final_matrix[0, 1]
            ),
            "False_Normal": int(
                final_matrix[1, 0]
            ),
            "True_Suspicious": int(
                final_matrix[1, 1]
            ),
        },
        "Class_Mapping": {
            "Normal": 0,
            "Suspicious": 1,
        },
    }

    with open(
        OUTPUT_DIR
        / "final_standard_knn_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_metrics,
            file,
            indent=4,
        )

    pd.DataFrame(
        final_matrix,
        index=[
            "Actual_Normal",
            "Actual_Suspicious",
        ],
        columns=[
            "Predicted_Normal",
            "Predicted_Suspicious",
        ],
    ).to_csv(
        OUTPUT_DIR
        / "final_standard_knn_confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Refit deployment model using the full dataset
    # --------------------------------------------------------

    final_standard_knn.fit(
        X,
        y,
    )

    joblib.dump(
        final_standard_knn,
        OUTPUT_DIR
        / "best_knn_model.pkl",
    )

    joblib.dump(
        final_standard_knn,
        ROOT_DEPLOYMENT_MODEL_PATH,
    )

    deployment_metadata = {
        "Model": "Standard KNN",
        "Model_File": str(
            ROOT_DEPLOYMENT_MODEL_PATH
        ),
        "Best_k": int(
            best_standard_k
        ),
        "Total_Input_Features": int(
            total_feature_count
        ),
        "Selected_Number_of_Features": int(
            best_feature_count
        ),
        "Original_Feature_Columns_In_Required_Order":
            X.columns.tolist(),
        "Selected_Features":
            selected_feature_names,
        "Class_Mapping": {
            "Normal": 0,
            "Suspicious": 1,
        },
        "Important_App_Instruction": (
            "Pass all original input feature columns to the "
            "pipeline in the listed order. The pipeline performs "
            "feature selection and scaling internally."
        ),
    }

    with open(
        OUTPUT_DIR
        / "best_knn_model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            deployment_metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Final terminal summary
    # --------------------------------------------------------

    print("\nFinal standard KNN held-out test metrics:")
    print(
        json.dumps(
            final_metrics,
            indent=4,
        )
    )

    print("\nSaved deployment model:")
    print(
        ROOT_DEPLOYMENT_MODEL_PATH
    )

    print("\nSaved reviewer-response outputs:")
    print(
        OUTPUT_DIR
    )

    print("\nCreated files:")

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        print(
            f" - {path.name}"
        )

    print(
        "\nReviewer #1 experiments completed successfully."
    )


if __name__ == "__main__":
    main()