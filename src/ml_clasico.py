"""Supervised classifiers (KNN, SVM, Naive Bayes) on the classic descriptors (T45, class 20).

They are an intermediate baseline between the hand-written rules (T40-T43) and the CNN (T60-T66): same
descriptor vector as the rules, but the decision is learnt from the labels. Run from the repository root after
python -m src.build_features:

    python -m src.ml_clasico

Hyperparameters are chosen only on train, with 5-fold cross-validation grouped by the time blocks of the split
(a random K-fold would put near-identical consecutive frames in train and validation). The unseen camera is never
used to choose anything. Output: results/clasico/clasificadores.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import data, metrics

OUT = data.RESULTS_DIR / "clasico" / "clasificadores.csv"
N_FOLDS = 5
RANDOM_STATE = 0

RAW_COLUMNS = [
    "lap_var", "gray_std", "entropy", "sat_mean", "hue_std", "skin_ratio", "orb_matches", "orb_inliers", "angle",
]
FEATURE_COLUMNS = [
    "lap_var", "gray_std", "entropy", "sat_mean", "hue_std", "skin_ratio", "orb_matches", "orb_inliers",
    "abs_angle", "angle_missing",
]
# Heavy-tailed counts and variances (0 to thousands): a log keeps one image from dominating the distances.
LOG_COLUMNS = ("lap_var", "orb_matches", "orb_inliers")

MODELS = {
    "knn": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [1, 3, 5, 7, 11, 15], "model__metric": ["euclidean", "manhattan"],
         "model__weights": ["uniform", "distance"]},
    ),
    "svm": (
        SVC(class_weight="balanced", random_state=RANDOM_STATE),
        {"model__kernel": ["linear", "rbf"], "model__C": [0.1, 1, 10, 100]},
    ),
    "naive_bayes": (GaussianNB(), {"model__var_smoothing": [1e-9, 1e-6, 1e-3]}),
}


def prepare_features(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix without missing values, in FEATURE_COLUMNS order.

    The angle is missing when there is no reliable rotation (too few RANSAC inliers): that fact is kept as the
    0/1 column angle_missing, and abs_angle is 0 in that case.
    """
    X = descriptors[RAW_COLUMNS[:-1]].astype(float).copy()
    for column in LOG_COLUMNS:
        X[column] = np.log1p(X[column])
    X["abs_angle"] = descriptors["angle"].abs().fillna(0.0)
    X["angle_missing"] = descriptors["angle"].isna().astype(int)
    return X[FEATURE_COLUMNS]


def grouped_folds(train: pd.DataFrame):
    """Cross-validation folds that keep every time block in a single fold, with balanced classes."""
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=False)
    return splitter.split(prepare_features(train), train["label"], groups=train["block"])


def fit(name: str, train: pd.DataFrame) -> GridSearchCV:
    """Choose the hyperparameters of one model by grouped cross-validation on train and refit on all of train."""
    if "set" in train and (train["set"] != "train").any():
        raise ValueError("fit only accepts rows of the train set.")
    estimator, grid = MODELS[name]
    pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
    search = GridSearchCV(pipeline, grid, cv=list(grouped_folds(train)), scoring="f1_macro", n_jobs=-1)
    return search.fit(prepare_features(train), train["label"])


def load_sets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train (without the 8 ORB references), val, test and unseen-camera descriptors, with their time blocks."""
    descriptors = pd.read_csv(data.RESULTS_DIR / "clasico" / "descriptores_ubicacion1.csv")
    blocks = data.load_split()[["path", "block"]]
    location1 = descriptors.merge(blocks, on="path", how="left")
    references = pd.read_csv(data.SPLITS_DIR / "referencias.csv")["path"]
    train = location1[(location1["set"] == "train") & ~location1["path"].isin(references)]
    val = location1[location1["set"] == "val"]
    test = location1[location1["set"] == "test"]
    unseen = pd.read_csv(data.RESULTS_DIR / "clasico" / "descriptores_camara_no_vista.csv")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), unseen


def _scores(model: GridSearchCV, X: pd.DataFrame) -> np.ndarray:
    """Sabotage score for the ROC curve (higher = more sabotage).

    KNN and Naive Bayes: 1 - P(normal). SVM: minus its one-vs-rest decision value for "normal" (distance to the
    boundary). SVM probabilities would need an internal calibration with random folds, the leakage the grouped
    cross-validation avoids, so they are not used.
    """
    normal = list(model.classes_).index(metrics.NORMAL)
    if hasattr(model.best_estimator_.named_steps["model"], "predict_proba"):
        return 1.0 - model.predict_proba(X)[:, normal]
    return -model.decision_function(X)[:, normal]


def main() -> None:
    train, val, test, unseen = load_sets()
    rows = []
    for name in MODELS:
        model = fit(name, train)
        for subset, frame in (("val", val), ("test", test), ("camara_no_vista", unseen)):
            X = prepare_features(frame)
            predicted = model.predict(X)
            rows.append(
                {
                    "modelo": name,
                    "hiperparametros": json.dumps({k.replace("model__", ""): v for k, v in model.best_params_.items()}),
                    "f1_cv_train": model.best_score_,
                    "conjunto": subset,
                    "n": len(frame),
                    **metrics.summary(frame["label"], predicted, _scores(model, X)),
                }
            )
        rows[-1]["ms_modelo_por_imagen"] = metrics.time_per_image(
            model.predict, [prepare_features(val.iloc[[i]]) for i in range(100)]
        )["mean_ms"]
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.drop(columns=["hiperparametros"]).round(3).to_string(index=False))
    for name, group in table.groupby("modelo", sort=False):
        print(f"{name}: {group['hiperparametros'].iloc[0]}")


if __name__ == "__main__":
    main()
