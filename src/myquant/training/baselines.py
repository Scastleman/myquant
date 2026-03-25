from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MajorityClassClassifier(BaseEstimator, ClassifierMixin):
    """Simple majority-class baseline with probability output."""

    def fit(self, X, y):  # noqa: N803 - sklearn interface
        counts = Counter(y)
        self.classes_ = np.array(sorted(counts))
        self.majority_class_ = max(counts, key=counts.get)
        self.majority_index_ = int(np.where(self.classes_ == self.majority_class_)[0][0])
        return self

    def predict(self, X):  # noqa: N803 - sklearn interface
        return np.repeat(self.majority_class_, len(X))

    def predict_proba(self, X):  # noqa: N803 - sklearn interface
        probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)
        probabilities[:, self.majority_index_] = 1.0
        return probabilities


def build_baseline_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Construct the first baseline model suite."""
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    hist_gb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=300,
        min_samples_leaf=20,
        random_state=random_state,
    )
    return {
        "majority": MajorityClassClassifier(),
        "logistic_regression": logistic,
        "hist_gradient_boosting": hist_gb,
    }
