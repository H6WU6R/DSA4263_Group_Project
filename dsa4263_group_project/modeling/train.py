"""Training routines."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from .. import config
from .. import features


DEFAULT_MODEL_PATH = config.MODELS_DIR / "baseline_clf.joblib"


def train_baseline(data_path: Path, target_column: str, model_path: Path | None = None) -> Path:
    """Train a placeholder dummy classifier and persist it to *model_path*."""

    frame = pd.read_csv(data_path)
    feature_frame = features.add_default_features(frame.drop(columns=[target_column]))
    model = DummyClassifier(strategy="most_frequent")
    model.fit(feature_frame, frame[target_column])

    destination = model_path or DEFAULT_MODEL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)

    return destination
