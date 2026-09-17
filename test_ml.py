import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from ml.data import apply_label, process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "census.csv")


@pytest.fixture(scope="module")
def data():
    """Load the census data once for all tests."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def split(data):
    """Split the data the same way train_model.py does."""
    return train_test_split(data, test_size=0.20, random_state=42, stratify=data["salary"])


@pytest.fixture(scope="module")
def processed(split):
    """Process a sample of the train and test data so the tests run quickly."""
    train, test = split
    train = train.sample(n=2000, random_state=42)
    test = test.sample(n=500, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test, encoder, lb


@pytest.fixture(scope="module")
def model(processed):
    """Train one model on the sample data and share it across tests."""
    X_train, y_train, _, _, _, _ = processed
    return train_model(X_train, y_train)


def test_train_test_split_size(data, split):
    """
    The train and test datasets have the expected sizes (80/20), keep every
    column, and preserve the share of >50K labels.
    """
    train, test = split
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(0.20 * len(data), abs=1)
    assert list(train.columns) == list(data.columns)
    assert list(test.columns) == list(data.columns)
    overall_rate = (data["salary"] == ">50K").mean()
    assert (train["salary"] == ">50K").mean() == pytest.approx(overall_rate, abs=0.01)
    assert (test["salary"] == ">50K").mean() == pytest.approx(overall_rate, abs=0.01)


def test_process_data_output(processed):
    """
    process_data returns numpy arrays with matching row counts, the same number
    of columns for train and test, binary labels, and fitted encoders.
    """
    X_train, y_train, X_test, y_test, encoder, lb = processed
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert set(np.unique(y_train)) <= {0, 1}
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)


def test_train_model(processed, model):
    """
    train_model returns a fitted RandomForestClassifier.
    """
    X_train, _, _, _, _, _ = processed
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")
    assert model.n_features_in_ == X_train.shape[1]


def test_inference(processed, model):
    """
    inference returns a numpy array with one binary prediction per row.
    """
    _, _, X_test, _, _, _ = processed
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X_test.shape[0],)
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics():
    """
    compute_model_metrics returns the expected precision, recall, and F1 for
    known labels and predictions.
    """
    y = np.array([1, 1, 0, 0, 1, 0])
    preds = np.array([1, 0, 0, 1, 1, 0])
    # 2 true positives, 1 false positive, 1 false negative
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)

    precision, recall, fbeta = compute_model_metrics(np.array([1, 0]), np.array([1, 0]))
    assert (precision, recall, fbeta) == (1.0, 1.0, 1.0)


def test_apply_labels():
    """
    apply_label converts binary predictions into the salary strings.
    """
    assert apply_label(np.array([1])) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"


def test_save_and_load_model(processed, model, tmp_path):
    """
    A model saved with save_model and loaded with load_model makes the same
    predictions as the original.
    """
    _, _, X_test, _, _, _ = processed
    path = os.path.join(tmp_path, "model.pkl")
    save_model(model, path)
    assert os.path.exists(path)
    loaded = load_model(path)
    assert isinstance(loaded, RandomForestClassifier)
    np.testing.assert_array_equal(inference(model, X_test), inference(loaded, X_test))