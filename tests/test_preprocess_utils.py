import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess_utils import drop_red_wine_samples, split_data


# -------------------
# Tests for drop_red_wine_samples
# -------------------

def test_drop_red_wine_samples_basic():
    df = pd.DataFrame({
        "feature": range(1605),
        "quality": [5] * 1605
    })

    result = drop_red_wine_samples(df)

    assert len(result) == 6
    assert result.iloc[0]["feature"] == 1599


def test_drop_red_wine_samples_exactly_1599_rows():
    df = pd.DataFrame({
        "feature": range(1599),
        "quality": [5] * 1599
    })

    result = drop_red_wine_samples(df)

    assert result.empty


def test_drop_red_wine_samples_too_short():
    df = pd.DataFrame({
        "feature": range(100),
        "quality": [5] * 100
    })

    with pytest.raises(ValueError):
        drop_red_wine_samples(df)


def test_drop_red_wine_samples_wrong_input():
    with pytest.raises(TypeError):
        drop_red_wine_samples([1, 2, 3])


# -------------------
# Tests for split_data
# -------------------

def test_split_data_basic():
    df = pd.DataFrame({
        "alcohol": range(20),
        "sugar": range(100, 120),
        "quality": [5, 6] * 10
    })

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)

    assert len(X_train) + len(X_valid) + len(X_test) == 20
    assert len(y_train) + len(y_valid) + len(y_test) == 20
    assert "quality" not in X_train.columns


def test_split_data_returns_expected_number_of_outputs():
    df = pd.DataFrame({
        "alcohol": range(20),
        "sugar": range(100, 120),
        "quality": [5, 6] * 10
    })

    result = split_data(df)

    assert len(result) == 6


def test_split_data_missing_quality_column():
    df = pd.DataFrame({
        "alcohol": range(20),
        "sugar": range(100, 120)
    })

    with pytest.raises(ValueError):
        split_data(df)


def test_split_data_wrong_input():
    with pytest.raises(TypeError):
        split_data("not a dataframe")