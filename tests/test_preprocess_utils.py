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
    """
    Verifies that the function correctly identifies and removes the first 1599 rows (assumed to be red wine) and returns the remaining samples.
    """
    df = pd.DataFrame({
        "feature": range(1605),
        "quality": [5] * 1605
    })

    result = drop_red_wine_samples(df)

    assert len(result) == 6
    assert result.iloc[0]["feature"] == 1599


def test_drop_red_wine_samples_exactly_1599_rows():
    """
    Checks that the function returns an empty DataFrame when the input contains exactly 1599 rows, as all rows are dropped.
    """
    df = pd.DataFrame({
        "feature": range(1599),
        "quality": [5] * 1599
    })

    result = drop_red_wine_samples(df)

    assert result.empty


def test_drop_red_wine_samples_too_short():
    """
    Ensures that a ValueError is raised if the input DataFrame has fewer than 1599 rows, making it impossible to drop that many.
    """
    df = pd.DataFrame({
        "feature": range(100),
        "quality": [5] * 100
    })

    with pytest.raises(ValueError):
        drop_red_wine_samples(df)


def test_drop_red_wine_samples_wrong_input():
    """
    Ensures a TypeError is raised when the input provided is not a pandas DataFrame.
    """
    with pytest.raises(TypeError):
        drop_red_wine_samples([1, 2, 3])


# -------------------
# Tests for split_data
# -------------------

def test_split_data_basic():
    """
    Validates that the dataset is correctly partitioned into six components (X and y for train, validation, and test sets) and that the total number of rows is preserved.
    """
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
    """
    Confirms that the function consistently returns a tuple of 6 items representing the data splits.
    """
    df = pd.DataFrame({
        "alcohol": range(20),
        "sugar": range(100, 120),
        "quality": [5, 6] * 10
    })

    result = split_data(df)

    assert len(result) == 6


def test_split_data_missing_quality_column():
    """
    Ensures a ValueError is raised if the input DataFrame does not contain the 'quality' target column required for splitting.
    """
    df = pd.DataFrame({
        "alcohol": range(20),
        "sugar": range(100, 120)
    })

    with pytest.raises(ValueError):
        split_data(df)


def test_split_data_wrong_input():
    """
    Ensures a TypeError is raised when the input provided to split_data is not a pandas DataFrame.
    """
    with pytest.raises(TypeError):
        split_data("not a dataframe")