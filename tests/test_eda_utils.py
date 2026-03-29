import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eda_utils import (
    save_feature_description,
    save_data_split_pie_chart,
    save_wine_quality_hist,
    save_feature_dist,
    save_corr_mat
)


# --- save_feature_description tests ---
def test_valid_path_creates_file():
    file_path = "test_output.csv"

    save_feature_description(file_path)

    assert os.path.exists(file_path)

    os.remove(file_path)


def test_custom_table_data_creates_file():
    file_path = "custom_output.csv"

    data = [["A", "Feature", "Type", "Unit", "Desc", "No"]]

    save_feature_description(file_path, table_data=data)

    assert os.path.exists(file_path)

    os.remove(file_path)


def test_non_string_path_raises_type_error():
    with pytest.raises(TypeError):
        save_feature_description(123)


def test_empty_path_raises_value_error():
    with pytest.raises(ValueError):
        save_feature_description("")


def test_invalid_table_data_type_raises_type_error():
    with pytest.raises(TypeError):
        save_feature_description("file.csv", table_data="not a list")


# --- Pie chart tests ---

def test_pie_chart_valid(tmp_path):
    file_path = tmp_path / "pie.png"

    X_train = [1, 2, 3]
    X_valid = [4, 5]
    X_test = [6]

    save_data_split_pie_chart(X_train, X_valid, X_test, str(file_path))

    assert os.path.exists(file_path)


def test_pie_chart_invalid_type():
    with pytest.raises(TypeError):
        save_data_split_pie_chart(123, [1], [1], "test.png")


def test_pie_chart_empty_data():
    with pytest.raises(ValueError):
        save_data_split_pie_chart([], [], [], "test.png")


def test_pie_chart_invalid_path():
    with pytest.raises(TypeError):
        save_data_split_pie_chart([1], [1], [1], 123)


def test_pie_chart_empty_path():
    with pytest.raises(ValueError):
        save_data_split_pie_chart([1], [1], [1], "")


# --- save_wine_quality_hist tests ---

def test_valid_input_creates_histogram_file(tmp_path):
    file_path = tmp_path / "hist.png"

    y = pd.DataFrame([3, 4, 5, 5, 6])

    save_wine_quality_hist(y, str(file_path))

    assert file_path.exists()


def test_non_string_path_raises_type_error_hist():
    y = pd.DataFrame([3, 4, 5])

    with pytest.raises(TypeError):
        save_wine_quality_hist(y, 123)


def test_empty_path_raises_value_error_hist():
    y = pd.DataFrame([3, 4, 5])

    with pytest.raises(ValueError):
        save_wine_quality_hist(y, "")


def test_invalid_y_train_type_raises_type_error():
    with pytest.raises(TypeError):
        save_wine_quality_hist("not a dataframe", "file.png")


# --- save_feature_dist tests ---

def test_valid_input_creates_feature_distribution_file(tmp_path):
    file_path = tmp_path / "dist.png"

    X = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    save_feature_dist(X, str(file_path))

    assert file_path.exists()


def test_non_string_path_raises_type_error_dist():
    X = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(TypeError):
        save_feature_dist(X, 123)


def test_empty_path_raises_value_error_dist():
    X = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(ValueError):
        save_feature_dist(X, "")


def test_invalid_X_train_type_raises_type_error():
    with pytest.raises(TypeError):
        save_feature_dist("not a dataframe", "file.png")


# --- Correlation matrix tests ---

def test_corr_mat_valid(tmp_path):
    file_path = tmp_path / "corr.png"

    X = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    y = pd.Series([7, 8, 9])

    save_corr_mat(X, y, str(file_path))

    assert os.path.exists(file_path)


def test_corr_mat_invalid_X_type():
    with pytest.raises(TypeError):
        save_corr_mat([1, 2, 3], pd.Series([1, 2, 3]), "test.png")


def test_corr_mat_invalid_y_type():
    with pytest.raises(TypeError):
        save_corr_mat(pd.DataFrame({"A": [1]}), [1], "test.png")


def test_corr_mat_empty_data():
    with pytest.raises(ValueError):
        save_corr_mat(pd.DataFrame(), pd.Series(), "test.png")


def test_corr_mat_invalid_path():
    with pytest.raises(TypeError):
        save_corr_mat(pd.DataFrame({"A": [1]}), pd.Series([1]), 123)


def test_corr_mat_empty_path():
    with pytest.raises(ValueError):
        save_corr_mat(pd.DataFrame({"A": [1]}), pd.Series([1]), "")