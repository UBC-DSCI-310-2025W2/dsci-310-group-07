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


# -------------------------------
# save_feature_description tests 
# -------------------------------
def test_valid_path_creates_file():
    """
    Checks that save_feature_description successfully exports a default feature table to a CSV file when a valid path is provided.
    """
    file_path = "test_output.csv"

    save_feature_description(file_path)

    assert os.path.exists(file_path)

    os.remove(file_path)


def test_custom_table_data_creates_file():
    """
    Verifies that the function can handle custom list data and correctly write it to the specified CSV output.
    """
    file_path = "custom_output.csv"

    data = [["A", "Feature", "Type", "Unit", "Desc", "No"]]

    save_feature_description(file_path, table_data=data)

    assert os.path.exists(file_path)

    os.remove(file_path)


def test_non_string_path_raises_type_error():
    """
    Ensures a TypeError is raised if the file path argument is not a string.
    """
    with pytest.raises(TypeError):
        save_feature_description(123)


def test_empty_path_raises_value_error():
    """
    Ensures a ValueError is raised when the file path string is empty.
    """
    with pytest.raises(ValueError):
        save_feature_description("")


def test_invalid_table_data_type_raises_type_error():
    """
    Ensures a TypeError is raised when table_data is not a list.
    """
    with pytest.raises(TypeError):
        save_feature_description("file.csv", table_data="not a list")


# ---------------------------
# Pie chart tests
# ---------------------------

def test_pie_chart_valid(tmp_path):
    """
    Tests that a pie chart representing the data split (train/valid/test) is successfully generated and saved as an image file.
    """
    file_path = tmp_path / "pie.png"

    X_train = [1, 2, 3]
    X_valid = [4, 5]
    X_test = [6]

    save_data_split_pie_chart(X_train, X_valid, X_test, str(file_path))

    assert os.path.exists(file_path)


def test_pie_chart_invalid_type():
    """
    Ensures a TypeError is raised if data inputs are not list-like (e.g., an integer).
    """
    with pytest.raises(TypeError):
        save_data_split_pie_chart(123, [1], [1], "test.png")


def test_pie_chart_empty_data():
    """
    Ensures a ValueError is raised if the input data lists are empty, as a chart cannot be generated from no data.
    """
    with pytest.raises(ValueError):
        save_data_split_pie_chart([], [], [], "test.png")


def test_pie_chart_invalid_path():
    """
    Ensures a TypeError is raised if the output path is not a string.
    """
    with pytest.raises(TypeError):
        save_data_split_pie_chart([1], [1], [1], 123)


def test_pie_chart_empty_path():
    """
    Ensures a ValueError is raised if the output path string is empty.
    """
    with pytest.raises(ValueError):
        save_data_split_pie_chart([1], [1], [1], "")


# ------------------------------
# save_wine_quality_hist tests 
# ------------------------------

def test_valid_input_creates_histogram_file(tmp_path):
    """
    Confirms that the function takes a target DataFrame and saves a histogram plot of the wine quality distribution.
    """
    file_path = tmp_path / "hist.png"

    y = pd.DataFrame([3, 4, 5, 5, 6])

    save_wine_quality_hist(y, str(file_path))

    assert file_path.exists()


def test_non_string_path_raises_type_error_hist():
    """
    Verifies that a TypeError is raised if the input 'y' is not a pandas DataFrame or Series.
    """
    y = pd.DataFrame([3, 4, 5])

    with pytest.raises(TypeError):
        save_wine_quality_hist(y, 123)


def test_empty_path_raises_value_error_hist():
    """
    Ensures a ValueError is raised for histogram saving if the path string is empty.
    """
    y = pd.DataFrame([3, 4, 5])

    with pytest.raises(ValueError):
        save_wine_quality_hist(y, "")


def test_invalid_y_train_type_raises_type_error():
    """
    Ensures a TypeError is raised if the target data 'y' is not a pandas object.
    """
    with pytest.raises(TypeError):
        save_wine_quality_hist("not a dataframe", "file.png")


# ---------------------------
# save_feature_dist tests 
# ---------------------------  

def test_valid_input_creates_feature_distribution_file(tmp_path):
    """
    Checks that the function generates a multi-plot figure showing the distribution of all numerical features in the dataset.
    """
    file_path = tmp_path / "dist.png"

    X = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    save_feature_dist(X, str(file_path))

    assert file_path.exists()


def test_non_string_path_raises_type_error_dist():
    """
    Ensures a TypeError is raised for feature distribution if the path is not a string.
    """
    X = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(TypeError):
        save_feature_dist(X, 123)


def test_empty_path_raises_value_error_dist():
    """
    Ensures a ValueError is raised for feature distribution if the path string is empty.
    """
    X = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(ValueError):
        save_feature_dist(X, "")


def test_invalid_X_train_type_raises_type_error():
    """
    Ensures a TypeError is raised if the feature data 'X' is not a pandas DataFrame.
    """
    with pytest.raises(TypeError):
        save_feature_dist("not a dataframe", "file.png")


# ---------------------------
# Correlation matrix tests 
# --------------------------- 

def test_corr_mat_valid(tmp_path):
    """
    Validates that a correlation matrix heatmap is successfully calculated and saved to the designated path.
    """
    file_path = tmp_path / "corr.png"

    X = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    y = pd.Series([7, 8, 9])

    save_corr_mat(X, y, str(file_path))

    assert os.path.exists(file_path)


def test_corr_mat_invalid_X_type():
    """
    Ensures a TypeError is raised if 'X' is not a pandas DataFrame in correlation matrix calculation.
    """
    with pytest.raises(TypeError):
        save_corr_mat([1, 2, 3], pd.Series([1, 2, 3]), "test.png")


def test_corr_mat_invalid_y_type():
    """
    Ensures a TypeError is raised if 'y' is not a pandas Series in correlation matrix calculation.
    """
    with pytest.raises(TypeError):
        save_corr_mat(pd.DataFrame({"A": [1]}), [1], "test.png")


def test_corr_mat_empty_data():
    """
    Ensures a ValueError occurs if the input DataFrame is empty, preventing the calculation of a null correlation matrix.
    """
    with pytest.raises(ValueError):
        save_corr_mat(pd.DataFrame(), pd.Series(), "test.png")


def test_corr_mat_invalid_path():
    """
    Ensures a TypeError is raised if the correlation matrix output path is not a string.
    """
    with pytest.raises(TypeError):
        save_corr_mat(pd.DataFrame({"A": [1]}), pd.Series([1]), 123)


def test_corr_mat_empty_path():
    """
    Ensures a ValueError is raised if the correlation matrix output path is an empty string.
    """
    with pytest.raises(ValueError):
        save_corr_mat(pd.DataFrame({"A": [1]}), pd.Series([1]), "")