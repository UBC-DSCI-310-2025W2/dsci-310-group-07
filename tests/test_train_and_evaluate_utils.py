import os
import sys
import pandas as pd
import numpy as np
import pytest

from sklearn.model_selection import GridSearchCV
from unittest.mock import MagicMock
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train_and_evaluate_utils import (
    save_best_params, 
    save_test_pred,
    save_conf_mat
)


# ---------------------------
# Fixtures 
# ---------------------------
@pytest.fixture
def create_grid_search():
    """
    Fixture providing a mock GridSearchCV object with best_params_.
    """
    grid_search = MagicMock(spec=GridSearchCV)
    grid_search.best_params_ = {"max_depth": 20, "n_estimators": 100}
    return grid_search

@pytest.fixture
def create_test_pred():
    """
    Fixture providing a mock test predictions 
    """
    pred = np.array([5, 7])
    
    return pred

@pytest.fixture
def create_y_test():
    """
    Fixture providing a mock ground truth target values 
    """
    return pd.DataFrame({"quality": [7, 2]})

@pytest.fixture
def create_test_pred_single_class():
    """
    Fixture providing a mock test predictions with a single class value
    """
    pred = np.array([5, 5])
    
    return pred

@pytest.fixture
def create_y_test_single_class():
    """
    Fixture providing a mock ground truth target values with a single class value
    """
    return pd.DataFrame({"quality": [5, 5]})

# ---------------------------
# Tests for save_best_params
# ---------------------------
def test_save_best_params_create_file(create_grid_search, tmp_path):
    """
    Valid case 1: check that the CSV file is created.
    tmp_path automatically creates a folder that deletes itself after the test 
    """
    path = tmp_path/ "best_model_parameters.csv"
    save_best_params(create_grid_search, str(path))

    assert os.path.exists(path)

def test_save_best_params_correct_values(create_grid_search, tmp_path):
    """
    Valid case 2: check that saved parameter values match best parameter defined in fixture.
    """
    path = tmp_path/ "best_model_parameters.csv"
    save_best_params(create_grid_search, str(path))
    df = pd.read_csv(path)
    
    assert df["max_depth"][0] == 20
    assert df["n_estimators"][0] == 100
    
def test_save_best_params_overwrite_existing_csv_file(create_grid_search, tmp_path):
    """
    Edge case: check that the function can overwrite the existing csv file in the specified path
    """
    path = tmp_path/ "best_model_parameters.csv"

    with open(path, "w") as f:
        f.write("this must be overwritten")
        
    save_best_params(create_grid_search, str(path))
    df = pd.read_csv(path)
    
    assert "this must be overwritten" not in df.to_string()


def test_save_best_params_invalid_gridsearch_type():
    """
    Invalid case 1: check if the function throws TypeError when grid_search is not GridSearchCV object
    """
    with pytest.raises(TypeError):
        save_best_params("grid_search", "../output/prediction/best_model_parameters.csv")
    
def test_save_best_params_invalid_path_type(create_grid_search):
    """
    Invalid case 2: check if the function throws TypeError when path to save is not a string
    """
    with pytest.raises(TypeError):
        save_best_params(create_grid_search, 123622)
    
def test_save_best_params_empty_string(create_grid_search):
    """
    Invalid case 3: check if the function throws ValueError when path to save is not an empty string
    """
    with pytest.raises(ValueError):
        save_best_params(create_grid_search, "")
    
    
# ---------------------------
# Tests for save_test_pred
# ---------------------------
def test_save_test_pred_create_file(create_test_pred, tmp_path):
    """
    Valid case 1: check that the CSV file is created.
    """
    path = tmp_path/ "test_predictions.csv"

    save_test_pred(create_test_pred, str(path))
    
    assert os.path.exists(path)

def test_save_test_pred_correct_values(create_test_pred, tmp_path):
    """
    Valid case 2: check that test predictions match with the values defined in fixture.
    """
    path = tmp_path/ "test_predictions.csv"
    save_test_pred(create_test_pred, str(path))
    df = pd.read_csv(path)
    
    assert df["predicted_quality"].iloc[0] == 5
    assert df["predicted_quality"].iloc[1] == 7
    
def test_save_test_pred_overwrite_existing_csv_file(create_test_pred, tmp_path):
    """
    Edge case: check that the function can overwrite the existing csv file in the specified path
    """
    path = tmp_path/ "test_predictions.csv"
    
    with open(path, "w") as f:
        f.write("this must be overwritten")
        
    save_test_pred(create_test_pred, str(path))
    df = pd.read_csv(path)
    
    assert "this must be overwritten" not in df.to_string()

def test_save_test_pred_invalid_test_pred_type(tmp_path):
    """
    Invalid case 1: check if the function throws TypeError when test_pred is not numpy ndarray
    """
    with pytest.raises(TypeError):
        path = tmp_path/ "test_predictions.csv"
        save_best_params(812837.12312, str(path))
    
def test_save_test_pred_invalid_path_type(create_test_pred):
    """
    Invalid case 2: check if the function throws TypeError when path to save is not a string
    """
    with pytest.raises(TypeError):
        save_best_params(create_test_pred, 1.23212)
    
def test_save_test_pred_empty_string(create_test_pred):
    """
    Invalid case 3: check if the function throws ValueError when path to save is not an empty string
    """
    with pytest.raises(ValueError):
        save_test_pred(create_test_pred, "")


# ---------------------------
# Tests for save_test_pred
# ---------------------------
def test_save_conf_mat_create_file(create_y_test, create_test_pred, tmp_path):
    """
    Valid case 1: check that the confusion matrix is created
    """
    path = tmp_path/ "confusion_matrix.png"

    save_conf_mat(create_y_test, create_test_pred, str(path))
    
    assert os.path.exists(path)
    
def test_save_conf_mat_correct_image(create_y_test, create_test_pred, tmp_path):
    """
    Valid case 2: check that the confusion matrix png file is a valid image  
    """
    path = tmp_path/ "confusion_matrix.png"
    save_conf_mat(create_y_test, create_test_pred, str(path))
    
    conf_mat_img = Image.open(path)
    
    assert conf_mat_img.size[0] > 0 and conf_mat_img.size[1] > 0

def test_save_conf_mat_single_class(create_y_test_single_class, create_test_pred_single_class, tmp_path):
    """
    Edge case: check that the confusion matrix is created and is a valid image
    even when both y_test and test_pred only contain a single class
    """
    path = tmp_path/ "confusion_matrix.png"
    save_conf_mat(create_y_test_single_class, create_test_pred_single_class, str(path))
    conf_mat_img = Image.open(path)
    
    assert os.path.exists(path)
    assert conf_mat_img.size[0] > 0 and conf_mat_img.size[1] > 0
    
def test_save_conf_mat_invalid_y_test(create_test_pred, tmp_path):
    """
    Invalid case 1: check if the function throws TypeError when y_test is not pd.DataFrame
    """
    path = tmp_path/ "confusion_matrix.png"
    
    with pytest.raises(TypeError):
        save_conf_mat([1,2], create_test_pred, str(path))
        
def test_save_conf_mat_invalid_test_pred(create_y_test, tmp_path):
    """
    Invalid case 2: check if the function throws TypeError when test_pred is not np.ndarray
    """
    path = tmp_path/ "confusion_matrix.png"
    
    with pytest.raises(TypeError):
        save_conf_mat(create_y_test, [4,5], str(path))

def test_save_conf_mat_invalid_path_type(create_y_test):
    """
    Invalid case 3: check if the function throws TypeError when path is not a string
    """
    with pytest.raises(TypeError):
        save_conf_mat(create_y_test, [4,5], [11231231])

def test_save_conf_mat_empty_path(create_y_test, create_test_pred):
    """
    Invalid case 4: check if the function throws ValueError when path is not a string
    """
    with pytest.raises(ValueError):
        save_conf_mat(create_y_test, create_test_pred, "")
        
def test_save_conf_mat_mismatching_input_shape(create_y_test, tmp_path):
    """
    Invalid case 5: check if the function throws ValueError when y_test and test_pred 
    have different length
    """
    path = tmp_path/ "confusion_matrix.png"
    
    with pytest.raises(ValueError):
        save_conf_mat(create_y_test, np.array([1]), str(path))
    