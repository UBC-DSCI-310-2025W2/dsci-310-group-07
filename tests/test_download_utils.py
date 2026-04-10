import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.download_utils import download_data


# -------------------
# Valid case (Mocked)
# -------------------
@patch('src.download_utils.fetch_ucirepo')
def test_download_data_valid(mock_fetch, tmp_path):
    # Create a fake dataset object to return
    mock_dataset = MagicMock()
    mock_dataset.data.features = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    mock_dataset.data.targets = pd.DataFrame({'target': [0, 1]})
    mock_fetch.return_value = mock_dataset

    file_path = tmp_path / "test.csv"
    df = download_data(45, str(file_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)  # 2 rows, 3 columns (2 features + 1 target)
    assert os.path.exists(file_path)
    assert mock_fetch.called


# -------------------
# Type errors
# -------------------
def test_download_data_invalid_uci_id_type():
    with pytest.raises(TypeError):
        download_data("45", "test.csv")  # string instead of int


def test_download_data_invalid_path_type():
    with pytest.raises(TypeError):
        download_data(45, 123)  # path should be string


# -------------------
# Value errors
# -------------------
def test_download_data_invalid_uci_id_value():
    with pytest.raises(ValueError):
        download_data(-1, "test.csv")  # negative id


def test_download_data_zero_uci_id():
    with pytest.raises(ValueError):
        download_data(0, "test.csv")  # zero is invalid


# -------------------
# File saving check
# -------------------
@patch('src.download_utils.fetch_ucirepo')
def test_download_data_file_content(mock_fetch, tmp_path):
    mock_dataset = MagicMock()
    mock_dataset.data.features = pd.DataFrame({'a': [10]})
    mock_dataset.data.targets = pd.DataFrame({'b': [20]})
    mock_fetch.return_value = mock_dataset

    file_path = tmp_path / "test.csv"
    download_data(45, str(file_path))
    
    loaded_df = pd.read_csv(file_path)
    assert loaded_df.iloc[0]['a'] == 10
    assert loaded_df.iloc[0]['b'] == 20