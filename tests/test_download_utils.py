import os
import pandas as pd
import pytest
from src.download_utils import download_data


# -------------------
# Valid case
# -------------------
def test_download_data_valid(tmp_path):
    file_path = tmp_path / "test.csv"

    df = download_data(45, str(file_path))  # 45 = valid UCI dataset

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert os.path.exists(file_path)


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
def test_download_data_file_content(tmp_path):
    file_path = tmp_path / "test.csv"

    df = download_data(45, str(file_path))
    loaded_df = pd.read_csv(file_path)

    assert df.shape == loaded_df.shape