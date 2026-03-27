import pytest
import os
import pandas as pd
from src.eda import (
    save_feature_description,
    save_wine_quality_hist,
    save_feature_dist
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


# --- save_wine_quality_hist tests ---

def test_valid_input_creates_histogram_file(tmp_path):
    import pandas as pd
    from src.eda import save_wine_quality_hist

    file_path = tmp_path / "hist.png"

    y = pd.DataFrame([3, 4, 5, 5, 6])

    save_wine_quality_hist(y, str(file_path))

    assert file_path.exists()


def test_non_string_path_raises_type_error_hist():
    import pandas as pd
    from src.eda import save_wine_quality_hist

    y = pd.DataFrame([3, 4, 5])

    with pytest.raises(TypeError):
        save_wine_quality_hist(y, 123)


def test_empty_path_raises_value_error_hist():
    import pandas as pd
    from src.eda import save_wine_quality_hist

    y = pd.DataFrame([3, 4, 5])

    with pytest.raises(ValueError):
        save_wine_quality_hist(y, "")


def test_invalid_y_train_type_raises_type_error():
    from src.eda import save_wine_quality_hist

    with pytest.raises(TypeError):
        save_wine_quality_hist("not a dataframe", "file.png")


# --- save_feature_dist tests ---

def test_valid_input_creates_feature_distribution_file(tmp_path):
    import pandas as pd
    from src.eda import save_feature_dist

    file_path = tmp_path / "dist.png"

    X = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    save_feature_dist(X, str(file_path))

    assert file_path.exists()


def test_non_string_path_raises_type_error_dist():
    import pandas as pd
    from src.eda import save_feature_dist

    X = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(TypeError):
        save_feature_dist(X, 123)


def test_empty_path_raises_value_error_dist():
    import pandas as pd
    from src.eda import save_feature_dist

    X = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(ValueError):
        save_feature_dist(X, "")


def test_invalid_X_train_type_raises_type_error():
    from src.eda import save_feature_dist

    with pytest.raises(TypeError):
        save_feature_dist("not a dataframe", "file.png")