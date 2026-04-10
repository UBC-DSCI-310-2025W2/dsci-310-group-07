import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_data(uci_id: int, path_to_save: str) -> pd.DataFrame:
    """
    Download a dataset from UCI repository and save it as a CSV file.

    Args:
        uci_id (int): ID of the dataset in the UCI repository.
        path_to_save (str): File path where the dataset will be saved.

    Returns:
        pd.DataFrame: The combined dataset (features + target).

    Raises:
        TypeError: If input types are incorrect
        ValueError: If uci_id is not positive integer
    """

    # -------------------
    # Input validation
    # -------------------
    if not isinstance(uci_id, int):
        raise TypeError("uci_id must be an integer")

    if uci_id <= 0:
        raise ValueError("uci_id must be positive")

    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string")

    # -------------------
    # Fetch data
    # -------------------
    dataset = fetch_ucirepo(id=uci_id)

    X = dataset.data.features
    y = dataset.data.targets

    # -------------------
    # Process data
    # -------------------
    df = pd.concat([X, y], axis=1)

    # -------------------
    # Save data
    # -------------------
    df.to_csv(path_to_save, index=False)

    return df