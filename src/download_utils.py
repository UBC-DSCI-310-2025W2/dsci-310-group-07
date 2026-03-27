
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_data(uci_id: int, path_to_save: str) -> pd.DataFrame:
    """
    Download a dataset from the UCI repository and save it as a CSV file.

    Args:
        uci_id : int
            ID of the dataset in the UCI repository.
        path_to_save : str
            File path where the dataset will be saved.
        
    Returns:
        pd.DataFrame
            The downloaded dataset as a pandas DataFrame.

    Raises:
        TypeError: If uci_id is not an interger
        ValueError: If uci_id is not positive
    """
    if not isinstance(uci_id, int):
        raise TypeError("uci_id must be an integer")
    if uci_id <= 0:
        raise ValueError("uci_id must be positive")
    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string")

    dataset = fetch_ucirepo(id=uci_id)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    df.to_csv(path_to_save, index=False)

    return df