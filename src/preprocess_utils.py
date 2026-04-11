import pandas as pd
from sklearn.model_selection import train_test_split


def drop_red_wine_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the red wine samples from the combined wine dataset.

    Assumes the first 1599 rows correspond to red wine samples and
    returns the remaining rows, which correspond to white wine samples.

    Parameters
    ----------
    df : pd.DataFrame
        Combined wine dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only white wine samples.

    Raises
    ------
    TypeError:
        If df is not a pandas DataFrame.
    ValueError:
        If df has fewer than 1599 rows.

    Examples
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'quality': range(2000)})
    >>> white_wine = drop_red_wine_samples(df)
    >>> len(white_wine)
    401
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if len(df) < 1599:
        raise ValueError("df must contain at least 1599 rows.")

    return df.iloc[1599:].reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """
    Split the cleaned dataset into training, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned wine dataset containing only white wine samples and a
        'quality' column as the target.

    Returns
    -------
    tuple
        X_train, X_valid, X_test, y_train, y_valid, y_test

    Raises
    ------
    TypeError:
        If df is not a pandas DataFrame.
    ValueError:
        If 'quality' is not a column in df.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': range(20), 'quality': range(20)})
    >>> X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    >>> X_train.shape
    (12, 1)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if "quality" not in df.columns:
        raise ValueError("df must contain a 'quality' column.")

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=123
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.21, shuffle=True, random_state=123
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test