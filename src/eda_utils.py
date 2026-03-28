import pandas as pd
import matplotlib.pyplot as plt


def save_feature_description(path_to_save, table_data=None):
    """
    Saves a feature description table as a CSV file.

    Args:
        path_to_save (str): Path where the CSV file will be saved.
        table_data (list of lists, optional): Custom table data to save.
            If None, a default wine dataset feature description is used.

    Returns:
        None

    Raises:
        TypeError: If path_to_save is not a string.
        ValueError: If path_to_save is empty.
        TypeError: If table_data is not a list when provided.
    """

    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string")

    if path_to_save.strip() == "":
        raise ValueError("path_to_save cannot be empty")

    if table_data is not None and not isinstance(table_data, list):
        raise TypeError("table_data must be a list of lists")

    if table_data is None:
        table_data = [
            ["fixed_acidity", "Feature", "Continuous", "g/dm³",
             "Fixed acids (primarily tartaric acid) that do not evaporate easily.", "No"],
            ["volatile_acidity", "Feature", "Continuous", "g/dm³",
             "Amount of acetic acid in wine; high levels can lead to an unpleasant vinegar taste.", "No"],
            ["citric_acid", "Feature", "Continuous", "g/dm³",
             "Citric acid content, which can add freshness and flavor to wine.", "No"],
            ["residual_sugar", "Feature", "Continuous", "g/dm³",
             "Amount of sugar remaining after fermentation stops.", "No"],
            ["chlorides", "Feature", "Continuous", "g/dm³",
             "Salt content in the wine.", "No"],
            ["free_sulfur_dioxide", "Feature", "Continuous", "mg/dm³",
             "Free form of sulfur dioxide that prevents microbial growth and oxidation.", "No"],
            ["total_sulfur_dioxide", "Feature", "Continuous", "mg/dm³",
             "Total amount of sulfur dioxide (free + bound forms).", "No"],
            ["density", "Feature", "Continuous", "g/cm³",
             "Density of the wine, influenced by alcohol and sugar content.", "No"],
            ["pH", "Feature", "Continuous", "–",
             "Acidity level of the wine; lower values indicate higher acidity.", "No"],
            ["sulphates", "Feature", "Continuous", "g/dm³",
             "Potassium sulphate concentration, contributing to sulfur dioxide levels and preservation.", "No"],
            ["alcohol", "Feature", "Continuous", "% (vol)",
             "Alcohol content of the wine.", "No"],
            ["quality", "Target", "Integer", "Score (0–10)",
             "Wine quality score based on expert sensory evaluation.", "No"]
        ]

    columns = [
        "Feature Name",
        "Role",
        "Type",
        "Units",
        "Description",
        "Missing Values"
    ]

    df = pd.DataFrame(table_data, columns=columns)
    df.to_csv(path_to_save, index=False)


def save_data_split_pie_chart(X_train, X_valid, X_test, path_to_save):
    """
    Save a pie chart showing the proportion of train, validation, and test data.

    Args:
        X_train, X_valid, X_test: Iterable datasets with length
        path_to_save (str): File path to save the image

    Returns:
        None

    Raises:
        TypeError: Invalid input types
        ValueError: If empty path to save figure or empty dataset
    """

    # --- Input validation ---
    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string")

    if path_to_save.strip() == "":
        raise ValueError("path_to_save cannot be empty")

    for dataset in [X_train, X_valid, X_test]:
        if not hasattr(dataset, "__len__"):
            raise TypeError("All datasets must have a length")

    total_size = len(X_train) + len(X_valid) + len(X_test)

    if total_size == 0:
        raise ValueError("Datasets cannot all be empty")

    # --- Compute correlation ---
    sizes = [
        len(X_train) / total_size,
        len(X_valid) / total_size,
        len(X_test) / total_size
    ]

    # --- Plot ---
    plt.figure(figsize=(7, 7))
    plt.pie(
        sizes,
        labels=["Train Set", "Validation Set", "Test Set"],
        autopct="%1.0f%%"
    )
    plt.title("Data Split Proportion")

    # --- Save ---
    plt.savefig(path_to_save)
    plt.close()


def save_wine_quality_hist(y_train, path_to_save):
    """
    Saves a histogram of wine quality distribution.

    Args:
        y_train (pd.DataFrame or pd.Series): Wine quality values.
        path_to_save (str): Path to save the plot.

    Returns:
        None
        
    Raises:
        TypeError: If path to save is not a string or y_train is not a Datafrane or Series
        ValueError: If path to save histogram is empty
    """

    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string")

    if path_to_save.strip() == "":
        raise ValueError("path_to_save cannot be empty")

    if not isinstance(y_train, (pd.DataFrame, pd.Series)):
        raise TypeError("y_train must be a pandas DataFrame or Series")

    plt.figure(figsize=(7, 7))
    y_train.squeeze().value_counts().sort_index().plot(kind="bar")
    plt.title("Distribution of Wine Quality")
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

    plt.savefig(path_to_save)
    plt.close()


def save_feature_dist(X_train, path_to_save):
    """
    Saves histograms of feature distributions.

    Args:
        X_train (pd.DataFrame): Feature dataset.
        path_to_save (str): Path to save the plot.

    Returns:
        None
        
    Raises:
        TypeError: If X_train is not a Dataframe or path to save is not a string
        ValueError: If path to save is an empty path
    """

    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string")

    if path_to_save.strip() == "":
        raise ValueError("path_to_save cannot be empty")

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")

    plt.figure()
    X_train.hist(bins=20, figsize=(12, 10))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()

    plt.savefig(path_to_save)
    plt.close()


def save_corr_mat(X_train, y_train, path_to_save):
    """
    Save a correlation matrix heatmap.

    Args:
        X_train (pd.DataFrame): Feature data
        y_train (pd.DataFrame or pd.Series): Target data
        path_to_save (str): File path to save image

    Returns:
        None

    Raises:
        TypeError: If X_train is not a Dataframe or y_train is empty
        ValueError: Empty data or invalid path
    """

    # --- Input validation ---
    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string")

    if path_to_save.strip() == "":
        raise ValueError("path_to_save cannot be empty")

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")

    if not isinstance(y_train, (pd.DataFrame, pd.Series)):
        raise TypeError("y_train must be a pandas DataFrame or Series")

    if X_train.empty or len(y_train) == 0:
        raise ValueError("Input data cannot be empty")

    # --- Compute correlation ---
    y_train = pd.DataFrame(y_train)  # ensure consistent format
    corr = pd.concat([X_train, y_train], axis=1).corr()

    # --- Plot ---
    plt.figure(figsize=(7, 7))
    plt.imshow(corr, interpolation="none")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()

    # --- Save ---
    plt.savefig(path_to_save)
    plt.close()