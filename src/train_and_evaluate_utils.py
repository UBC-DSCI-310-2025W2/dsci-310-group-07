import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_best_params(grid_search: GridSearchCV, path_to_save: str) -> None:
    """
    Save best model parameters found from grid search into a csv file

    Args:
        grid_search (sklearn.model_selection.GridSearchCV):
            Fitted grid search object containing best parameters found during cross-validation.
        path_to_save (str): File path where the best parameters will be saved.

    Returns:
        None

    Raises:
        TypeError: 
            If grid_search is not a GridSearchCV object or if path_to_save is not a string
        ValueError: 
            If path to save is an empty string.

    Examples:
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import SVC
    >>> import pandas as pd
    >>> import os
    >>> # Setup a simple grid search
    >>> X, y = [[1, 2], [3, 4]], [0, 1]
    >>> grid = GridSearchCV(SVC(), {'C': [1, 10]}).fit(X, y)
    >>> save_best_params(grid, "results/best_params.csv")
    >>> os.path.exists("results/best_params.csv")
    True
    """
    if not isinstance(grid_search, GridSearchCV):
        raise TypeError("gird_search must be a sklearn.model_selection.GridSearchCV object.")
    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string.")
    if not path_to_save:
        raise ValueError("path_to_save must not be an empty string.")
    
    # save best parmams as a df 
    df = pd.DataFrame([grid_search.best_params_])
    
    # define path to save and save it in the specified csv file
    df.to_csv(path_to_save, index=False)
    
    print(f'Best model parameters saved in {path_to_save}')


def save_test_pred(test_pred: np.ndarray, path_to_save: str) -> None:
    """
    Save test prediction results into a csv file

    Args:
        test_pred (np.ndarray):
            Model prediction results on test data.
        path_to_save (str): 
            File path where the test predictions will be saved.

    Returns:
        None

    Raises:
        TypeError: 
            If test_pred is not a numpy array or path to save is not a string
        ValueError: 
            If path to save is an empty string.

    Examples:
    >>> import numpy as np
    >>> import os
    >>> preds = np.array([5, 6, 7])
    >>> save_test_pred(preds, "results/test_predictions.csv")
    >>> os.path.exists("results/test_predictions.csv")
    True
    """
    if not isinstance(test_pred, np.ndarray):
        raise TypeError("test_pred must be a numpy array.")
    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string.")
    if not path_to_save:
        raise ValueError("path_to_save must not be an empty string.")
    
    # save best parmams as a df 
    df = pd.DataFrame(test_pred, columns=['predicted_quality'])
    
    # define path to save and save it in the specified csv file
    df.to_csv(path_to_save, index=False)
    
    print(f'Test prediction saved in {path_to_save}')
    
    
def save_conf_mat(y_test: pd.DataFrame, test_pred: np.ndarray, path_to_save: str) -> None:
    """
    Construct confusion matrix using test predictions and ground truth target values, 
    and save it in the specified path.

    Args:
        y_test (pd.DataFrame):
            Ground truth target values of the test dataset.
        test_pred (np.ndarray): 
            Model prediction results on test data.
        path_to_save (str): 
            File path where the confusion matrix will be saved.
        

    Returns:
        None

    Raises:
        TypeError: 
            If y_test is not a pandas DataFrame, 
            test_pred is not anumpy array, 
            or path to save is not a string
        ValueError: 
            If y_test and test_pred have different length or path to save is an empty string.

    Examples:
    >>> import pandas as pd
    >>> import numpy as np
    >>> import os
    >>> y_true = pd.DataFrame({'quality': [5, 6, 7]})
    >>> y_pred = np.array([5, 5, 7])
    >>> save_conf_mat(y_true, y_pred, "results/cm_plot.png")
    >>> os.path.exists("results/cm_plot.png")
    True
    """
    if not isinstance(y_test, pd.DataFrame):
        raise TypeError("y_test must be a pd.DataFrame")
    if not isinstance(test_pred, np.ndarray):
        raise TypeError("test_pred must be a numpy ndarray")
    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string")
    if not path_to_save:
        raise ValueError("path_to_save must not be an empty string")
    if len(y_test) != len(test_pred):
        raise ValueError("y_test and test_pred must have the same length")
        
    # manually specify labels to avoid removig labels with no samples
    labels = [i for i in range(11)]
    
    # define conf mat 
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Test Data Confusion Matrix")
    
    # save plot
    plt.savefig(path_to_save)
    
    print(f'Confusion matrix saved in {path_to_save}')