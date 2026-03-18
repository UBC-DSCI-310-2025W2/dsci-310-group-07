import argparse
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


'''
function to save best model parameters fround from grid search into csv file
'''
def save_best_params(grid_search, path_to_output):
    # save best parmams as a df 
    df = pd.DataFrame([grid_search.best_params_])
    
    # define path to save and save it in the specified csv file
    path_to_save = f'{path_to_output}/best_model_parameters.csv'
    df.to_csv(path_to_save, index=False)
    
    print(f'Best model parameters saved in {path_to_save}')


'''
function to save model predictions on test data
'''
def save_test_pred(pred, path_to_output):
    # save best parmams as a df 
    df = pd.DataFrame(pred, columns=['predicted_quality'])
    
    # define path to save and save it in the specified csv file
    path_to_save = f'{path_to_output}/test_prediction.csv'
    df.to_csv(path_to_save, index=False)
    
    print(f'Test prediction saved in {path_to_save}')
    
    
'''
function to construct confusion matrix on test data and save it in the specified path
'''
def save_conf_mat(y_test, pred, path_to_output):
    # manually specify labels to avoid removig labels with no samples
    labels = [i for i in range(11)]
    
    # define conf mat 
    cm = confusion_matrix(y_test, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Figure 5: Test Data Confusion Matrix")
    
    # save plot
    path_to_save = f'{path_to_output}/confusion_matrix.png'
    plt.savefig(path_to_save)
    
    print(f'Confusion matrix saved in {path_to_save}')
    

if __name__ == "__main__":
    # define parser and add args
    parser = argparse.ArgumentParser(description='Train and evaluate model and save evaluation results')
    parser.add_argument('--path_to_processed_data', type=str, help='Path to train, validation, and test data')
    parser.add_argument('--path_to_prediction_data', type=str, help='Path to save test prediction data')
    parser.add_argument('--path_to_output', type=str, help='Path to save best model parameters and evaluation plot and table')
    
    # parse args
    args = parser.parse_args()
    
    # load data
    X_train = pd.read_csv(f'{args.path_to_processed_data}/train_data/X_train.csv')
    y_train = pd.read_csv(f'{args.path_to_processed_data}/train_data/y_train.csv')

    X_valid = pd.read_csv(f'{args.path_to_processed_data}/valid_data/X_valid.csv')
    y_valid = pd.read_csv(f'{args.path_to_processed_data}/valid_data/y_valid.csv')
    
    X_test = pd.read_csv(f'{args.path_to_processed_data}/test_data/X_test.csv')
    y_test = pd.read_csv(f'{args.path_to_processed_data}/test_data/y_test.csv')
    
    # define random forest model
    rf_model = RandomForestClassifier(random_state=123)
    
    # define param grid for gird search
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 400],
        'max_depth': [2, 10, 20, 30, 40, 50]
    }
    
    # define and fit grid search
    grid_search = GridSearchCV(rf_model, param_grid, cv=2, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train.values.ravel())

    # save best params
    save_best_params(grid_search, args.path_to_output)
    
    # retrieve best performing model
    best_model = grid_search.best_estimator_
    
    # predict on test data
    pred = best_model.predict(X_test)
    
    # save test predictions
    save_test_pred(pred, args.path_to_prediction_data)
    
    # save confusion matrix
    save_conf_mat(y_test, pred, args.path_to_output)